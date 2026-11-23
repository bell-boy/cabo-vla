import json
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import load_file

from cabo_vla.models.qwen35vla.qwen35 import Qwen3_5Config, Qwen3_5Model


def load_hf_vit(model, repo="google/vit-base-patch16-224"):
    # dict of name -> numpy array, e.g. "vit.encoder.layer.0.attention.attention.query.weight"
    sd = load_file(hf_hub_download(repo, "model.safetensors"))

    def put(var, arr, key):
        if var[...].shape != arr.shape:
            raise RuntimeError(
                f"Shape mismatch loading '{key}' from {repo}: model expects {var[...].shape}, checkpoint has {arr.shape} (after layout conversion)."
            )
        var[...] = jnp.asarray(arr)

    def linear(mod, name):
        put(mod.kernel, sd[f"{name}.weight"].T, f"{name}.weight")  # torch [out, in] -> flax [in, out]
        put(mod.bias, sd[f"{name}.bias"], f"{name}.bias")

    def norm(mod, name):
        put(mod.scale, sd[f"{name}.weight"], f"{name}.weight")
        put(mod.bias, sd[f"{name}.bias"], f"{name}.bias")

    e = "vit.embeddings"

    def emb(var, name, perm=None):
        key = f"{e}.{name}"
        arr = sd[key] if perm is None else sd[key].transpose(perm)
        put(var, arr, key)

    emb(model.patch_embed.kernel, "patch_embeddings.projection.weight", (2, 3, 1, 0))
    emb(model.patch_embed.bias, "patch_embeddings.projection.bias")
    emb(model.cls_token, "cls_token")
    emb(model.pos_embed, "position_embeddings")

    # transformer blocks
    for i, blk in enumerate(model.blocks):
        p = f"vit.encoder.layer.{i}"
        norm(blk.ln1, f"{p}.layernorm_before")
        linear(blk.q, f"{p}.attention.attention.query")
        linear(blk.k, f"{p}.attention.attention.key")
        linear(blk.v, f"{p}.attention.attention.value")
        linear(blk.o, f"{p}.attention.output.dense")
        norm(blk.ln2, f"{p}.layernorm_after")
        linear(blk.fc1, f"{p}.intermediate.dense")
        linear(blk.fc2, f"{p}.output.dense")

    norm(model.out_norm, "vit.layernorm")

    # classifier is optional (skip it for the VLA)
    if model.projector_mlp is not None:
        linear(model.projector_mlp, "classifier")


def _qwen35_weight_name(path: tuple[object, ...]) -> str:
    *module, parameter = map(str, path)
    name = ".".join(module)
    if name == "visual.patch_embed":
        name += ".proj"
    elif name.startswith("visual.blocks."):
        prefix, layer = name.rsplit(".", 1)
        if layer in ("qkv", "proj"):
            name = f"{prefix}.attn.{layer}"
        elif layer in ("linear_fc1", "linear_fc2"):
            name = f"{prefix}.mlp.{layer}"
    elif name.startswith("visual.merger_"):
        layer = name.removeprefix("visual.merger_")
        name = f"visual.merger.{layer if layer == 'norm' else 'linear_' + layer}"
    if parameter in ("kernel", "embedding", "scale"):
        parameter = "weight"
    return f"model.{name}.{parameter}"


def load_hf_qwen35(config: Qwen3_5Config, checkpoint: Path) -> Qwen3_5Model:
    abstract_model = nnx.eval_shape(lambda: Qwen3_5Model(config, nnx.Rngs(0)))
    graphdef, state = nnx.split(abstract_model)
    params = {_qwen35_weight_name(path): (path, param) for path, param in state.flat_state()}

    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(checkpoint / "model.safetensors", framework="flax") as weights:
            weight_map = dict.fromkeys(weights.keys(), "model.safetensors")

    by_shard = defaultdict(list)
    for name in params:
        by_shard[weight_map[name]].append(name)

    loaded = {}
    for shard, names in by_shard.items():
        with safe_open(checkpoint / shard, framework="flax") as weights:
            for name in names:
                path, param = params[name]
                value = weights.get_tensor(name)
                if path[-1] == "kernel":
                    # PyTorch stores output channels first. Flax stores them last.
                    axes = (*range(2, value.ndim), 1, 0)
                    value = value.transpose(axes)
                if value.shape != param.shape:
                    raise ValueError(f"{name}: checkpoint shape {value.shape} does not match model shape {param.shape}")
                loaded[path] = param.replace(value=jnp.asarray(value, dtype=param.dtype))

    return nnx.merge(graphdef, nnx.State.from_flat_path(loaded))
