import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from cabo_vla.models.vit import ViT


def load_hf_vit(
    model: ViT,
    repo: str = "google/vit-base-patch16-224",
    *,
    revision: str | None = None,
) -> None:
    # dict of name -> numpy array, e.g. "vit.encoder.layer.0.attention.attention.query.weight"
    sd = load_file(hf_hub_download(repo, "model.safetensors", revision=revision))

    def put(var: nnx.Param | None, arr: np.ndarray, key: str) -> None:
        if var is None:
            raise ValueError(f"Missing model parameter for '{key}' from {repo}.")
        if var[...].shape != arr.shape:
            raise RuntimeError(
                f"Shape mismatch loading '{key}' from {repo}: model expects {var[...].shape}, checkpoint has {arr.shape} (after layout conversion)."
            )
        var[...] = jnp.asarray(arr)

    def linear(mod: nnx.Linear, name: str) -> None:
        put(mod.kernel, sd[f"{name}.weight"].T, f"{name}.weight")  # torch [out, in] -> flax [in, out]
        put(mod.bias, sd[f"{name}.bias"], f"{name}.bias")

    def norm(mod: nnx.LayerNorm, name: str) -> None:
        put(mod.scale, sd[f"{name}.weight"], f"{name}.weight")
        put(mod.bias, sd[f"{name}.bias"], f"{name}.bias")

    e = "vit.embeddings"

    def emb(var: nnx.Param | None, name: str, perm: tuple[int, ...] | None = None) -> None:
        key = f"{e}.{name}"
        arr = sd[key] if perm is None else sd[key].transpose(perm)
        put(var, arr, key)

    emb(model.patch_embed.kernel, "patch_embeddings.projection.weight", (2, 3, 1, 0))
    emb(model.patch_embed.bias, "patch_embeddings.projection.bias")
    put(model.cls_token, sd[f"{e}.cls_token"][0], f"{e}.cls_token")
    put(model.pos_embed, sd[f"{e}.position_embeddings"][0], f"{e}.position_embeddings")

    # transformer blocks
    for i, blk in enumerate(model.blocks):
        p = f"vit.encoder.layer.{i}"
        norm(blk._ln1, f"{p}.layernorm_before")
        linear(blk._W_q, f"{p}.attention.attention.query")
        linear(blk._W_k, f"{p}.attention.attention.key")
        linear(blk._W_v, f"{p}.attention.attention.value")
        linear(blk._W_o, f"{p}.attention.output.dense")
        norm(blk._ln2, f"{p}.layernorm_after")
        linear(blk._up_proj, f"{p}.intermediate.dense")
        linear(blk._down_proj, f"{p}.output.dense")

    norm(model.out_norm, "vit.layernorm")

    # classifier is optional (skip it for the VLA)
    if model.projector_mlp is not None:
        linear(model.projector_mlp, "classifier")
