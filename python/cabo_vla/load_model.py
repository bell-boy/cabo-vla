import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file


def load_hf_vit(model, repo="google/vit-base-patch16-224"):
    # dict of name -> numpy array, e.g. "vit.encoder.layer.0.attention.attention.query.weight"
    sd = load_file(hf_hub_download(repo, "model.safetensors"))

    def put(var, arr):
        assert var[...].shape == arr.shape, (var[...].shape, arr.shape)
        var[...] = jnp.asarray(arr)

    def linear(mod, name):
        put(mod.kernel, sd[f"{name}.weight"].T)  # torch [out, in] -> flax [in, out]
        put(mod.bias, sd[f"{name}.bias"])

    def norm(mod, name):
        put(mod.scale, sd[f"{name}.weight"])
        put(mod.bias, sd[f"{name}.bias"])

    e = "vit.embeddings"
    put(model.patch_embed.kernel, sd[f"{e}.patch_embeddings.projection.weight"].transpose(2, 3, 1, 0))
    put(model.patch_embed.bias, sd[f"{e}.patch_embeddings.projection.bias"])
    put(model.cls_token, sd[f"{e}.cls_token"])
    put(model.pos_embed, sd[f"{e}.position_embeddings"])

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
    if model.head is not None:
        linear(model.head, "classifier")
