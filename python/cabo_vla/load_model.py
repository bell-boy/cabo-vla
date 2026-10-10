import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file


def load_hf_vit(model, repo="google/vit-base-patch16-224"):
    # dict of name -> numpy array, e.g. "vit.encoder.layer.0.attention.attention.query.weight"
    sd = load_file(hf_hub_download(repo, "model.safetensors"))

    def put(var, arr, key):
        if var[...].shape != arr.shape:
            raise RuntimeError(
                f"Shape mismatch loading '{key}' from {repo}: "
                f"model expects {var[...].shape}, checkpoint has {arr.shape} "
                "(after layout conversion)."
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
