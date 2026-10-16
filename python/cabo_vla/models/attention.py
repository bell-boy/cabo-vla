import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float


def attention(
    q: Float[jax.Array, "Batch T Heads Dim"],
    k: Float[jax.Array, "Batch S Heads Dim"],
    v: Float[jax.Array, "Batch S Heads Dim"],
    mask: Bool[jax.Array, "... T S"] | None = None,
    multiple: int = 8,
) -> Float[jax.Array, "Batch T Heads Dim"]:
    # cudnn flash attention for any sequence length. jax's cudnn backward pass rejects odd lengths (197 for the vit),
    # so pad q and kv up to a multiple and pass the true lengths, which makes cudnn mask out the padding
    if q.dtype not in (jnp.bfloat16, jnp.float16):  # cudnn is half precision only, so f32 (e.g. the hf port test) uses xla
        return jax.nn.dot_product_attention(q, k, v, mask=mask)
    B, T = q.shape[:2]
    S = k.shape[1]
    pad_t, pad_s = -T % multiple, -S % multiple

    def pad_seq(x, n):
        return jnp.pad(x, ((0, 0), (0, n), (0, 0), (0, 0)))

    if mask is not None:
        mask = jnp.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(0, pad_t), (0, pad_s)])
    out = jax.nn.dot_product_attention(
        pad_seq(q, pad_t),
        pad_seq(k, pad_s),
        pad_seq(v, pad_s),
        mask=mask,
        query_seq_lengths=jnp.full((B,), T, jnp.int32),
        key_value_seq_lengths=jnp.full((B,), S, jnp.int32),
        implementation="cudnn",
    )
    return out[:, :T]
