import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Bool, Float

from cabo_vla.models.attention import attention

KVCacheType = tuple[Float[jax.Array, "Seq Head Dim"], Float[jax.Array, "Seq Head Dim"]]


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        *,
        rngs: nnx.Rngs,
        attention_bias: bool = False,
        norm_epsilon: float = 1e-6,
        approximate_gelu: bool = True,
    ):
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model.")
        d_head = d_model // n_heads

        self._W_q = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=attention_bias,
            rngs=rngs,
        )
        self._W_k = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=attention_bias,
            rngs=rngs,
        )
        self._W_v = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=attention_bias,
            rngs=rngs,
        )
        self._W_o = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=attention_bias,
            rngs=rngs,
        )
        self._d_model = d_model
        self._d_head = d_head
        self._n_heads = n_heads
        self._approximate_gelu = approximate_gelu

        self._ln1 = nnx.LayerNorm(d_model, epsilon=norm_epsilon, rngs=rngs)

        self._up_proj = nnx.Linear(in_features=d_model, out_features=d_hidden, rngs=rngs)

        self._down_proj = nnx.Linear(in_features=d_hidden, out_features=d_model, rngs=rngs)

        self._ln2 = nnx.LayerNorm(d_model, epsilon=norm_epsilon, rngs=rngs)

    def __call__(
        self,
        x: Float[jax.Array, "Seq Dim"],
        kv_cache: KVCacheType | None = None,
        save_cache: bool = False,
        mask: Bool[jax.Array, "T S"] | None = None,
    ) -> tuple[Float[jax.Array, "Seq Dim"], KVCacheType | None]:
        h = self._ln1(x)
        q, k, v = [W(h).reshape(-1, self._n_heads, self._d_head) for W in (self._W_q, self._W_k, self._W_v)]
        if kv_cache is not None:
            old_k, old_v = kv_cache
            k = jnp.concat((old_k, k), axis=0)
            v = jnp.concat((old_v, v), axis=0)

        if mask is None:
            seq_len = x.shape[0]
            old_seq_len = k.shape[0] - seq_len
            mask = jnp.tril(jnp.ones((seq_len, k.shape[0]), dtype=jnp.bool), k=old_seq_len)

        mha_output = attention(q[None, ...], k[None, ...], v[None, ...], mask=mask[None, None, :, :])[0].reshape(-1, self._d_model)
        x = x + self._W_o(mha_output)
        h = self._ln2(x)
        x = x + self._down_proj(jax.nn.gelu(self._up_proj(h), approximate=self._approximate_gelu))
        new_cache = (k, v) if save_cache else None
        return x, new_cache


class Transformer(nnx.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_hidden: int, rngs: nnx.Rngs):
        self._layers = nnx.List([TransformerBlock(d_model, n_heads, d_hidden, rngs=rngs) for _ in range(n_layers)])

    def __call__(
        self,
        x: Float[jax.Array, "Seq Dim"],
        kv_caches: list[KVCacheType] | None = None,
        save_caches: bool = False,
        mask: Bool[jax.Array, "T S"] | None = None,
    ) -> tuple[Float[jax.Array, "Seq Dim"], list[KVCacheType] | None]:
        if kv_caches is not None and len(kv_caches) != len(self._layers):
            raise ValueError("Expected one KV cache per transformer layer.")

        new_caches: list[KVCacheType] = []
        for i, layer in enumerate(self._layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=kv_cache, save_cache=save_caches, mask=mask)
            if new_cache is not None:
                new_caches.append(new_cache)

        return x, new_caches if save_caches else None
