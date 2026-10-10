import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Float

from cabo_vla.config import ViTConfig


class ViTBlock(nnx.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.heads = heads
        self.ln1 = nnx.LayerNorm(dim, epsilon=1e-12, rngs=rngs)
        self.ln2 = nnx.LayerNorm(dim, epsilon=1e-12, rngs=rngs)
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.o = nnx.Linear(dim, dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)

    def __call__(self, x):
        h = self.ln1(x)
        q, k, v = (rearrange(proj(h), "b n (h d) -> b n h d", h=self.heads) for proj in (self.q, self.k, self.v))
        a = jax.nn.dot_product_attention(q, k, v)
        x = x + self.o(rearrange(a, "b n h d-> b n (h d)"))
        x = x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x)), approximate=False))
        return x


class ViT(nnx.Module):
    def __init__(self, cfg: ViTConfig, cls_out_params: int = 1000, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.blocks = nnx.List([ViTBlock(cfg.dim, cfg.heads, cfg.mlp_dim, rngs=rngs) for _ in range(cfg.layers)])
        self.patch_embed = nnx.Conv(
            in_features=3,
            out_features=cfg.dim,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=cfg.patch_size,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(jnp.zeros((1, 1, cfg.dim)))  # might be able to remove with model surgery, or maybe not
        self.pos_embed = nnx.Param(jnp.zeros((1, (cfg.input_size // cfg.patch_size) ** 2 + 1, cfg.dim)))  # with base config should be 1, 197, 768
        self.out_norm = nnx.LayerNorm(cfg.dim, epsilon=1e-12, rngs=rngs)
        self.projector_mlp = nnx.Linear(in_features=cfg.dim, out_features=cls_out_params, rngs=rngs)

    def __call__(self, x: Float[jax.Array, "Batch Channel Height Width"]):
        B = x.shape[0]
        patched = rearrange(self.patch_embed(rearrange(x, "b c h w-> b h w c")), "b h w f -> b (h w) f")
        with_cls = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, self.cfg.dim)), patched], axis=1)
        x = with_cls + self.pos_embed[...]
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x[:, 1:]  # ignore the cls token
