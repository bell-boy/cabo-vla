import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Float

from cabo_vla.config import ViTConfig
from cabo_vla.models.transformer import TransformerBlock


class ViT(nnx.Module):
    def __init__(self, cfg: ViTConfig, cls_out_params: int | None = 1000, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.blocks = nnx.List(
            [
                TransformerBlock(
                    cfg.dim,
                    cfg.heads,
                    cfg.mlp_dim,
                    rngs=rngs,
                    attention_bias=True,
                    norm_epsilon=1e-12,
                    approximate_gelu=False,
                )
                for _ in range(cfg.layers)
            ]
        )
        self.patch_embed = nnx.Conv(
            in_features=3,
            out_features=cfg.dim,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=cfg.patch_size,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(jnp.zeros((1, cfg.dim)))
        self.pos_embed = nnx.Param(jnp.zeros(((cfg.input_size // cfg.patch_size) ** 2 + 1, cfg.dim)))
        self.out_norm = nnx.LayerNorm(cfg.dim, epsilon=1e-12, rngs=rngs)
        # Optional ImageNet classifier weights for checkpoint parity checks.
        self.projector_mlp = nnx.Linear(in_features=cfg.dim, out_features=cls_out_params, rngs=rngs) if cls_out_params is not None else None

    def __call__(self, x: Float[jax.Array, "Channel Height Width"]) -> Float[jax.Array, "Patch Dim"]:
        patched = rearrange(self.patch_embed(rearrange(x, "c h w -> h w c")), "h w f -> (h w) f")
        with_cls = jnp.concatenate([self.cls_token[...], patched], axis=0)
        x = with_cls + self.pos_embed[...]
        mask = jnp.ones((x.shape[0], x.shape[0]), dtype=jnp.bool)
        for block in self.blocks:
            x, _ = block(x, mask=mask)
        x = self.out_norm(x)
        return x[1:]  # ignore the cls token
