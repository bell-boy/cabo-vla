import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Float

from cabo_vla.data.dataloader import Batch
from cabo_vla.vit import ViT


class VLABlock(nnx.Module):
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

    def __call__(self, x, cache=None, mask=None):
        B, N, D = x.shape
        h = self.ln1(x)
        q, k, v = (rearrange(proj(h), "b n (h d) -> b n h d", h=self.heads) for proj in (self.q, self.k, self.v))
        new_kv = (k, v)
        if cache is not None:
            old_k, old_v = cache
            k = jnp.concatenate([old_k, k], axis=1)
            v = jnp.concatenate([old_v, v], axis=1)
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + self.o(rearrange(a, "b n h d-> b n (h d)"))
        x = x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x)), approximate=False))
        return x, new_kv


class VLA(nnx.Module):
    def __init__(
        self,
        patch_size: int,
        layers: int,
        dim: int,
        heads: int,
        mlp_dim: int,
        input_img_size: int = 224,
        input_state_size: int = 32,
        action_size: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        self.ViT = ViT(patch_size, layers, dim, heads, mlp_dim, input_img_size, rngs=rngs)
        self.blocks = nnx.List([VLABlock(dim, heads, mlp_dim, rngs=rngs) for _ in range(layers)])
        self.state_proj = nnx.Linear(input_state_size, dim, rngs=rngs)
        self.action_in = nnx.Linear(action_size, dim, rngs=rngs)
        self.action_out = nnx.Linear(dim, action_size, rngs=rngs)

    def encode_state_imgs(
        self, imgs: Float[jax.Array, "Batch Camera Channel Height Width"], state: Float[jax.Array, "Batch Horizon Action"], time: Float[jax.Array, "Batch"]
    ):  # pi 0 style state
        # the attention mask is structured in a way to allow for caching the encoding of the state/img prefix and reuse the same intermediate tokens for the action decoding
        B, CA, C, H, W = imgs.shape
        imgs = rearrange(imgs, "b c1 c2 h w -> (b c1) c2 h w")  # combine batch and camera dimensions
        vit_tokens = self.ViT(imgs)  # (b*c1, n, d)
        vit_tokens = rearrange(vit_tokens, "(b c1) n d-> b (c1 n) d", c1=CA)
        state = self.state_proj(state)
        x = jnp.concatenate([vit_tokens, state], axis=1)

        cache = []
        mask = None  #  unused for now, but when we start using pretrained siglip weights, we'll want to prevent the vla from attending to the state tokens (pi 0 style), or if we go pi 0.5 style with fast in the prompt, then we'll need to think more about how we want to structure the mask
        for block in self.blocks:
            x, kv = block(x, mask)
            cache.append(kv)
        return x, cache

    def run_action_head(self, cache, action_t: Float[jax.Array, "Batch Horizon Action"], traj_time: int):
        x = self.action_in(action_t)  # + self.embed_time(traj_time)
        for block, kv_cache in zip(self.blocks, cache):
            x, _ = block(x, kv_cache)
        return self.action_out(x)
