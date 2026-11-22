from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx

from cabo_vla.data.dataloader import Batch
from cabo_vla.models.qwen35vla.qwen35 import Qwen3_5Config, Qwen3_5Model, VisionGrids


@dataclass(frozen=True)
class Qwen3_5VLAConfig:
    qwen_config: Qwen3_5Config
    action_layers: int = 6
    action_dim: int = 768
    action_heads: int = 12
    action_mlp_dim: int = 3072
    state_size: int = 32
    action_size: int = 32
    action_chunk_size: int = 50
    # Match the dataloader's pixel scaling and instruction padding.
    observation_range: tuple[float, float] = (0.0, 1.0)
    pad_token_id: int | None = None

    def __post_init__(self):
        if min(self.action_layers, self.action_dim, self.action_heads, self.action_mlp_dim, self.action_size, self.action_chunk_size) <= 0:
            raise ValueError("action head dimensions, depth, and chunk size must be positive")
        if self.action_dim % self.action_heads:
            raise ValueError("action_dim must be divisible by action_heads")
        if self.state_size <= 0:
            raise ValueError("state_size must be positive")
        lower, upper = self.observation_range
        if not (isfinite(lower) and isfinite(upper) and lower < upper):
            raise ValueError("observation_range must be finite and increasing")


class Qwen3_5VLAActionCache(NamedTuple):
    key_values: tuple[tuple[jax.Array, jax.Array], ...]
    attention_mask: jax.Array


class Qwen3_5ActionBlock(nnx.Module):
    def __init__(self, config: Qwen3_5VLAConfig, *, rngs: nnx.Rngs):
        dim = config.action_dim
        context_dim = config.qwen_config.text_config.hidden_size
        self.num_heads = config.action_heads
        self.self_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.self_qkv = nnx.Linear(dim, 3 * dim, rngs=rngs)
        self.self_out = nnx.Linear(dim, dim, rngs=rngs)
        self.cross_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.cross_query = nnx.Linear(dim, dim, rngs=rngs)
        self.cross_kv = nnx.Linear(context_dim, 2 * dim, rngs=rngs)
        self.cross_out = nnx.Linear(dim, dim, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp_up = nnx.Linear(dim, config.action_mlp_dim, rngs=rngs)
        self.mlp_down = nnx.Linear(config.action_mlp_dim, dim, rngs=rngs)

    def encode_context(self, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        k, v = rearrange(self.cross_kv(context), "b n (kv h d) -> kv b n h d", kv=2, h=self.num_heads)
        return k, v

    def __call__(self, x: jax.Array, key_values: tuple[jax.Array, jax.Array], context_mask: jax.Array) -> jax.Array:
        q, k, v = rearrange(self.self_qkv(self.self_norm(x)), "b n (qkv h d) -> qkv b n h d", qkv=3, h=self.num_heads)
        attended = jax.nn.dot_product_attention(q, k, v)
        x = x + self.self_out(rearrange(attended, "b n h d -> b n (h d)"))

        q = rearrange(self.cross_query(self.cross_norm(x)), "b n (h d) -> b n h d", h=self.num_heads)
        k, v = key_values
        attended = jax.nn.dot_product_attention(q, k.astype(q.dtype), v.astype(q.dtype), mask=context_mask[:, None, None, :])
        cross = self.cross_out(rearrange(attended, "b n h d -> b n (h d)"))
        x = x + jnp.where(context_mask.any(axis=1)[:, None, None], cross, 0)
        return x + self.mlp_down(jax.nn.gelu(self.mlp_up(self.mlp_norm(x)), approximate=False))


class Qwen3_5ActionHead(nnx.Module):
    def __init__(self, config: Qwen3_5VLAConfig, *, rngs: nnx.Rngs):
        dim = config.action_dim
        self.action_dim = dim
        self.action_size = config.action_size
        self.action_chunk_size = config.action_chunk_size
        self.action_in = nnx.Linear(config.action_size, dim, rngs=rngs)
        self.action_out = nnx.Linear(dim, config.action_size, rngs=rngs)
        self.pos_embed = nnx.Param(jnp.zeros((config.action_chunk_size, dim), dtype=jnp.float32))
        self.time_embed_up = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.time_embed_down = nnx.Linear(dim, dim, rngs=rngs)
        self.action_blocks = nnx.List([Qwen3_5ActionBlock(config, rngs=rngs) for _ in range(config.action_layers)])
        self.action_norm = nnx.LayerNorm(dim, rngs=rngs)

    def encode_context(self, context: jax.Array, mask: jax.Array) -> Qwen3_5VLAActionCache:
        context = jnp.where(mask[..., None], context, 0)
        return Qwen3_5VLAActionCache(tuple(block.encode_context(context) for block in self.action_blocks), mask)

    def __call__(self, cache: Qwen3_5VLAActionCache, action_t: jax.Array, traj_time: jax.Array) -> jax.Array:
        batch, horizon, action_size = action_t.shape
        if action_size != self.action_size or not 0 < horizon <= self.action_chunk_size:
            raise ValueError("action_t must have shape [batch, horizon, action_size] with horizon up to action_chunk_size")
        if traj_time.shape != (batch,):
            raise ValueError("traj_time must have shape [batch]")
        if cache.attention_mask.shape[0] != batch:
            raise ValueError("action batch size must match the encoded prefix")

        x = self.action_in(action_t)
        x = x + self.pos_embed[:horizon].astype(x.dtype)
        min_period, max_period = 0.004, 4.0
        periods = min_period * (max_period / min_period) ** jnp.linspace(0, 1, (self.action_dim + 1) // 2)
        angles = traj_time[:, None] * (2 * jnp.pi / periods)
        time = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)[:, : self.action_dim]
        time = jnp.broadcast_to(time.astype(x.dtype)[:, None, :], x.shape)
        x = self.time_embed_down(jax.nn.silu(self.time_embed_up(jnp.concatenate((x, time), axis=-1))))
        for block, key_values in zip(self.action_blocks, cache.key_values, strict=True):
            x = block(x, key_values, cache.attention_mask)
        return self.action_out(self.action_norm(x))


class Qwen3_5VLA(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5VLAConfig,
        rngs: nnx.Rngs | jax.Array,
        *,
        backbone: Qwen3_5Model | None = None,
    ):
        if not isinstance(rngs, nnx.Rngs):
            rngs = nnx.Rngs(rngs)

        context_dim = config.qwen_config.text_config.hidden_size
        if backbone is not None and backbone.language_model.embed_tokens.features != context_dim:
            raise ValueError("backbone hidden size must match qwen_config.text_config.hidden_size")
        self.backbone = Qwen3_5Model(config.qwen_config, rngs) if backbone is None else backbone
        self.action_head = Qwen3_5ActionHead(config, rngs=rngs)
        self.state_size = config.state_size
        self.state_proj = nnx.Linear(config.state_size, context_dim, rngs=rngs)
        self.observation_range = config.observation_range
        self.pad_token_id = config.pad_token_id

    def encode_prefix(
        self,
        input_ids: jax.Array,
        *,
        state: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        pixel_values: jax.Array | None = None,
        image_grid_thw: VisionGrids = (),
        pixel_values_videos: jax.Array | None = None,
        video_grid_thw: VisionGrids = (),
    ) -> Qwen3_5VLAActionCache:
        context, _ = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        mask = jnp.ones_like(input_ids, dtype=jnp.bool_) if attention_mask is None else attention_mask.astype(jnp.bool_)
        if state is not None:
            if state.shape != (input_ids.shape[0], self.state_size):
                raise ValueError("state must have shape [batch, state_size]")
            state_token = self.state_proj(state).astype(context.dtype)[:, None, :]
            context = jnp.concatenate((context, state_token), axis=1)
            mask = jnp.pad(mask, ((0, 0), (0, 1)), constant_values=True)
        return self.action_head.encode_context(context, mask)

    def encode_batch(self, batch: Batch[jax.Array]) -> Qwen3_5VLAActionCache:
        batch_size, cameras, channels, height, width = batch.observations.shape
        input_ids = batch.instruction
        mask = jnp.ones_like(input_ids, dtype=jnp.bool_) if self.pad_token_id is None else input_ids != self.pad_token_id
        input_ids = jnp.where(mask, input_ids, 0)
        state = batch.prio.reshape(batch_size, -1)

        pixel_values = None
        grids: VisionGrids = ()
        if cameras:
            vision = self.backbone.visual
            if channels != vision.in_channels or min(height, width) <= 0:
                raise ValueError("observations must have the vision encoder's channel count and positive spatial dimensions")
            patch, merge = vision.patch_size, vision.spatial_merge_size
            block = patch * merge
            lower, upper = self.observation_range
            images = (batch.observations.astype(jnp.float32) - lower) * (2 / (upper - lower)) - 1
            images = jnp.pad(images, ((0, 0), (0, 0), (0, 0), (0, -height % block), (0, -width % block)), mode="edge")

            # Pack each camera in spatial merge order, repeating its frame for the temporal projection.
            pixel_values = repeat(images, "b n c (h r p) (w s q) -> (b n h w r s) (c t p q)", r=merge, s=merge, p=patch, q=patch, t=vision.temporal_patch_size)
            grid_h, grid_w = images.shape[-2] // patch, images.shape[-1] // patch
            grids = ((1, grid_h, grid_w),) * (batch_size * cameras)
            image_ids = jnp.full((batch_size, cameras * (grid_h // merge) * (grid_w // merge)), self.backbone.image_token_id, dtype=jnp.int32)
            input_ids = jnp.concatenate((image_ids, input_ids), axis=1)
            mask = jnp.concatenate((jnp.ones_like(image_ids, dtype=jnp.bool_), mask), axis=1)

        if input_ids.shape[1] == 0:
            input_ids = jnp.zeros((batch_size, 1), dtype=jnp.int32)
            mask = jnp.zeros_like(input_ids, dtype=jnp.bool_)
        return self.encode_prefix(
            input_ids,
            state=state if state.shape[1] else None,
            attention_mask=mask,
            pixel_values=pixel_values,
            image_grid_thw=grids,
        )

    def run_action_head(
        self,
        cache: Qwen3_5VLAActionCache,
        action_t: jax.Array,
        traj_time: jax.Array,
    ) -> jax.Array:
        flat_actions = action_t.reshape(*action_t.shape[:2], -1)
        return self.action_head(cache, flat_actions, traj_time).reshape(action_t.shape)

    def __call__(
        self,
        batch: Batch[jax.Array],
        action_t: jax.Array | None = None,
        traj_time: jax.Array | None = None,
    ) -> jax.Array:
        """Predict velocity for the batch actions at flow time zero, or for supplied noisy actions and flow times."""
        if traj_time is None:
            if action_t is not None:
                raise ValueError("traj_time is required when supplying action_t")
            traj_time = jnp.zeros(batch.action.shape[0], dtype=jnp.float32)
        actions = batch.action if action_t is None else action_t
        return self.run_action_head(self.encode_batch(batch), actions, traj_time)
