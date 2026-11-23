from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby
from math import isqrt, prod
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx
from jax.core import Tracer


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    layer_types: list[str]
    intermediate_size: int = 12288
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)


@dataclass
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    rope_theta: float = 10_000.0


@dataclass
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057


VisionGrids = tuple[tuple[int, int, int], ...]

# Full attention stores keys and values. Linear attention stores convolution history and the recurrent state.
LayerCache = tuple[jax.Array, jax.Array]


def _vision_frame_groups(grid_thw: VisionGrids) -> Iterator[tuple[int, int, int]]:
    for (height, width), grids in groupby(grid_thw, key=lambda grid: grid[1:]):
        yield sum(temporal for temporal, _, _ in grids), height, width


class Qwen3_5Cache(NamedTuple):
    layers: tuple[LayerCache, ...]
    attention_mask: jax.Array


class Qwen3_5MultimodalCache(NamedTuple):
    text_cache: Qwen3_5Cache
    next_position: jax.Array


def _linear(config: Qwen3_5TextConfig, in_features: int, out_features: int, rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(in_features, out_features, use_bias=False, kernel_init=jax.nn.initializers.normal(config.initializer_range), rngs=rngs)


class Qwen3_5RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float):
        self.weight = nnx.Param(jnp.zeros(dim, dtype=jnp.float32))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        x32 = x.astype(jnp.float32)
        normalized = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
        return (normalized * (1.0 + self.weight[...].astype(jnp.float32))).astype(x.dtype)


class Qwen3_5TextRotaryEmbedding(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        self.theta = config.rope_theta
        self.sections = tuple(config.mrope_section)

    def __call__(self, position_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        inv_freq = self.theta ** (-jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        angles = position_ids[..., None].astype(jnp.float32) * inv_freq
        if position_ids.ndim == 3:
            temporal, height, width = angles
            height_bins = slice(1, 3 * self.sections[1], 3)
            width_bins = slice(2, 3 * self.sections[2], 3)
            angles = temporal.at[..., height_bins].set(height[..., height_bins])
            angles = angles.at[..., width_bins].set(width[..., width_bins])
        return jnp.cos(angles)[..., None, :], jnp.sin(angles)[..., None, :]


def _apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    half = cos.shape[-1]
    left, right, rest = jnp.split(x, (half, 2 * half), axis=-1)
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    return jnp.concatenate((left * cos - right * sin, right * cos + left * sin, rest), axis=-1)


class Qwen3_5Attention(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = config.head_dim
        query_dim = config.num_attention_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim
        self.q_proj = _linear(config, config.hidden_size, 2 * query_dim, rngs)
        self.k_proj = _linear(config, config.hidden_size, kv_dim, rngs)
        self.v_proj = _linear(config, config.hidden_size, kv_dim, rngs)
        self.o_proj = _linear(config, query_dim, config.hidden_size, rngs)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(
        self,
        x: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        mask: jax.Array,
        cache: LayerCache | None,
    ) -> tuple[jax.Array, LayerCache]:
        q, gate = jnp.split(rearrange(self.q_proj(x), "b t (h d) -> b t h d", d=2 * self.head_dim), 2, axis=-1)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d", d=self.head_dim)
        q = _apply_rotary(self.q_norm(q), *position_embeddings)
        k = _apply_rotary(self.k_norm(k), *position_embeddings)
        if cache is not None:
            old_k, old_v = cache
            k = jnp.concatenate((old_k, k), axis=1)
            v = jnp.concatenate((old_v, v), axis=1)

        out = jax.nn.dot_product_attention(q, k, v, mask=mask[:, None])
        # The softmax can return nonzero values for rows with no visible keys.
        out = jnp.where(mask.any(axis=-1)[..., None, None], out, 0)
        out = rearrange(out * jax.nn.sigmoid(gate), "b t h d -> b t (h d)")
        return self.o_proj(out), (k, v)


def _scan_gated_delta(state, q, k, v, beta, log_decay):
    def step(state, inputs):
        q, k, v, beta, decay = inputs
        state = state * decay[..., None, None]
        prediction = jnp.einsum("bhkv,bhk->bhv", state, k)
        delta = (v - prediction) * beta[..., None]
        state = state + k[..., None] * delta[..., None, :]
        return state, jnp.einsum("bhkv,bhk->bhv", state, q)

    inputs = (q, k, v.astype(jnp.float32), beta, jnp.exp(log_decay))
    state, out = jax.lax.scan(step, state, tuple(jnp.swapaxes(y, 0, 1) for y in inputs))
    return state, jnp.swapaxes(out, 0, 1)


def _delta_chunk(state, inputs):
    q, k, v, beta, log_decay = (x.astype(jnp.float32) for x in inputs)
    # Each row is a token. Entry (i,j) discounts token j's update through token i.
    prefix = jnp.cumsum(log_decay, axis=-1)
    causal = jnp.tri(q.shape[-2], dtype=bool)
    weights = jnp.exp(jnp.where(causal, prefix[..., :, None] - prefix[..., None, :], -jnp.inf))
    incoming = jnp.exp(prefix)[..., None]
    kt = k.swapaxes(-2, -1)
    with jax.default_matmul_precision("highest"):
        # Delta_i depends on all strictly earlier deltas via the decayed key Gram matrix.
        lower = jnp.tril((k @ kt) * weights * beta[..., None], -1)
        rhs = beta[..., None] * (v - incoming * (k @ state))
        delta = jax.lax.linalg.triangular_solve(lower, rhs, left_side=True, lower=True, unit_diagonal=True)
        out = incoming * (q @ state) + ((q @ kt) * weights) @ delta
        state = incoming[..., -1:, :] * state + kt @ (weights[..., -1, :, None] * delta)
    return state, out


@jax.custom_vjp
def _delta_chunks(state, inputs):
    return jax.lax.scan(_delta_chunk, state, inputs)


def _delta_chunks_fwd(state, inputs):
    return _delta_chunks(state, inputs), (state, inputs)


def _delta_chunks_bwd(residual, cotangent):
    state, inputs = residual

    # Replay the chunk scan to recover its boundary states, then rematerialize
    # the triangular solve one chunk at a time in the transposed scan.
    def replay(state, inputs):
        return jax.lax.scan(jax.checkpoint(_delta_chunk, prevent_cse=False), state, inputs)

    _, pullback = jax.vjp(replay, state, inputs)
    return pullback(cotangent)


_delta_chunks.defvjp(_delta_chunks_fwd, _delta_chunks_bwd)


def _chunk_gated_delta(state, q, k, v, beta, log_decay):
    length = q.shape[1]
    size = 64

    def chunks(x):
        padding = [(0, 0)] * x.ndim
        padding[1] = (0, -length % size)
        # Zero log decay and zero updates leave the tail state unchanged.
        return rearrange(jnp.pad(x, padding), "b (n t) h ... -> n b h t ...", t=size)

    state, out = _delta_chunks(state, tuple(chunks(x) for x in (q, k, v, beta, log_decay)))
    return state, rearrange(out, "n b h t d -> b (n t) h d")[:, :length]


class Qwen3_5GatedDeltaNet(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        if config.linear_num_value_heads % config.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_num_key_heads * self.head_k_dim
        self.num_v_heads = config.linear_num_value_heads
        self.value_groups = self.num_v_heads // config.linear_num_key_heads
        value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + value_dim
        self.conv_context = config.linear_conv_kernel_dim - 1

        self.in_proj_qkv = _linear(config, config.hidden_size, self.conv_dim, rngs)
        self.in_proj_z = _linear(config, config.hidden_size, value_dim, rngs)
        self.in_proj_b = _linear(config, config.hidden_size, self.num_v_heads, rngs)
        self.in_proj_a = _linear(config, config.hidden_size, self.num_v_heads, rngs)
        self.conv1d = nnx.Conv(
            self.conv_dim,
            self.conv_dim,
            config.linear_conv_kernel_dim,
            padding="VALID",
            feature_group_count=self.conv_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.dt_bias = nnx.Param(jnp.ones(self.num_v_heads, dtype=jnp.float32))
        self.A_log = nnx.Param(jnp.log(jax.random.uniform(rngs.params(), (self.num_v_heads,), dtype=jnp.float32, minval=0.01, maxval=16.0)))
        self.norm = nnx.RMSNorm(self.head_v_dim, epsilon=config.rms_norm_eps, rngs=rngs)
        self.out_proj = _linear(config, value_dim, config.hidden_size, rngs)

    def __call__(self, x: jax.Array, token_mask: jax.Array, cache: LayerCache | None) -> tuple[jax.Array, LayerCache]:
        batch, length, _ = x.shape
        x = jnp.where(token_mask[..., None], x, 0)
        qkv = self.in_proj_qkv(x)
        if cache is None:
            history = jnp.zeros((batch, self.conv_context, self.conv_dim), dtype=qkv.dtype)
            state = jnp.zeros((batch, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=jnp.float32)
        else:
            history, state = cache

        qkv = jnp.concatenate((history, qkv), axis=1)
        history = qkv[:, length:]
        q, k, v = jnp.split(jax.nn.silu(self.conv1d(qkv)), (self.key_dim, 2 * self.key_dim), axis=-1)
        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_k_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        def normalize(y: jax.Array) -> jax.Array:
            y = y.astype(jnp.float32)
            y = y * jax.lax.rsqrt(jnp.sum(jnp.square(y), axis=-1, keepdims=True) + 1e-6)
            return jnp.repeat(y, self.value_groups, axis=2)

        q = normalize(q) * self.head_k_dim**-0.5
        k = normalize(k)
        beta = jax.nn.sigmoid(self.in_proj_b(x)).astype(jnp.float32)
        rate = jnp.exp(self.A_log[...].astype(jnp.float32))
        dt = jax.nn.softplus(self.in_proj_a(x).astype(jnp.float32) + self.dt_bias[...])
        rule = _chunk_gated_delta if cache is None else _scan_gated_delta
        state, out = rule(state, q, k, v, beta, -rate * dt)
        out = out.astype(v.dtype)
        gate = rearrange(self.in_proj_z(x), "b t (h d) -> b t h d", d=self.head_v_dim)
        out = (self.norm(out).astype(out.dtype) * jax.nn.silu(gate.astype(jnp.float32))).astype(out.dtype)
        return self.out_proj(rearrange(out, "b t h d -> b t (h d)")), (history, state)


class Qwen3_5MLP(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.gate_proj = _linear(config, config.hidden_size, config.intermediate_size, rngs)
        self.up_proj = _linear(config, config.hidden_size, config.intermediate_size, rngs)
        self.down_proj = _linear(config, config.intermediate_size, config.hidden_size, rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, rngs=rngs)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, rngs=rngs)
        else:
            raise ValueError(f"Unknown Qwen3.5 layer type: {self.block_type}")
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3_5MLP(config, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        token_mask: jax.Array,
        causal_mask: jax.Array | None,
        cache: LayerCache | None,
    ) -> tuple[jax.Array, LayerCache]:
        h = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            h, cache = self.linear_attn(h, token_mask, cache)
        else:
            assert causal_mask is not None
            h, cache = self.self_attn(h, position_embeddings, causal_mask, cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class Qwen3_5TextModel(nnx.Module):
    def __init__(self, config: Qwen3_5TextConfig, rngs: nnx.Rngs | jax.Array):
        if not isinstance(rngs, nnx.Rngs):
            rngs = nnx.Rngs(rngs)
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, embedding_init=jax.nn.initializers.normal(config.initializer_range), rngs=rngs)
        self.layers = nnx.List([Qwen3_5DecoderLayer(config, i, rngs=rngs) for i in range(len(config.layer_types))])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.has_full_attention = "full_attention" in config.layer_types

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        *,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        cache: Qwen3_5Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[jax.Array, Qwen3_5Cache | None]:
        """Return hidden states and, when requested or supplied, an updated cache.

        Masks and position IDs describe the current tokens. Position IDs can be
        [batch, length] for text or [3, batch, length] for temporal, height and width.
        By default, positions start after the cached prefix.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Specify input_ids or inputs_embeds")
            x = self.embed_tokens(input_ids)
        else:
            if input_ids is not None:
                raise ValueError("Specify only one of input_ids and inputs_embeds")
            x = inputs_embeds

        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("Qwen3.5 requires at least one input token")
        token_mask = jnp.ones((batch, length), dtype=jnp.bool_) if attention_mask is None else attention_mask.astype(jnp.bool_)
        if token_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, current length]")

        past_length = cache.attention_mask.shape[1] if cache is not None else 0
        current_positions = past_length + jnp.arange(length)
        position_embeddings = self.rotary_emb(current_positions[None] if position_ids is None else position_ids)
        key_mask = jnp.concatenate((cache.attention_mask, token_mask), axis=1) if cache is not None else token_mask
        causal_mask = None
        if self.has_full_attention:
            causal_mask = (jnp.arange(past_length + length) <= current_positions[:, None]) & key_mask[:, None, :]

        layer_caches = cache.layers if cache is not None else (None,) * len(self.layers)
        next_layers = []
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            x, next_layer = layer(x, position_embeddings, token_mask, causal_mask, layer_cache)
            next_layers.append(next_layer)
        next_cache = Qwen3_5Cache(tuple(next_layers), key_mask) if use_cache or cache is not None else None
        return self.norm(x), next_cache


class Qwen3_5VisionBlock(nnx.Module):
    def __init__(self, config: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        dim = config.hidden_size
        self.num_heads = config.num_heads
        init = jax.nn.initializers.normal(config.initializer_range)
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, use_fast_variance=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, use_fast_variance=False, rngs=rngs)
        self.qkv = nnx.Linear(dim, 3 * dim, kernel_init=init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=init, rngs=rngs)
        self.linear_fc1 = nnx.Linear(dim, config.intermediate_size, kernel_init=init, rngs=rngs)
        self.linear_fc2 = nnx.Linear(config.intermediate_size, dim, kernel_init=init, rngs=rngs)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        q, k, v = rearrange(self.qkv(self.norm1(x)), "t n (qkv h d) -> qkv t n h d", qkv=3, h=self.num_heads)
        q = _apply_rotary(q.astype(jnp.float32), cos, sin).astype(q.dtype)
        k = _apply_rotary(k.astype(jnp.float32), cos, sin).astype(k.dtype)
        attended = jax.nn.dot_product_attention(q, k, v)
        x = x + self.proj(rearrange(attended, "t n h d -> t n (h d)"))
        return x + self.linear_fc2(jax.nn.gelu(self.linear_fc1(self.norm2(x)), approximate=True))


class Qwen3_5VisionModel(nnx.Module):
    def __init__(self, config: Qwen3_5VisionConfig, rngs: nnx.Rngs | jax.Array):
        if config.num_heads <= 0 or config.hidden_size % (4 * config.num_heads):
            raise ValueError("vision hidden_size must be divisible by four times num_heads")
        if min(config.patch_size, config.temporal_patch_size, config.spatial_merge_size, config.num_position_embeddings) <= 0:
            raise ValueError("vision patch, merge, and position grid sizes must be positive")
        self.grid_side = isqrt(config.num_position_embeddings)
        if self.grid_side**2 != config.num_position_embeddings:
            raise ValueError("vision num_position_embeddings must form a square grid")
        if not isinstance(rngs, nnx.Rngs):
            rngs = nnx.Rngs(rngs)

        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.rope_theta = config.rope_theta
        self.head_dim = config.hidden_size // config.num_heads
        patch_shape = (self.temporal_patch_size, self.patch_size, self.patch_size)
        init = jax.nn.initializers.normal(config.initializer_range)
        self.patch_embed = nnx.Conv(self.in_channels, self.hidden_size, patch_shape, padding="VALID", kernel_init=init, rngs=rngs)
        self.pos_embed = nnx.Embed(config.num_position_embeddings, self.hidden_size, embedding_init=init, rngs=rngs)
        self.blocks = nnx.List([Qwen3_5VisionBlock(config, rngs=rngs) for _ in range(config.depth)])
        merged_dim = self.hidden_size * self.spatial_merge_size**2
        self.merger_norm = nnx.LayerNorm(self.hidden_size, epsilon=1e-6, use_fast_variance=False, rngs=rngs)
        self.merger_fc1 = nnx.Linear(merged_dim, merged_dim, kernel_init=init, rngs=rngs)
        self.merger_fc2 = nnx.Linear(merged_dim, config.out_hidden_size, kernel_init=init, rngs=rngs)

    def _encode_grid(self, x: jax.Array, height: int, width: int) -> jax.Array:
        merge = self.spatial_merge_size
        row, col = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        row, col = (rearrange(p, "(h r) (w s) -> (h w r s)", r=merge, s=merge) for p in (row, col))

        # Interpolate the learned position table with aligned corners.
        y = row * (self.grid_side - 1) / max(height - 1, 1)
        x_pos = col * (self.grid_side - 1) / max(width - 1, 1)
        low_y, low_x = y.astype(jnp.int32), x_pos.astype(jnp.int32)
        high_y, high_x = jnp.minimum(low_y + 1, self.grid_side - 1), jnp.minimum(low_x + 1, self.grid_side - 1)
        y_frac, x_frac = y - low_y, x_pos - low_x
        indices = jnp.stack([r * self.grid_side + c for r in (low_y, high_y) for c in (low_x, high_x)], axis=-1)
        weights = jnp.stack([a * b for a in (1 - y_frac, y_frac) for b in (1 - x_frac, x_frac)], axis=-1)
        x = x + jnp.sum(self.pos_embed(indices) * weights[..., None], axis=-2).astype(x.dtype)

        spatial_dim = self.head_dim // 2
        freqs = self.rope_theta ** (-jnp.arange(0, spatial_dim, 2, dtype=jnp.float32) / spatial_dim)
        angles = jnp.concatenate((row[:, None] * freqs, col[:, None] * freqs), axis=-1)[:, None, :]
        cos, sin = jnp.cos(angles), jnp.sin(angles)
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.merger_norm(x).reshape(-1, self.hidden_size * merge**2)
        return self.merger_fc2(jax.nn.gelu(self.merger_fc1(x), approximate=False))

    def __call__(self, pixel_values: jax.Array, grid_thw: VisionGrids) -> jax.Array:
        """Encode flattened C,T,H,W patches ordered by frame and spatial merge block.

        Grids give the temporal, height, and width patch counts for each input.
        Keep the grid tuples static under JIT.
        """
        merge = self.spatial_merge_size
        if not grid_thw or any(min(grid) <= 0 or grid[1] % merge or grid[2] % merge for grid in grid_thw):
            raise ValueError("vision grids must be nonempty and spatially divisible by spatial_merge_size")
        patch_width = self.in_channels * self.temporal_patch_size * self.patch_size**2
        if pixel_values.shape != (sum(map(prod, grid_thw)), patch_width):
            raise ValueError("pixel_values shape does not match the vision grids and patch size")

        patches = rearrange(pixel_values, "n (c t h w) -> n t h w c", c=self.in_channels, t=self.temporal_patch_size, h=self.patch_size, w=self.patch_size)
        embedded = self.patch_embed(patches.astype(self.patch_embed.kernel.dtype)).reshape(-1, self.hidden_size)
        start = 0
        outputs = []
        for temporal, height, width in _vision_frame_groups(grid_thw):
            end = start + temporal * height * width
            # Frames are the attention batch, so they cannot attend to one another.
            frames = embedded[start:end].reshape(temporal, height * width, self.hidden_size)
            outputs.append(self._encode_grid(frames, height, width))
            start = end
        return jnp.concatenate(outputs, axis=0)


def _vision_token_layout(grid_thw: VisionGrids, merge: int) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    positions, steps, frame_sizes = [], [], []
    for temporal, height, width in _vision_frame_groups(grid_thw):
        rows, cols = height // merge, width // merge
        row, col = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
        frame_positions = jnp.stack((jnp.zeros_like(row), row, col), axis=-1).reshape(-1, 3)
        positions.append(jnp.tile(frame_positions, (temporal, 1)))
        frame_size = rows * cols
        # Advance the text position after the frame's last token.
        frame_steps = jnp.zeros(frame_size, dtype=jnp.int32).at[-1].set(max(rows, cols))
        steps.append(jnp.tile(frame_steps, temporal))
        frame_sizes.extend([frame_size] * temporal)
    return jnp.concatenate(positions), jnp.concatenate(steps), tuple(frame_sizes)


def _vision_token_indices(mask: jax.Array, frame_sizes: tuple[int, ...]) -> jax.Array:
    num_tokens = sum(frame_sizes)
    if num_tokens > mask.size:
        raise ValueError("vision features exceed the number of input tokens")
    if isinstance(mask, Tracer):
        return jnp.flatnonzero(mask, size=num_tokens, fill_value=mask.size)

    indices = np.flatnonzero(np.asarray(mask))
    if indices.size != num_tokens:
        raise ValueError(f"expected {num_tokens} vision placeholders, found {indices.size}")
    start = 0
    for size in frame_sizes:
        first, last = indices[start], indices[start + size - 1]
        if last - first != size - 1 or first // mask.shape[1] != last // mask.shape[1]:
            raise ValueError("each vision frame requires a contiguous placeholder span in one sequence")
        start += size
    return jnp.asarray(indices, dtype=jnp.int32)


class Qwen3_5Model(nnx.Module):
    def __init__(self, config: Qwen3_5Config, rngs: nnx.Rngs | jax.Array):
        if config.vision_config.out_hidden_size != config.text_config.hidden_size:
            raise ValueError("vision out_hidden_size must match text hidden_size")
        if config.image_token_id == config.video_token_id:
            raise ValueError("image_token_id and video_token_id must differ")
        if not isinstance(rngs, nnx.Rngs):
            rngs = nnx.Rngs(rngs)
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.visual = Qwen3_5VisionModel(config.vision_config, rngs=rngs)
        self.language_model = Qwen3_5TextModel(config.text_config, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        pixel_values: jax.Array | None = None,
        image_grid_thw: VisionGrids = (),
        pixel_values_videos: jax.Array | None = None,
        video_grid_thw: VisionGrids = (),
        attention_mask: jax.Array | None = None,
        cache: Qwen3_5MultimodalCache | None = None,
        use_cache: bool = False,
    ) -> tuple[jax.Array, Qwen3_5MultimodalCache | None]:
        """Replace image and video placeholders in batch order, then run the text model.

        Each image or video frame uses a contiguous placeholder span. Under JIT,
        keep the grids static and supply the matching placeholder spans.
        """
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("Qwen3.5 requires at least one input token")
        token_mask = jnp.ones_like(input_ids, dtype=jnp.bool_) if attention_mask is None else attention_mask.astype(jnp.bool_)
        if token_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match the shape of the current input_ids")

        embeddings = self.language_model.embed_tokens(input_ids)
        flat_embeddings = embeddings.reshape(batch * length, -1)
        steps = token_mask.reshape(-1).astype(jnp.int32)
        coords = jnp.zeros((batch * length, 3), dtype=jnp.int32)
        if any(temporal != 1 for temporal, _, _ in image_grid_thw):
            raise ValueError("each image grid must have a temporal patch count of one")
        for pixels, grids, token_id in (
            (pixel_values, image_grid_thw, self.image_token_id),
            (pixel_values_videos, video_grid_thw, self.video_token_id),
        ):
            if pixels is None and not grids:
                continue
            if pixels is None or not grids:
                raise ValueError("vision pixel values and grids must be supplied together")
            features = self.visual(pixels, grids)
            vision_coords, vision_steps, frame_sizes = _vision_token_layout(grids, self.visual.spatial_merge_size)
            indices = _vision_token_indices((input_ids == token_id) & token_mask, frame_sizes)
            flat_embeddings = flat_embeddings.at[indices].set(features.astype(embeddings.dtype))
            steps = steps.at[indices].set(vision_steps)
            coords = coords.at[indices].set(vision_coords)

        steps = steps.reshape(batch, length)
        start = cache.next_position[:, None] if cache is not None else 0
        end_positions = start + jnp.cumsum(steps, axis=1)
        positions = (end_positions - steps)[..., None] + coords.reshape(batch, length, 3)
        position_ids = jnp.moveaxis(jnp.where(token_mask[..., None], positions, 0), -1, 0)
        hidden_states, text_cache = self.language_model(
            inputs_embeds=flat_embeddings.reshape(embeddings.shape),
            attention_mask=token_mask,
            position_ids=position_ids,
            cache=cache.text_cache if cache is not None else None,
            use_cache=use_cache,
        )
        next_cache = Qwen3_5MultimodalCache(text_cache, end_positions[:, -1]) if text_cache is not None else None
        return hidden_states, next_cache
