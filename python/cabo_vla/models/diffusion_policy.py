import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float

from cabo_vla.models.transformer import Transformer
from cabo_vla.models.vit import ViT


class DiffusionPolicy(nnx.Module):
    def __init__(
        self,
        vision_encoder: ViT,
        action_dim: int,
        action_horizon: int,
        *,
        rngs: nnx.Rngs,
        num_cameras: int = 1,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_hidden: int = 1024,
    ):
        if min(action_dim, action_horizon, num_cameras, n_layers, d_hidden) <= 0:
            raise ValueError("Action, camera, and transformer sizes must be positive.")
        if d_model < 2 or d_model % 2:
            raise ValueError("d_model must be positive and even.")

        self.vision_encoder = vision_encoder
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._d_model = d_model
        self._observation_proj = nnx.Linear(vision_encoder.cfg.dim, d_model, rngs=rngs)
        self._camera_embed = nnx.Param(0.02 * jax.random.normal(rngs.params(), (num_cameras, 1, d_model)))
        self._action_embed = nnx.Param(0.02 * jax.random.normal(rngs.params(), (action_horizon, d_model)))
        self._action_proj = nnx.Linear(action_dim, d_model, rngs=rngs)
        self._time_up = nnx.Linear(d_model, d_hidden, rngs=rngs)
        self._time_down = nnx.Linear(d_hidden, d_model, rngs=rngs)
        self._transformer = Transformer(n_layers, d_model, n_heads, d_hidden, rngs=rngs)
        self._out_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self._action_out = nnx.Linear(d_model, action_dim, rngs=rngs)

    def encode_images(self, observations: Float[jax.Array, "Batch Camera Channel Height Width"]) -> Float[jax.Array, "Batch Camera Patch Dim"]:
        return jax.vmap(jax.vmap(self.vision_encoder))(observations)

    def _project_features(self, features: jax.Array) -> jax.Array:
        tokens = self._observation_proj(features) + self._camera_embed[...]
        return tokens.reshape(features.shape[0], -1, self._d_model)

    def _encode_observations(self, observations: jax.Array) -> jax.Array:
        return self._project_features(self.encode_images(observations))

    def _embed_actions(self, noisy_actions: jax.Array, time: jax.Array) -> jax.Array:
        batch_size = noisy_actions.shape[0]
        time = jnp.broadcast_to(time, (batch_size,))
        frequencies = jnp.exp(-jnp.log(10_000.0) * jnp.arange(self._d_model // 2) / max(self._d_model // 2 - 1, 1))
        angles = time[:, None] * frequencies[None, :] * (2 * jnp.pi)
        time_embed = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
        time_embed = self._time_down(jax.nn.silu(self._time_up(time_embed)))
        return self._action_proj(noisy_actions) + self._action_embed[...] + time_embed[:, None, :]

    def _predict_velocity(self, observation_tokens: jax.Array, noisy_actions: jax.Array, time: jax.Array) -> jax.Array:
        action_tokens = self._embed_actions(noisy_actions, time)
        tokens = jnp.concatenate((observation_tokens, action_tokens), axis=1)
        conditioning_length = observation_tokens.shape[1]
        # Conditioning stays independent of actions; actions attend to the full chunk.
        mask = jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
        mask = mask.at[:conditioning_length, conditioning_length:].set(False)
        hidden, _ = jax.vmap(lambda x: self._transformer(x, mask=mask))(tokens)
        return self._action_out(self._out_norm(hidden[:, -self._action_horizon :]))

    def _predict_cached_velocity(
        self,
        conditioning_cache: list[tuple[jax.Array, jax.Array]],
        noisy_actions: jax.Array,
        time: jax.Array,
    ) -> jax.Array:
        action_tokens = self._embed_actions(noisy_actions, time)
        conditioning_length = conditioning_cache[0][0].shape[1]
        mask = jnp.ones(
            (self._action_horizon, conditioning_length + self._action_horizon),
            dtype=jnp.bool_,
        )
        hidden, _ = jax.vmap(lambda x, cache: self._transformer(x, kv_caches=cache, mask=mask))(action_tokens, conditioning_cache)
        return self._action_out(self._out_norm(hidden))

    def __call__(
        self,
        observations: Float[jax.Array, "Batch Camera Channel Height Width"],
        noisy_actions: Float[jax.Array, "Batch Horizon Action"],
        time: Float[jax.Array, " Batch"] | Float[jax.Array, ""],
    ) -> Float[jax.Array, "Batch Horizon Action"]:
        return self._predict_velocity(self._encode_observations(observations), noisy_actions, time)

    def sample_actions(
        self,
        observations: Float[jax.Array, "Batch Camera Channel Height Width"],
        *,
        rng: jax.Array,
        num_steps: int = 10,
        noise: Float[jax.Array, "Batch Horizon Action"] | None = None,
    ) -> Float[jax.Array, "Batch Horizon Action"]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        observation_tokens = self._encode_observations(observations)
        conditioning_length = observation_tokens.shape[1]
        conditioning_mask = jnp.ones((conditioning_length, conditioning_length), dtype=jnp.bool_)
        _, conditioning_cache = jax.vmap(lambda x: self._transformer(x, save_caches=True, mask=conditioning_mask))(observation_tokens)
        assert conditioning_cache is not None
        shape = (observations.shape[0], self._action_horizon, self._action_dim)
        if noise is None:
            noise = jax.random.normal(rng, shape)
        noise = noise.astype(jnp.float32)
        dt = 1.0 / num_steps

        def step(i: jax.Array, actions: jax.Array) -> jax.Array:
            time = jnp.asarray(i * dt, dtype=jnp.float32)
            velocity = self._predict_cached_velocity(conditioning_cache, actions, time)
            return actions + dt * velocity

        return jax.lax.fori_loop(0, num_steps, step, noise)
