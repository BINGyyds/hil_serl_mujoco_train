from collections.abc import Mapping
from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded


class StateEncodingWrapper(nn.Module):
    """
    Lightweight encoder for state-only observations.

    Accepts either a raw state array or the standard SERL-style observation dict
    with a ``state`` key, and optionally flattens observation stacking from
    ``(T, C)``/``(B, T, C)`` into ``(T*C,)``/``(B, T*C)``.
    """

    enable_stacking: bool = False

    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
        del train, stop_gradient, is_encoded

        if isinstance(observations, Mapping):
            state = observations["state"]
        else:
            state = observations

        if self.enable_stacking:
            if len(state.shape) == 2:
                state = rearrange(state, "T C -> (T C)")
            elif len(state.shape) == 3:
                state = rearrange(state, "B T C -> B (T C)")

        return state
