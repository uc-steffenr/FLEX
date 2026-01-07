"""Defines Gaussian Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import gaussian

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class Gaussian(BaseMF):
    sig: Array
    mu: Array

    name: str = eqx.field(static=True, default="gauss", kw_only=True)

    def __call__(self, x: Array) -> Array:
        sig = jnp.maximum(self.sig, self.eps)
        mu  = self.mu

        return jnp.exp(-0.5*((x - mu) / sig)**2)

    @property
    def params(self) -> Array:
        return jnp.stack([self.sig, self.mu])

    def validate(self) -> None:
        if self.sig <= 0.0:
            raise ValueError("Assigned sigma value must be > 0.0.")
