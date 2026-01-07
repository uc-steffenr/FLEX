"""Defines Left and Right Shoulder Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class LeftShoulder(BaseMF):
    c: Array
    d: Array
    name: str = eqx.field(static=True, default="left", kw_only=True)

    def __call__(self, x: Array) -> Array:
        c = self.c
        d = self.d
        eps = self.eps

        # Numerical stability check
        d = jnp.maximum(d, c + eps)

        return jnp.clip((d - x) / (d - c), 0.0, 1.0)

    @property
    def params(self) -> Array:
        return jnp.stack([self.c, self.d])

    def validate(self) -> None:
        if not self.c < self.d:
            raise ValueError("c value must be < d value for left shoulder.")


class RightShoulder(BaseMF):
    a: Array
    b: Array
    name: str = eqx.field(static=True, default="right", kw_only=True)

    def __call__(self, x: Array) -> Array:
        a = self.a
        b = self.b
        eps = self.eps

        # Numerical stability check
        b = jnp.maximum(b, a + eps)

        return jnp.clip((x - a) / (b - a), 0.0, 1.0)

    @property
    def params(self) -> Array:
        return jnp.stack([self.a, self.b])

    def validate(self) -> None:
        if not self.a < self.b:
            raise ValueError("a value must be < b value for right shoulder.")
