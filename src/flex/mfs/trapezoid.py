"""Defines Trapezoid Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class Trapezoid(BaseMF):
    a: Array
    b: Array
    c: Array
    d: Array
    name: str = eqx.field(static=True, default="trap", kw_only=True)

    def __call__(self, x: Array) -> Array:
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        eps = self.eps

        # Numerical stability check
        b = jnp.maximum(b, a + eps)
        c = jnp.maximum(c, b)
        d = jnp.maximum(d, c + eps)

        left = jnp.clip((x - a) / (b - a), 0.0, 1.0)
        right = jnp.clip((d - x) / (d - c), 0.0, 1.0)

        return jnp.minimum(left, right)

    @property
    def params(self) -> Array:
        return jnp.stack([self.a, self.b, self.c, self.d])

    def validate(self) -> None:
        if not self.a < self.b < self.c < self.d:
            raise ValueError("a < b < c < d for trapezoidal membership function.")
