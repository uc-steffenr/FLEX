"""Defines Triangle Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class Triangle(BaseMF):
    a: Array
    b: Array
    c: Array
    name: str = eqx.field(static=True, default="tri", kw_only=True)

    def __call__(self, x: Array) -> Array:
        a = self.a
        b = self.b
        c = self.c
        eps = self.eps

        # Numerical stability check
        b = jnp.maximum(b, a + eps)
        c = jnp.maximum(c, b + eps)

        left = (x - a) / (b - a)
        right = (c - x) / (c - b)

        return jnp.maximum(jnp.minimum(left, right), 0.0)

    @property
    def params(self) -> Array:
        return jnp.stack([self.a, self.b, self.c])

    def validate(self) -> None:
        if not self.a < self.b < self.c:
            raise ValueError("a < b < c for triangular membership function.")
