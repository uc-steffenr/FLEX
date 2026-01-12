"""Defines Triangle Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import triangle

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class Triangle(BaseMF):
    name: str = eqx.field(static=True, default="tri", kw_only=True)

    def __call__(self, x: Array, nodes: Array) -> Array:
        idx = self.idx
        eps = self.eps

        a = nodes[idx - 1]
        b = nodes[idx]
        c = nodes[idx + 1]

        return triangle(x, a, b, c, eps)
