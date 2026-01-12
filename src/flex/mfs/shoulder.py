"""Defines Left and Right Shoulder Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import left_shoulder, right_shoulder

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class LeftShoulder(BaseMF):
    name: str = eqx.field(static=True, default="left", kw_only=True)

    def __call__(self, x: Array, nodes: Array) -> Array:
        idx = self.idx
        eps = self.eps

        c = nodes[idx]
        d = nodes[idx + 1]

        return left_shoulder(x, c, d, eps)


class RightShoulder(BaseMF):
    name: str = eqx.field(static=True, default="right", kw_only=True)

    def __call__(self, x: Array, nodes: Array) -> Array:
        idx = self.idx
        eps = self.eps

        a = nodes[idx - 1]
        b = nodes[idx]

        return right_shoulder(x, a, b, eps)
