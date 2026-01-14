"""Defines Left and Right Shoulder Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import left_shoulder, right_shoulder

import equinox as eqx
import jax.numpy as jnp

from ..utils.types import Array


class LeftShoulder(BaseMF):
    name: str = eqx.field(static=True, default="left", kw_only=True)

    def __call__(self, x: Array, nodes: Array) -> Array:
        idx = self.idx
        eps = self.eps

        c = nodes[idx]
        d = nodes[idx + 1]

        return left_shoulder(x, c, d, eps)

    def get_params(self, nodes: Array) -> Array:
        c = nodes[self.idx]
        d = nodes[self.idx + 1]

        return jnp.array([c, d])


class RightShoulder(BaseMF):
    name: str = eqx.field(static=True, default="right", kw_only=True)

    def __call__(self, x: Array, nodes: Array) -> Array:
        idx = self.idx
        eps = self.eps

        a = nodes[idx - 1]
        b = nodes[idx]

        return right_shoulder(x, a, b, eps)

    def get_params(self, nodes: Array) -> Array:
        a = nodes[self.idx - 1]
        b = nodes[self.idx]

        return jnp.array([a, b])
