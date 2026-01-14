"""Defines Gaussian Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import gaussian

import jax.numpy as jnp
import equinox as eqx

from ..utils.types import Array


class Gaussian(BaseMF):
    sig_idx: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="gauss", kw_only=True)

    def __call__(self, x: Array, nodes: Array, sigs: Array) -> Array:
        idx = self.idx
        sig_idx = self.sig_idx
        eps = self.eps

        sig = sigs[sig_idx]
        mu = nodes[idx]

        return gaussian(x, sig, mu, eps)

    def get_params(self, nodes: Array, sigs: Array) -> Array:
        sig = sigs[self.sig_idx]
        mu = nodes[self.idx]

        return jnp.array([sig, mu])
