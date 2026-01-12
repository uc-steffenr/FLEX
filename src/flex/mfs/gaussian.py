"""Defines Gaussian Membership Function Class."""
from __future__ import annotations

from .base_mf import BaseMF
from .functions import gaussian

import equinox as eqx
import jax.numpy as jnp


Array = jnp.ndarray


class Gaussian(BaseMF):
    sig_idx: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="gauss", kw_only=True)

    def __call__(self, x: Array, nodes: Array, sigs: Array) -> Array:
        idx = self.idx
        sig_idx = self.sig_idx
        eps = self.eps

        sig = nodes[sig_idx]
        mu = sigs[idx]

        return gaussian(x, sig, mu, eps)
