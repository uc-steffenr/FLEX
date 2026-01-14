"""Defines useful functions for FLEX."""
from __future__ import annotations

import warnings

from jax.nn import softplus
import jax.numpy as jnp

from .types import Array, ScalarLike


warnings.simplefilter("once", UserWarning)

# NOTE: assumes minval and maxval are already in the node array
def nodes2gaps(n: Array, eps: float=1e-8) -> Array:
    gaps = jnp.maximum(jnp.diff(n), eps)
    return jnp.log(jnp.expm1(gaps) + eps)

def gaps2nodes(g: Array, xmin: ScalarLike, xmax: ScalarLike, eps: float=1e-8) -> Array:
    xmin = jnp.asarray(xmin)
    xmax = jnp.asarray(xmax)

    L = jnp.maximum(xmax - xmin, eps)
    gaps = softplus(g) + eps

    gaps = gaps / jnp.sum(gaps) * L

    boundaries = xmin + jnp.cumsum(gaps)

    return jnp.concatenate([jnp.atleast_1d(xmin), boundaries], axis=0)

def sig_softplus(s: Array, eps: float=1e-8) -> Array:
    return softplus(s) + eps

def inv_sig_softplus(s: Array, eps: float=1e-8) -> Array:
    return jnp.log(jnp.expm1(s) + eps)
