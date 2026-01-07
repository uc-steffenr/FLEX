"""Defines fuzzy variable class."""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .mfs import gaussian, triangle, trapezoid, left_shoulder, right_shoulder

Array = jnp.ndarray


# TODO: figure out membership function storing, parameter storing, indexing, and how to vectorize call
class FuzzyVariable(eqx.Module):

    def __call__(self, x: Array):
        pass
