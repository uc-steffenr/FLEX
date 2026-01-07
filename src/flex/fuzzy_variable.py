"""Defines fuzzy variable class."""
from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax.numpy as jnp

from .mfs import BaseMF

Array = jnp.ndarray


class FuzzyVariable(eqx.Module):
    theta: Array  # parameters of corresponding MFs
    name: str = eqx.field(static=True, default="x", kw_only=True)
    mode: str = eqx.field(static=True, default="manual")

    def __init__(self, mfs: Sequence[BaseMF], *, name: str="x") -> None:
        pass

    @classmethod
    def ruspini(cls, *, n_mfs: int, kind: str="triangle", name: str="x") -> "FuzzyVariable":       
        obj = cls(name=name, mode="ruspini")
        return obj

    def __call__(self, x: Array):
        pass
