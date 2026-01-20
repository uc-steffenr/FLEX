"""Defines base FIS class."""
from __future__ import annotations

import abc

import jax.numpy as jnp
import equinox as eqx

from .rule_base import RuleBase
from ..fuzzy_variable import FuzzyVariable
from ..utils.types import Array


class BaseFIS(eqx.Module, abc.ABC):
    input_vars: tuple[FuzzyVariable, ...]

    rb: RuleBase = eqx.field(static=True)
    n_mfs_max: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="fis", kw_only=True)

    @classmethod
    @abc.abstractmethod
    def init(cls, input_vars: tuple[FuzzyVariable, ...], *, rb_mode, name) -> "BaseFIS":
        raise NotImplementedError

    def fuzzify(self, x: Array) -> Array:
        x = jnp.asarray(x)

    @abc.abstractmethod
    def defuzzify(self, x: Array, w: Array, mus: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> Array:
        mus = self.fuzzify(x)
        w = self.rb.fire(mus)

        return self.defuzzify(x, w, mus)
