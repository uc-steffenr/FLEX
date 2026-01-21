"""Defines base FIS class."""
from __future__ import annotations

import abc
from typing import Literal, Sequence, Tuple

import jax.numpy as jnp
import equinox as eqx

from .rule_base import RuleBase
from ..fuzzy_variable import FuzzyVariable
from ..utils.types import Array


# TODO: there should be a mechanism for multi-output in a fis (reuses rule base, different consequents)
class BaseFIS(eqx.Module, abc.ABC):
    input_vars: tuple[FuzzyVariable, ...]

    rb: RuleBase = eqx.field(static=True)
    n_mfs_max: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="fis", kw_only=True)

    @classmethod
    def init(
        cls,
        input_vars: tuple[FuzzyVariable, ...],
        *,
        antecedents: Sequence[Sequence[Tuple[int, int]]]|None = None,
        name: str="fis",
        tnorm: Literal["prod", "min"] = "prod",
    ) -> "BaseFIS":
        n_mfs_per_var = [fv.n_mfs for fv in input_vars]
        n_mfs_max = max(n_mfs_per_var)

        if antecedents is None:  # assume dense rulebase
            rb = RuleBase.dense(n_mfs_per_var=n_mfs_per_var, tnorm=tnorm)
        else:
            assert antecedents.shape[1] == len(input_vars)
            rb = RuleBase.sparse(n_vars=len(input_vars), rules=antecedents, tnorm=tnorm)

        return cls(input_vars=input_vars, rb=rb, n_mfs_max=n_mfs_max, name=name)

    def fuzzify(self, x: Array) -> Array:
        x = jnp.asarray(x)

        mus = []
        for i, var in enumerate(self.input_vars):
            mu = var(x[..., i])
            pad_width = self.n_mfs_max - mu.shape[-1]

            if pad_width > 0:
                mu = jnp.pad(
                    mu,
                    pad_width=((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0.0,
                )

            mus.append(mu)

        return jnp.stack(mus, axis=-2)

    @abc.abstractmethod
    def defuzzify(self, w: Array, x: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> Array:
        mus = self.fuzzify(x)
        w = self.rb.fire(mus)

        return self.defuzzify(w, x)
