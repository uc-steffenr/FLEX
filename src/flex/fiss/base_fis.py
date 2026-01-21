"""Defines base FIS class."""
from __future__ import annotations

import abc
from typing import Literal, Self, Sequence

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
    eps: float = eqx.field(static=True, default=1e-6, kw_only=True)

    @classmethod
    def init(
        cls,
        input_vars: tuple[FuzzyVariable, ...],
        *,
        antecedents: Sequence[Sequence[tuple[int, int]]]|None = None,
        tnorm: Literal["prod", "min"] = "prod",
        name: str = "fis",
        eps: float = 1e-6,
    ) -> Self:
        input_vars = tuple(input_vars)

        n_mfs_per_var = [fv.n_mfs for fv in input_vars]
        n_mfs_max = max(n_mfs_per_var)

        if antecedents is None:  # assume dense rulebase
            rb = RuleBase.dense(n_mfs_per_var=n_mfs_per_var, tnorm=tnorm)
        else:
            rb = RuleBase.sparse(n_vars=len(input_vars), rules=antecedents, tnorm=tnorm)

        return cls(input_vars=input_vars, rb=rb, n_mfs_max=n_mfs_max, name=name, eps=eps)

    def fuzzify(self, x: Array) -> Array:
        """Fuzzifies input values.

        Parameters
        ----------
        x : Array
            Input values of shape (..., n_inps).

        Returns
        -------
        Array
            Membership functions values, of shape (..., n_inps, n_mfs_max).
        """
        x = jnp.asarray(x)

        eqx.error_if(
            x,
            x.shape[-1] != self.n_inps,
            f"x last dimension must be {self.n_inpts}."
        )

        mus = []
        for i, var in enumerate(self.input_vars):
            mu = var(x[..., i])
            pad = self.n_mfs_max - mu.shape[-1]

            if pad > 0:
                pad_spec = ((0, 0),) * (mu.ndim - 1) + ((0, pad),)
                mu = jnp.pad(
                    mu,
                    pad_width=pad_spec,
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

    @property
    def n_inps(self) -> int:
        return len(self.input_vars)

    @property
    def n_rules(self) -> int:
        return self.rb.n_rules
