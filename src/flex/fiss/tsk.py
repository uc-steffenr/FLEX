"""Defines Takagi Sugeno-Kang FIS class."""
from __future__ import annotations

from typing import Sequence, Self, Literal

import jax
import equinox as eqx
import jax.numpy as jnp

from .base_fis import BaseFIS
from ..fuzzy_variable import FuzzyVariable
from ..utils.types import Array


class TSK(BaseFIS):
    consequents: Array  # shape: (n_out, n_rules, n_features)
    order: int = eqx.field(static=True, default=1)

    @classmethod
    def init(
        cls,
        input_vars: tuple[FuzzyVariable, ...],
        *,
        n_out: int = 1,
        order: Literal[0, 1, 2] = 1,
        antecedents: Sequence[Sequence[tuple[int, int]]]|None = None,
        tnorm: Literal["prod", "min"] = "prod",
        name: str = "tsk",
        eps: float = 1e-6,
        key: Array,
        init_scale: float = 1e-2,
    ) -> Self:
        if n_out <= 0:
            raise ValueError("n_out must be >= 1.")

        base = super().init(
            input_vars=input_vars,
            antecedents=antecedents,
            name=name,
            tnorm=tnorm,
        )

        n_inps = base.n_inps

        if order == 0:
            n_features = 1
        elif order == 1:
            n_features = 1 + n_inps
        elif order == 2:
            n_features = 1 + n_inps + n_inps * (n_inps + 1) // 2
        else:
            raise ValueError("order must be 0, 1, or 2.")

        n_rules = base.rb.n_rules

        k1, = jax.random.split(key, 1)
        consequents = init_scale * jax.random.normal(
            k1, (n_out, n_rules, n_features),
        )

        return cls(
            input_vars=input_vars,
            rb=base.rb,
            n_mfs_max=base.n_mfs_max,
            name=base.name,
            consequents=consequents,
            order=int(order),
            eps=eps,
        )

    def phi(self, x: Array) -> Array:
        x = jnp.asarray(x)
        ones = jnp.ones((*x.shape[:-1], 1), dtype=x.dtype)

        if self.order == 0:
            return ones

        if self.order == 1:
            return jnp.concatenate([ones, x], axis=-1)

        n = x.shape[-1]
        quad_terms = []
        for i in range(n):
            xi = x[..., i]
            for j in range(i, n):
                quad_terms.append(xi * x[..., j])
        quad = jnp.stack(quad_terms, axis=-1)
        return jnp.concatenate([ones, x, quad], axis=-1)

    def defuzzify(self, w: Array, x: Array) -> Array:
        x = jnp.asarray(x)
        w = jnp.asarray(w)

        phi = self.phi(x)

        fr = jnp.einsum("...f,orf->...or", phi, self.consequents)

        w_sum = jnp.sum(w, axis=-1, keepdims=True)
        w_norm = w / (w_sum + self.eps)
        y = jnp.einsum("...r,...or->...o", w_norm, fr)

        return y

    @property
    def n_out(self) -> int:
        return self.consequents.shape[0]

    @property
    def n_features(self) -> int:
        return self.consequents.shape[2]
