"""Defines rulebase and rule stats class."""
from __future__ import annotations

from typing import Literal, Sequence, Tuple

import jax.numpy as jnp
import equinox as eqx

from ..utils.types import Array


class RuleBase(eqx.Module):
    antecedents: Array  # shape (n_rules, n_vars) -> each rule contains n_vars w/ specific mf index
    tnorm: Literal["prod", "min"] = eqx.field(static=True)

    def __init__(
        self,
        antecedents: Array,
        *,
        tnorm: Literal["prod", "min"] = "prod",
    ) -> None:
        ants = jnp.asarray(antecedents, dtype=jnp.int32)
        if ants.ndim != 2:
            raise ValueError(f"antecedents must be 2D (n_rules, n_vars), got {ants.shape}")

        self.antecedents = ants
        self.tnorm = tnorm

    @property
    def n_rules(self) -> int:
        return int(self.antecedents.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.antecedents.shape[1])

    def fire(self, mu: Array) -> Array:
        if mu.ndim == 2:
            mu_batched = mu[None, :, :]
            squeeze_out = True
        elif mu.ndim == 3:
            mu_batched = mu
            squeeze_out = False
        else:
            raise ValueError(f"mu must either have ndim=2 or ndim=3, got {mu.ndim}.")

        _, V, M = mu_batched.shape
        if V != self.n_vars:
            raise ValueError(f"mu has n_vars={V}, but rulebase has n_vars={self.n_vars}.")

        ants = self.antecedents

        if jnp.any((ants >= M) | (ants < -1)):
            raise ValueError("antecedents contain invalid MF indices (must be in [-1, max_mfs-1]).")

        idx = jnp.maximum(ants, 0)

        gathered = jnp.take_along_axis(
            mu_batched[:, None, :, :],
            idx[None, :, :, None],
            axis=-1,
        ).squeeze(-1)

        gathered = jnp.where(ants[None, :, :] == -1, 1.0, gathered)

        if self.tnorm == "prod":
            w = jnp.prod(gathered, axis=-1)
        elif self.tnorm == "min":
            w = jnp.min(gathered, axis=-1)
        else:
            raise ValueError(f"Unknown tnorm: {self.tnorm}.")

        if squeeze_out:
            return w[0]
        return w

    @classmethod
    def dense(
        cls, 
        n_mfs_per_var: Sequence[int],
        *,
        tnorm: Literal["prod", "min"] = "prod",
    ) -> "RuleBase":
        n_mfs_per_var = tuple(int(m) for m in n_mfs_per_var)
        if len(n_mfs_per_var) == 0:
            raise ValueError("n_mfs_per_var must be non-empty.")
        if any(m <= 0 for m in n_mfs_per_var):
            raise ValueError(f"All MF counts must be positive, got {n_mfs_per_var}")

        R = 1
        for m in n_mfs_per_var:
            R *= m

        r = jnp.arange(R, dtype=jnp.int32)  # (R,)
        ants_cols = []
        base = 1
        for m in n_mfs_per_var:
            ants_cols.append((r // base) % m)
            base *= m

        ants = jnp.stack(ants_cols, axis=1)  # (R, V)
        return cls(ants, tnorm=tnorm)

    @classmethod
    def sparse(
        cls,
        *,
        n_vars: int,
        rules: Sequence[Sequence[Tuple[int, int]]],
        tnorm: Literal["prod", "min"] = "prod",
    ) -> "RuleBase":
        V = int(n_vars)
        if V <= 0:
            raise ValueError("n_vars must be positive.")

        R = len(rules)
        ants = -jnp.ones((R, V), dtype=jnp.int32)

        for i, rule in enumerate(rules):
            for v, mf in rule:
                v = int(v)
                mf = int(mf)
                if not (0 <= v < V):
                    raise ValueError(f"Rule {i} has invalid var index {v} for n_vars={V}")
                if mf < 0:
                    raise ValueError(f"Rule {i} has invalid mf index {mf} (must be >= 0)")
                ants = ants.at[i, v].set(mf)

        return cls(ants, tnorm=tnorm)
