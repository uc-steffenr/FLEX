"""Defines rulebase and rule stats class."""
from __future__ import annotations

from typing import Literal, Sequence, Tuple

import jax.numpy as jnp
import equinox as eqx

from ..utils.types import Array


class RuleBase(eqx.Module):
    antecedents: Array  # shape (n_rules, n_vars) -> -1 for "don't care" rules
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
        if mu.ndim < 2:
            raise ValueError(f"mu must have at least 2 dims (..., n_vars, max_mfs), got {mu.shape}")

        *batch_shape, V, M = mu.shape
        if V != self.n_vars:
            raise ValueError(f"mu has n_vars={V}, but rulebase has n_vars={self.n_vars}.")

        # Flatten arbitrary batch dims -> (B, V, M) without computing B in Python from JAX ops.
        if len(batch_shape) == 0:
            mu_batched = mu[None, :, :]      # (1, V, M)
            squeeze_out = True
        else:
            mu_batched = jnp.reshape(mu, (-1, V, M))  # (B, V, M)
            squeeze_out = False

        ants = self.antecedents            # (R, V)
        idx = jnp.maximum(ants, 0)         # (R, V)

        gathered = jnp.take_along_axis(
            mu_batched[:, None, :, :],     # (B, 1, V, M)
            idx[None, :, :, None],         # (1, R, V, 1)
            axis=-1,
        ).squeeze(-1)                      # (B, R, V)

        one = jnp.array(1.0, dtype=mu_batched.dtype)
        gathered = jnp.where(ants[None, :, :] == -1, one, gathered)

        if self.tnorm == "prod":
            w = jnp.prod(gathered, axis=-1)   # (B, R)
        elif self.tnorm == "min":
            w = jnp.min(gathered, axis=-1)
        else:
            raise ValueError(f"Unknown tnorm: {self.tnorm}.")

        if squeeze_out:
            return w[0]  # (R,)

        # Restore original batch shape: (..., R)
        return jnp.reshape(w, (*batch_shape, self.n_rules))

    @classmethod
    def dense(
        cls, 
        n_mfs_per_var: Sequence[int],
        *,
        tnorm: Literal["prod", "min"] = "prod",
    ) -> "RuleBase":
        """Iinitializes dense rule base based on FIS fuzzy variable values.

        Parameters
        ----------
        n_mfs_per_var : Sequence[int]
            Number of membership functions per variable.
        tnorm : Literal[&quot;prod&quot;, &quot;min&quot;], optional
            Aggergation method to use, by default "prod".

        Returns
        -------
        RuleBase
            Instantiation of rule base.

        Raises
        ------
        ValueError
            Must have a n_mfs_per_var list of len > 0.
        ValueError
            Must have positive integers in n_mfs_per_var.
        """
        n_mfs_per_var = tuple(int(m) for m in n_mfs_per_var)
        if len(n_mfs_per_var) == 0:
            raise ValueError("n_mfs_per_var must be non-empty.")
        if any(m <= 0 for m in n_mfs_per_var):
            raise ValueError(f"All MF counts must be positive, got {n_mfs_per_var}")

        # compute number of rules
        R = 1
        for m in n_mfs_per_var:
            R *= m

        # assembling dense rules -- mixed radix enumeration (better than nested loops)
        r = jnp.arange(R, dtype=jnp.int32)  # (R,)
        ants_cols = []
        base = 1
        for m in n_mfs_per_var:
            ants_cols.append((r // base) % m)
            base *= m

        ants = jnp.stack(ants_cols, axis=1)  # (R, V)
        return cls(ants, tnorm=tnorm)

    # NOTE: It's not always the case that all variables are included in the rulebase,
    # so n_vars needs to be a separate argument
    @classmethod
    def sparse(
        cls,
        *,
        n_vars: int,
        rules: Sequence[Sequence[Tuple[int, int]]],
        tnorm: Literal["prod", "min"] = "prod",
    ) -> "RuleBase":
        """Initializes sparse rule base based on user specification.

        Parameters
        ----------
        n_vars : int
            Number of variables inherent to the FIS.
        rules : Sequence[Sequence[Tuple[int, int]]]
            List of rules to be used as antecedents.
        tnorm : Literal[&quot;prod&quot;, &quot;min&quot;], optional
            Aggregation method to use, by default "prod".

        Returns
        -------
        RuleBase
            Instantiation of rule base.

        Raises
        ------
        ValueError
            n_vars must be positive.
        ValueError
            Invalid variable index.
        ValueError
            Invalid mf index.
        """
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
