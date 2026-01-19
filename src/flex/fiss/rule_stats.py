"""Defines class to track rulebase states."""
from __future__ import annotations

import equinox as eqx
from jax.numpy import jnp

from ..utils.types import Array


class RuleStats(eqx.Module):
    mass: Array
    count: Array
    ema_mass: Array

    @classmethod
    def init(cls, n_rules: int, *, dtype=jnp.float32) -> "RuleStats":
        z = jnp.zeros((int(n_rules),), dtype=dtype)
        return cls(mass=z, count=z, ema_mass=z)


def update_rule_stats(
    stats: RuleStats,
    *,
    w: Array,
    antecedents: Array,
    tau: float = 1e-3,
    ema_alpha: float = 0.01,
) -> RuleStats:
    if w.ndim == 1:
        w_b = w[None, :]
    elif w.ndim == 2:
        w_b = w
    else:
        raise ValueError(f"w must have 1 or 2 dimensions, got {w.ndim}.")

    w_b = w_b.astype(stats.mass.dtype)
