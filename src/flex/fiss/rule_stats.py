"""Defines class to track rulebase states."""
from __future__ import annotations

from typing import Literal

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
    w: Array,                  # (R,) or (B, R)
    tau: float = 1e-3,
    ema_alpha: float = 0.01,
    reduce: Literal["sum", "mean"] = "sum",
) -> RuleStats:
    """Hot-path update: O(B*R) reductions only (no antecedent aggregation).

    Parameters
    ----------
    w : (R,) or (B,R)
        Rule firing strengths.
    tau :
        Threshold used to define a "fire" event contributing to count.
    ema_alpha :
        EMA update rate for ema_mass.
    reduce :
        "sum" accumulates totals; "mean" accumulates batch-means (batch-size invariant).

    Returns
    -------
    Updated RuleStats
    """
    if w.ndim == 1:
        w_b = w[None, :]
    elif w.ndim == 2:
        w_b = w
    else:
        raise ValueError(f"w must have shape (R,) or (B,R), got {w.shape}")

    w_b = w_b.astype(stats.mass.dtype)

    if reduce == "sum":
        batch_mass = jnp.sum(w_b, axis=0)  # (R,)
        batch_count = jnp.sum((w_b > tau).astype(w_b.dtype), axis=0)  # (R,)
    elif reduce == "mean":
        batch_mass = jnp.mean(w_b, axis=0)
        batch_count = jnp.mean((w_b > tau).astype(w_b.dtype), axis=0)
    else:
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce}")

    mass = stats.mass + batch_mass
    count = stats.count + batch_count
    ema_mass = (1.0 - ema_alpha) * stats.ema_mass + ema_alpha * batch_mass

    # Avoid eqx.tree_at in the hot path.
    return RuleStats(mass=mass, count=count, ema_mass=ema_mass)

def mf_usage_from_rule_values(
    *,
    antecedents: Array,   # int32 (R, V), -1 is don't-care
    rule_values: Array,   # (R,) typically stats.mass or stats.ema_mass or stats.count
    max_mfs: int,
    normalize: bool = False,
) -> Array:
    """Cold-path aggregation: convert per-rule values into per-(var, MF) usage.

    Complexity: O(R*V + V*M) via V bincounts. Intended for periodic analysis/pruning.

    Returns
    -------
    (V, max_mfs) array
    """
    ants = jnp.asarray(antecedents, dtype=jnp.int32)
    vals = jnp.asarray(rule_values)

    if ants.ndim != 2:
        raise ValueError(f"antecedents must be (R,V), got {ants.shape}")
    if vals.ndim != 1:
        raise ValueError(f"rule_values must be (R,), got {vals.shape}")
    if ants.shape[0] != vals.shape[0]:
        raise ValueError(f"R mismatch: {ants.shape[0]} vs {vals.shape[0]}")

    R, V = ants.shape
    M = int(max_mfs)

    active = (ants != -1)        # (R, V)
    idx = jnp.maximum(ants, 0)   # safe, (R, V)

    out_per_v = []
    for v in range(V):
        w = vals * active[:, v].astype(vals.dtype)  # exclude don't-care for this var
        out_per_v.append(jnp.bincount(idx[:, v], weights=w, length=M))

    out = jnp.stack(out_per_v, axis=0)  # (V, M)

    if normalize:
        denom = jnp.sum(out, axis=1, keepdims=True)
        out = jnp.where(denom > 0, out / denom, out)

    return out

def mf_usage_from_stats(
    *,
    antecedents: Array,   # (R, V)
    stats: RuleStats,
    max_mfs: int,
    which: Literal["mass", "ema_mass", "count"] = "ema_mass",
    normalize: bool = False,
) -> Array:
    """Cold-path convenience wrapper: selects a field from RuleStats then aggregates."""
    if which == "mass":
        rule_values = stats.mass
    elif which == "ema_mass":
        rule_values = stats.ema_mass
    elif which == "count":
        rule_values = stats.count
    else:
        raise ValueError(f"which must be one of ('mass','ema_mass','count'), got {which}")

    return mf_usage_from_rule_values(
        antecedents=antecedents,
        rule_values=rule_values,
        max_mfs=max_mfs,
        normalize=normalize,
    )
