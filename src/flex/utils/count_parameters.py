"""Defines tool to count parameters."""
from __future__ import annotations

import jax
import equinox as eqx


def count_parameters(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_array)
    )
    return sum(leaf.size for leaf in leaves)
