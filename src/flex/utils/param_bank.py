"""Defines parameters data class"""
from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx


Array = jnp.ndarray


class ParamBank(eqx.Module):
    gaps: Array
    raw_sigmas: Array | None = None
