"""Defines types used throughout FLEX."""
from __future__ import annotations

from typing import Union

import jax.numpy as jnp

Array = jnp.ndarray
ScalarLike = Union[float, int, Array]
