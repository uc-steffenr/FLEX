"""Defines Trapezoid Membership Function."""
from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def gaussian(x: Array, sig: float, mu: float, eps: float=1e-12) -> Array:
    # Numerical stability check
    sig = jnp.maximum(sig, eps)
    return jnp.exp(-0.5*((x - mu) / sig)**2)

def triangle(
    x: Array,
    a: float,
    b: float,
    c: float,
    eps: float=1e-12,
) -> Array:
    # Numerical stability check
    b = jnp.maximum(b, a + eps)
    c = jnp.maximum(c, b + eps)

    left = (x - a) / (b - a)
    right = (c - x) / (c - b)

    return jnp.maximum(jnp.minimum(left, right), 0.0)

def trapezoid(
    x: Array,
    a: float,
    b: float,
    c: float,
    d: float,
    eps: float=1e-12,
) -> Array:
    # Numerical stability check
    b = jnp.maximum(b, a + eps)
    c = jnp.maximum(c, b)
    d = jnp.maximum(d, c + eps)

    left = jnp.clip((x - a) / (b - a), 0.0, 1.0)
    right = jnp.clip((d - x) / (d - c), 0.0, 1.0)

    return jnp.minimum(left, right)

def left_shoulder(x: Array, c: float, d: float, eps: float=1e-12) -> Array:
    # Numerical stability check
    d = jnp.maximum(d, c + eps)

    return jnp.clip((d - x) / (d - c), 0.0, 1.0)

def right_shoulder(x: Array, a: float, b: float, eps: float=1e-12) -> Array:
    # Numerical stability check
    b = jnp.maximum(b, a + eps)

    return jnp.clip((x - a) / (b - a), 0.0, 1.0)
