"""Defines train method."""
import os
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx


def train(model, x_train, y_train, x_val, y_val):
    raise NotImplementedError

def train_from_shards():
    raise NotImplementedError
