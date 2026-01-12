"""Defines fuzzy variable class."""
from __future__ import annotations

from typing import Callable, Tuple, Sequence, Dict

import equinox as eqx
import jax.numpy as jnp

from .mfs import BaseMF, LeftShoulder, RightShoulder, Trapezoid, Triangle, Gaussian


Array = jnp.ndarray


# maps str name to corresponding class
MF_DICT: Dict[str, Callable] = {
    "left_shoulder": LeftShoulder,
    "right_shoulder": RightShoulder,
    "trapezoid": Trapezoid,
    "triangle": Triangle,
    "gaussian": Gaussian,
}

# maps str name to number of parameters
MF_ARITY: Dict[str, int] = {
    "left_shoulder": 2,
    "right_shoulder": 2,
    "trapezoid": 4,
    "triangle": 3,
    "gaussian": 2,
}


# Each MF has a corresponding index that they use for the parameters theta
# These parameters should be unconstrained.... need a nodes method using softplus to retrive the actual values
class FuzzyVariable(eqx.Module):
    thetas: Array  # parameters of corresponding MFs
    sigmas: Array|None = None  # standard deviations of gaussians -> NOTE: MIGHT BREAK THINGS

    mfs: Tuple[BaseMF] = eqx.field(static=True)
    minval: float = eqx.field(static=True, default=0.0)
    maxval: float = eqx.field(static=True, default=1.0)
    name: str = eqx.field(static=True, default="x")
    mode: str = eqx.field(static=True, default="manual")

    # TODO: allow manual to specify node positions, and to allow ruspini with node positions
    @classmethod 
    def manual(
        cls,
        mfs: Sequence[str],
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        mode: str = "manual",
        init: str = "manual",
        noise_scaler: float = 0.1,
        params: Sequence[Sequence[float]]|None = None,
    ) -> "FuzzyVariable":
        if any(mfs not in MF_DICT.keys()):
            raise ValueError("Invalid MF type specified. Must be  ",
                             "\"left_shoulder\", \"right_shoulder\", ",
                             "\"triangle\", \"trapezoid\", or \"gaussian\".")

    # TODO: think a bit about unconstrained parameters with softplus vs simply clipping parameters in call
    # TODO: specify key for jax prng should noisy initialization be used
    @classmethod
    def ruspini(
        cls,
        kind: str,
        n_mfs: int,
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        init: str = "uniform",
        noise_scaler: float = 0.1,
    ) -> "FuzzyVariable":
        # Sanity checks
        if kind not in ["triangle", "trapezoid"]:
            raise ValueError(f"kind must be \"triangle\" or \"trapezoid\", got {kind}.")

        if n_mfs <= 1:
            raise ValueError(f"n_mfs must be > 1 for ruspini partitions, got {n_mfs}.")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", go {init}.")

        # given kind of membership function, set up should
        if kind == "triangle":
            nn = n_mfs - 2
        elif kind == "trapezoid":
            nn = 2*(n_mfs - 2)

        if init == "uniform":
            nodes = jnp.linspace(minval, maxval, nn)  # want to turn into unconstrained parameters

    def __call__(self, x: Array):
        pass
