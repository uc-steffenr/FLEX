"""Defines fuzzy variable class."""
from __future__ import annotations

from typing import Tuple, Sequence, Dict, Type

import jax
import jax.numpy as jnp
import equinox as eqx

from .mfs import BaseMF, LeftShoulder, RightShoulder, Trapezoid, Triangle, Gaussian
from .utils.param_bank import ParamBank
from .utils.types import Array, ScalarLike
from .utils.unconstrained_parameters import gaps2nodes, sig_softplus


# maps str name to corresponding class
MF_DICT: Dict[str, Type[BaseMF]] = {
    "left_shoulder": LeftShoulder,
    "right_shoulder": RightShoulder,
    "trapezoid": Trapezoid,
    "triangle": Triangle,
    "gaussian": Gaussian,
}


# Each MF has a corresponding index that they use for the parameters theta
# These parameters should be unconstrained.... need a nodes method using softplus to retrive the actual values
class FuzzyVariable(eqx.Module):
    params: ParamBank

    mfs: Tuple[BaseMF, ...] = eqx.field(static=True)
    minval: float = eqx.field(static=True, default=0.0)
    maxval: float = eqx.field(static=True, default=1.0)
    name: str = eqx.field(static=True, default="x")

    @classmethod 
    def manual(
        cls,
        mfs: Sequence[str],
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        mode: str = "manual",
        init: str = "uniform",
        key: Array|None = None,
        noise_scaler: float = 0.1,
        params: Sequence[Sequence[float]]|None = None,
    ) -> "FuzzyVariable":
        raise NotImplementedError("Manual initialization has not been properly implemented yet.")

        # Sanity checks
        if any(mf not in MF_DICT for mf in mfs):
            raise ValueError("Invalid MF type specified. Must be  ",
                             "\"left_shoulder\", \"right_shoulder\", ",
                             "\"triangle\", \"trapezoid\", or \"gaussian\".")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", go {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy initi must use a PRNGKey for initialization.")

        if init == "manual" and params is None:
            raise ValueError("Manual initialization requires parameters to be specified.")

        if mode == "ruspini" and "gaussian" in mfs:
            raise ValueError("Ruspini partitions cannot be used with gaussian mfs.")

        if mode == "ruspini" and ("left_shoulder" != mfs[0] and "right_shoulder" != mfs[-1]):
            raise ValueError("If Ruspini partitions are used, the first mf must be \"left_shoulder\" and the last mf must be \"right_shoulder\".")

        n_mfs = len(mfs)

        # TODO: include manual init for ruspini partitions
        if mode == "ruspini":
            nn = 0
            for mf in mfs[1:-1]:
                if mf == "triangle":
                    nn += 1
                elif mf == "trapezoid":
                    nn += 2

            gaps = jnp.zeros((nn - 1,))

            if init == "noisy":
                gaps = gaps + noise_scaler * jax.random.normal(key, gaps.shape)
            
            _mfs = [LeftShoulder(idx=0)]

    @classmethod
    def gaussian(
        cls,
        n_mfs: int,
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        init: str = "uniform",
        key: Array|None = None,
        noise_scaler: float = 0.1,
    ) -> "FuzzyVariable":
        if n_mfs < 1:
            raise ValueError(f"n_mfs must be >= 1 for Gaussian mfs, got {n_mfs}.")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", got {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy init must use a PRNGKey for initialization.")

        # NOTE: when looking at strictly gaussian mfs, ease constraint of binding to minval and maxval
        gaps = jnp.zeros((n_mfs + 1,))
        raw_sigs = jnp.zeros((n_mfs,))

        if init == "noisy":
            keys = jax.random.split(key, 2)
            gaps = gaps + noise_scaler * jax.random.normal(keys[0], gaps.shape)
            raw_sigs = raw_sigs + noise_scaler * jax.random.normal(keys[1], raw_sigs.shape)

        params = ParamBank(gaps=gaps, raw_sigmas=raw_sigs)
        mfs = [Gaussian(idx=i+1, sig_idx=i) for i in range(n_mfs)]

        return FuzzyVariable(params, tuple(mfs), minval, maxval, name)

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
        key: Array|None = None,
        noise_scaler: float = 0.1,
    ) -> "FuzzyVariable":
        # Sanity checks
        if kind not in ["triangle", "trapezoid"]:
            raise ValueError(f"kind must be \"triangle\" or \"trapezoid\", got {kind}.")

        if n_mfs <= 1:
            raise ValueError(f"n_mfs must be > 1 for ruspini partitions, got {n_mfs}.")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", got {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy init must use a PRNGKey for initialization.")

        if kind == "triangle":
            nn = n_mfs
        elif kind == "trapezoid":
            nn = 2*n_mfs

        gaps = jnp.zeros((nn - 1,))

        if init == "noisy":
            gaps = gaps + noise_scaler * jax.random.normal(key, gaps.shape)

        params = ParamBank(gaps=gaps)

        mfs = [LeftShoulder(idx=0)]

        for i in range(1, n_mfs-1):
            if kind == "triangle":
                mfs.append(Triangle(idx=i))
            elif kind == "trapezoid":
                mfs.append(Trapezoid(idx=2*i - 1))

        if kind == "triangle":
            mfs.append(RightShoulder(idx=n_mfs-1))
        elif kind == "trapezoid":
            mfs.append(RightShoulder(idx=2*n_mfs - 3))

        return FuzzyVariable(params, tuple(mfs), minval, maxval, name)

    def __call__(self, x: ScalarLike) -> Array:
        x = jnp.asarray(x)

        nodes = gaps2nodes(self.params.gaps, self.minval, self.maxval)

        if self.params.raw_sigmas is not None or len(self.params.raw_sigmas) == 0:
            sigs = sig_softplus(self.params.raw_sigmas)

        mus = [mf(x, nodes) if type(mf) is not Gaussian else mf(x, nodes, sigs) for mf in self.mfs]
        return jnp.stack(mus, axis=-1)  # (n_vals, n_mfs)

    @property
    def nodes(self) -> Array:
        return gaps2nodes(self.params.gaps, self.minval, self.maxval)

    @property
    def mf_params(self) -> list[Array]:
        nodes = gaps2nodes(self.params.gaps, self.minval, self.maxval)

        if self.params.raw_sigmas is not None:
            sigs = sig_softplus(self.params.raw_sigmas)

        params = []
        for mf in self.mfs:
            if type(mf) is not Gaussian:
                params.append(mf.get_params(nodes))
            else:
                params.append(mf.get_params(nodes, sigs))

        return params
