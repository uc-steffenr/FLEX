"""Defines fuzzy variable class."""
from __future__ import annotations

from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from .mfs import BaseMF, LeftShoulder, RightShoulder, Trapezoid, Triangle, Gaussian
from .utils.param_bank import ParamBank
from .utils.types import Array, ScalarLike
from .utils.unconstrained_parameters import gaps2nodes, sig_softplus


MFS = ["left_shoulder", "right_shoulder", "triangle", "trapezoid", "gaussian"]

# NOTE: overlaps must be enforced because of how nodes/gaps are defined
class FuzzyVariable(eqx.Module):
    params: ParamBank

    mfs: Tuple[BaseMF, ...] = eqx.field(static=True)
    minval: float = eqx.field(static=True, default=0.0)
    maxval: float = eqx.field(static=True, default=1.0)
    name: str = eqx.field(static=True, default="fv")

    # NOTE: if params are specified, this overrides init="uniform" or init="noisy"
    @classmethod 
    def manual(
        cls,
        mfs: Sequence[str],
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        mf_names: Sequence[str]|None = None,
        mode: str = "manual",
        init: str = "uniform",
        key: Array|None = None,
        noise_scaler: float = 0.1,
        params: Sequence[Sequence[float]]|None = None,
    ) -> "FuzzyVariable":
        # Sanity checks
        if any(mf not in MFS for mf in mfs):
            raise ValueError("Invalid MF type specified. Must be  ",
                             "\"left_shoulder\", \"right_shoulder\", ",
                             "\"triangle\", \"trapezoid\", or \"gaussian\".")

        if minval >= maxval:
            raise ValueError(f"minval must be stricly < maxval, got minval={minval} and maxval={maxval}")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", go {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy initi must use a PRNGKey for initialization.")

        if mode == "ruspini" and "gaussian" in mfs:
            raise ValueError("Ruspini partitions cannot be used with gaussian mfs.")

        if mode == "ruspini" and ("left_shoulder" != mfs[0] and "right_shoulder" != mfs[-1]):
            raise ValueError("If Ruspini partitions are used, the first mf must be \"left_shoulder\" and the last mf must be \"right_shoulder\".")

        n_mfs = len(mfs)

        if mf_names is not None and len(mf_names) != n_mfs:
            raise ValueError("Number of mf_names is not equal to the number of mfs requested.")

        if mf_names is None:
            mf_names = [f"mf_{i+1}" for i in range(n_mfs)]

        if mode == "ruspini":
            nn = 0
            for mf in mfs[1:-1]:
                if mf == "triangle":
                    nn += 1
                elif mf == "trapezoid":
                    nn += 2

            if params is None:
                gaps = jnp.zeros((nn + 1,))

                if init == "noisy":
                    gaps = gaps + noise_scaler * jax.random.normal(key, gaps.shape)

                params = ParamBank(gaps=gaps)
            else:
                raise NotImplementedError("Parameter specification for ruspini partitions is not implemented yet.")

            # Build mf list and assign indices
            _mfs = [LeftShoulder(idx=0, name=mf_names[0])]

            n = 1  # node counter
            for i in range(1, n_mfs-1):
                if mfs[i] == "triangle":
                    _mfs.append(Triangle(idx=n, name=mf_names[i]))
                    n += 1
                elif mfs[i] == "trapezoid":
                    _mfs.append(Trapezoid(idx=n, name=mf_names[i]))
                    n += 2

            _mfs.append(RightShoulder(idx=n, name=mf_names[-1]))

        elif mode == "manual":
            if "left_shoulder" in mfs and (mfs[0] != "left_shoulder" or mfs.count("left_shoulder") > 1):
                raise ValueError("left_shoulder can only be used once and it must be the first mf.")

            if "right_shoulder" in mfs and (mfs[-1] != "right_shoulder" or mfs.count("right_shoulder") > 1):
                raise ValueError("right_shoulder can only be used once and it must be the last mf.")

            nn = 2  # start and end nodes
            ns = 0
            for mf in mfs:
                if mf == "triangle":
                    nn += 1
                elif mf == "trapezoid":
                    nn += 2
                elif mf == "gaussian":
                    nn += 1
                    ns += 1

            if params is None:
                # if statement here ensures last mf isn't pinned to maxval
                gaps = jnp.zeros((nn - 1,))
                raw_sigs = jnp.ones((ns,)) * -2.5

                if init == "noisy":
                    keys = jax.random.split(key, 2)
                    gaps = gaps + noise_scaler * jax.random.normal(keys[0], gaps.shape)
                    raw_sigs = raw_sigs + noise_scaler * jax.random.normal(keys[1], raw_sigs.shape)

                params = ParamBank(gaps=gaps, raw_sigmas=raw_sigs)
            else:
                raise NotImplementedError("Parameter specification for manual mode is not implemented yet.")

            # Build mf list and assign indices
            _mfs = []
            n = 0
            sn = 0
            for i in range(n_mfs):
                # Check first mf
                if i == 0:
                    if mfs[i] == "left_shoulder":
                        _mfs.append(LeftShoulder(idx=0, name=mf_names[0]))
                        n += 1
                        continue
                    # Make sure first mf isn't pinned to minval
                    else:
                        n += 1

                # Check last mf
                if i == n_mfs - 1 and mfs[i] == "right_shoulder":
                    _mfs.append(RightShoulder(idx=nn-1, name=mf_names[-1]))
                    break

                if mfs[i] == "triangle":
                    _mfs.append(Triangle(idx=n, name=mf_names[i]))
                    n += 1
                elif mfs[i] == "trapezoid":
                    _mfs.append(Trapezoid(idx=n, name=mf_names[i]))
                    n += 2
                elif mfs[i] == "gaussian":
                    _mfs.append(Gaussian(idx=n, sig_idx=sn, name=mf_names[i]))
                    n += 1
                    sn += 1

        return FuzzyVariable(params, tuple(_mfs), minval, maxval, name)

    @classmethod
    def gaussian(
        cls,
        n_mfs: int,
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        mf_names: Sequence[str]|None = None,
        init: str = "uniform",
        key: Array|None = None,
        noise_scaler: float = 0.1,
    ) -> "FuzzyVariable":
        if n_mfs < 1:
            raise ValueError(f"n_mfs must be >= 1 for Gaussian mfs, got {n_mfs}.")

        if minval >= maxval:
            raise ValueError(f"minval must be stricly < maxval, got minval={minval} and maxval={maxval}")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", got {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy init must use a PRNGKey for initialization.")

        if mf_names is not None and len(mf_names) != n_mfs:
            raise ValueError("Number of mf_names is not equal to the number of mfs requested.")

        if mf_names is None:
            mf_names = [f"mf_{i+1}" for i in range(n_mfs)]

        # NOTE: when looking at strictly gaussian mfs, ease constraint of binding to minval and maxval
        gaps = jnp.zeros((n_mfs + 1,))
        raw_sigs = jnp.ones((n_mfs,)) * -2.5

        if init == "noisy":
            keys = jax.random.split(key, 2)
            gaps = gaps + noise_scaler * jax.random.normal(keys[0], gaps.shape)
            raw_sigs = raw_sigs + noise_scaler * jax.random.normal(keys[1], raw_sigs.shape)

        params = ParamBank(gaps=gaps, raw_sigmas=raw_sigs)

        mfs = [Gaussian(idx=i+1, sig_idx=i, name=mf_names[i]) for i in range(n_mfs)]

        return FuzzyVariable(params, tuple(mfs), minval, maxval, name)

    @classmethod
    def ruspini(
        cls,
        n_mfs: int,
        *,
        kind: str = "triangle",
        minval: float = 0.0,
        maxval: float = 1.0,
        name: str = "x",
        mf_names: Sequence[str]|None = None,
        init: str = "uniform",
        key: Array|None = None,
        noise_scaler: float = 0.1,
    ) -> "FuzzyVariable":
        # Sanity checks
        if n_mfs <= 1:
            raise ValueError(f"n_mfs must be > 1 for ruspini partitions, got {n_mfs}.")

        if kind not in ["triangle", "trapezoid"]:
            raise ValueError(f"kind must be \"triangle\" or \"trapezoid\", got {kind}.")

        if minval >= maxval:
            raise ValueError(f"minval must be stricly < maxval, got minval={minval} and maxval={maxval}")

        if init not in ["uniform", "noisy"]:
            raise ValueError(f"init must be \"uniform\" or \"noisy\", got {init}.")

        if init == "noisy" and key is None:
            raise ValueError("Noisy init must use a PRNGKey for initialization.")

        if mf_names is not None and len(mf_names) != n_mfs:
            raise ValueError("Number of mf_names is not equal to the number of mfs requested.")

        if mf_names is None:
            mf_names = [f"mf_{i+1}" for i in range(n_mfs)]

        if kind == "triangle":
            nn = n_mfs
        elif kind == "trapezoid":
            nn = 2*(n_mfs - 1)

        gaps = jnp.zeros((nn - 1,))

        if init == "noisy":
            gaps = gaps + noise_scaler * jax.random.normal(key, gaps.shape)

        params = ParamBank(gaps=gaps)

        mfs = [LeftShoulder(idx=0, name=mf_names[0])]

        for i in range(1, n_mfs-1):
            if kind == "triangle":
                mfs.append(Triangle(idx=i, name=mf_names[i]))
            elif kind == "trapezoid":
                mfs.append(Trapezoid(idx=2*i - 1, name=mf_names[i]))

        if kind == "triangle":
            mfs.append(RightShoulder(idx=nn-1, name=mf_names[-1]))
        elif kind == "trapezoid":
            mfs.append(RightShoulder(idx=nn-1, name=mf_names[-1]))

        return FuzzyVariable(params, tuple(mfs), minval, maxval, name)

    def __call__(self, x: ScalarLike) -> Array:
        x = jnp.asarray(x)

        nodes = gaps2nodes(self.params.gaps, self.minval, self.maxval)

        if self.params.raw_sigmas is not None and len(self.params.raw_sigmas) > 0:
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

    @property
    def mf_names(self) -> list[str]:
        return [mf.name for mf in self.mfs]
