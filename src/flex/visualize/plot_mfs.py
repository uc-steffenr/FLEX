"""Defines method to plot membership functions."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

import jax.numpy as jnp

from ..fuzzy_variable import FuzzyVariable
from ..mfs import LeftShoulder, RightShoulder, Triangle, Trapezoid, Gaussian
from ..mfs.functions import gaussian


def plot_mfs(fv: FuzzyVariable, path: str|None = None, show: bool=True) -> None:
    var_name = fv.name
    mf_names = fv.mf_names
    params = fv.mf_params

    fig, ax = plt.subplots(1, 1)

    ax.set_title(f"{var_name} Membership Functions")

    xmin = fv.minval - 0.5
    xmax = fv.maxval + 0.5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(f"{var_name} Values")
    ax.set_ylabel(f"Membership Value")

    for i, mf in enumerate(fv.mfs):
        xs = params[i]
        if type(mf) == Triangle:
            assert len(xs) == 3
            ys = [0.0, 1.0, 0.0]

        elif type(mf) == Trapezoid:
            assert len(xs) == 4
            ys = [0.0, 1.0, 1.0, 0.0]

        elif type(mf) == LeftShoulder:
            assert len(xs) == 2
            xs = [xmin, xs[0], xs[1]]
            ys = [1.0, 1.0, 0.0]

        elif type(mf) == RightShoulder:
            assert len(xs) == 2
            xs = [xs[0], xs[1], xmax]
            ys = [0.0, 1.0, 1.0]

        elif type(mf) == Gaussian:
            assert len(xs) == 2
            sig = xs[0]
            mu = xs[1]
            xs = jnp.linspace(xmin, xmax, 1000)
            ys = gaussian(xs, sig, mu)

        else:
            raise ValueError(f"Specified MF is not recognized and cannot be plotted.")

        ax.plot(xs, ys, label=mf_names[i])

    ax.set_aspect("equal")
    ax.grid()
    ax.legend()

    if show:
        plt.show()

    if path is not None:
        fig.savefig(path)

    plt.close()
