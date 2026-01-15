"""Defines method to plot membership functions."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ..fuzzy_variable import FuzzyVariable
from ..mfs import LeftShoulder, RightShoulder, Triangle, Trapezoid, Gaussian


def plot_mfs(fv: FuzzyVariable, path: str, show: bool=False) -> None:
    pass
