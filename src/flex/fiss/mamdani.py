"""Defines Mamdani FIS class."""
from __future__ import annotations

from typing import Sequence, Self, Literal

import jax
import equinox as eqx
import jax.numpy as jnp

from .base_fis import BaseFIS
from ..fuzzy_variable import FuzzyVariable
from ..utils.types import Array


# NOTE: Mamdani needs to work a little differently... consequent rules needs to be able to adjust
class Mamdani(BaseFIS):
    output_vars: tuple[FuzzyVariable, ...]
    defuzz_method: str = eqx.field(static=True, default="com")

    @classmethod
    def init(
        cls,
        input_vars: tuple[FuzzyVariable, ...],
        output_vars: tuple[FuzzyVariable, ...],
        *,
        antecedents: Sequence[Sequence[tuple[int, int]]]|None = None,
        tnorm: Literal["prod", "min"] = "prod",
        name: str = "mamdani",
        eps: float = 1e-6,
    ) -> Self:
        raise NotImplementedError("Mamdani FIS has not been fully implemented yet.")

        input_vars, rb, n_mfs_max, name, eps = BaseFIS._build_base(
            input_vars,
            antecedents=antecedents,
            tnorm=tnorm,
            name=name,
            eps=eps,
        )