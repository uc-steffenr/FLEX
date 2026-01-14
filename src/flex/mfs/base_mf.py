"""Define Base Membership Function Class."""
from __future__ import annotations

import abc

import equinox as eqx

from ..utils.types import Array


class BaseMF(eqx.Module, abc.ABC):
    """Membership funciton interface.
    """
    idx: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="", kw_only=True)
    eps: float = eqx.field(static=True, default=1e-12, kw_only=True)

    @abc.abstractmethod
    def __call__(self, x: Array, nodes: Array) -> Array:
        raise NotImplementedError("__call__ is not implemented for base MF class.")

    @abc.abstractmethod
    def get_params(self, nodes: Array) -> Array:
        raise NotImplementedError("get_params is not implemented for base MF class.")
