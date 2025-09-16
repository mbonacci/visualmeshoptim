"""Generic utilities used across the :mod:`visualmeshoptim` package."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import torch


def get_logger(name: str = "visualmeshoptim") -> logging.Logger:
    """Return a package logger configured with a stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    return logger


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Return a valid :class:`torch.device` for the given argument."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TunableParameter:
    """Container describing a UI-editable parameter."""

    value: Any
    dtype: str = "float"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[Sequence[Any]] = None
    description: str = ""

    def copy(self) -> "TunableParameter":
        """Return a shallow copy of the parameter definition."""
        return replace(self)

    def _coerce(self, new_value: Any) -> Any:
        dtype = self.dtype.lower()
        if dtype == "float":
            coerced = float(new_value)
        elif dtype == "int":
            coerced = int(new_value)
        elif dtype == "bool":
            coerced = bool(new_value)
        elif dtype in {"choice", "enum"}:
            if self.options is None:
                raise ValueError("Choice parameters require an options list")
            if new_value not in self.options:
                raise ValueError(f"{new_value!r} not in allowed options {self.options}")
            coerced = new_value
        else:
            coerced = new_value
        if isinstance(coerced, (float, int)):
            if self.min_value is not None:
                coerced = max(self.min_value, coerced)
            if self.max_value is not None:
                coerced = min(self.max_value, coerced)
        return coerced

    def update(self, new_value: Any) -> None:
        """Update the stored value after coercion and validation."""
        self.value = self._coerce(new_value)


def ensure_numpy(array: Any) -> Any:
    """Return a NumPy array from torch tensors or sequences."""
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):
        return array.numpy()
    return array


def ensure_tensor(array: Any, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a tensor on the desired device and dtype."""
    tensor = torch.as_tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor
