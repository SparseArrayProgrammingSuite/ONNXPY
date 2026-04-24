"""Type hierarchy for the ONNXPY IR."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TypeAlias


Dim: TypeAlias = int | str | None


@dataclass(frozen=True, slots=True)
class ONNXPYTypes(ABC):
    """Abstract base class for all ONNXPY IR types."""


@dataclass(frozen=True, slots=True)
class UnknownType(ONNXPYTypes):
    """Placeholder type used when ONNX metadata is absent or unresolved."""

    reason: str = "unknown"


@dataclass(frozen=True, slots=True)
class TensorType(ONNXPYTypes):
    """Type for dense tensor values."""

    dtype: str
    shape: tuple[Dim, ...]


@dataclass(frozen=True, slots=True)
class SequenceType(ONNXPYTypes):
    """Type for ONNX-style sequence values."""

    elem_type: ONNXPYTypes


@dataclass(frozen=True, slots=True)
class OptionalType(ONNXPYTypes):
    """Type for optional values."""

    elem_type: ONNXPYTypes
