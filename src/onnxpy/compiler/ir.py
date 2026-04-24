"""Structured SSA IR for the ONNXPY compiler.

This IR is intentionally higher level than a CFG-based SSA form:

- Functions are sequences of operations.
- There are no basic blocks, jumps, or phi nodes.
- Control flow is represented structurally via region-based ops such as
  ``IfElseOp``, ``LoopOp``, and ``ScanOp``.
- Higher-level neural-network operators can remain outside the core IR and be
  expressed as macro/function calls that expand into this core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from .opkinds import OptionalKind, PointwiseKind, ReduceKind, SequenceKind, ShapeKind
from .types import ONNXPYTypes

AttributeValue: TypeAlias = object


@dataclass(frozen=True, slots=True)
class Value:
    """A named SSA value."""

    name: str
    type: ONNXPYTypes


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    """Function input and output types."""

    inputs: tuple[Value, ...]
    outputs: tuple[ONNXPYTypes, ...]


@dataclass(frozen=True, slots=True)
class Region:
    """A nested, structured op body with explicit parameters and yielded values."""

    inputs: tuple[Value, ...]
    ops: tuple[Op, ...]
    yields: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Op:
    """Base class for all structured SSA operations."""

    outputs: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class ConstantOp(Op):
    """Creates one or more constant SSA values."""

    value: AttributeValue


@dataclass(frozen=True, slots=True)
class PointwiseOp(Op):
    """Applies a scalar function elementwise across tensor inputs."""

    kind: PointwiseKind
    inputs: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class ShapeOp(Op):
    """Applies a tensor shape/layout transformation or indexing operation."""

    kind: ShapeKind
    inputs: tuple[Value, ...]
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True, slots=True)
class ReduceOp(Op):
    """Reduces one tensor input along one or more axes."""

    kind: ReduceKind
    input: Value
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True, slots=True)
class SequenceOp(Op):
    """Performs a sequence operation."""

    kind: SequenceKind
    inputs: tuple[Value, ...]
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True, slots=True)
class OptionalOp(Op):
    """Performs an optional-value operation."""

    kind: OptionalKind
    inputs: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class IfElseOp(Op):
    """Structured conditional with explicit outputs from each branch."""

    cond: Value
    if_true: Region
    if_false: Region


@dataclass(frozen=True, slots=True)
class LoopOp(Op):
    """Structured loop with explicit loop-carried SSA values."""

    max_trip_count: Value | None
    initial_cond: Value | None
    carried: tuple[Value, ...]
    body: Region


@dataclass(frozen=True, slots=True)
class ScanOp(Op):
    """Structured scan with explicit carried state and per-step scan outputs."""

    initial_state: tuple[Value, ...]
    scan_inputs: tuple[Value, ...]
    body: Region
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True, slots=True)
class CallOp(Op):
    """Calls a non-core helper or macro-style function.

    This is the hook for higher-level operators such as Conv, Pool, Gemm, or
    BatchNormalization before they are expanded into the core IR.
    """

    callee: str
    inputs: tuple[Value, ...]
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True, slots=True)
class ReturnOp:
    """Function terminator for the top-level body."""

    values: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Function:
    """A structured SSA function body."""

    name: str
    signature: FunctionSignature
    body: tuple[Op, ...]
    ret: ReturnOp
    attrs: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(slots=True)
class Module:
    """A collection of named functions plus optional metadata."""

    functions: dict[str, Function] = field(default_factory=dict)
    attrs: dict[str, AttributeValue] = field(default_factory=dict)
