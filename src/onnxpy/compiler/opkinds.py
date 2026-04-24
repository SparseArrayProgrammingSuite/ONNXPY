"""Enum-backed operator kinds for the ONNXPY IR."""

from __future__ import annotations

from enum import Enum
import math
import operator


def _identity(x: object) -> object:
    """Placeholder scalar identity function for enum-backed operator tags."""
    return x


def _relu(x: object) -> object:
    """Placeholder scalar ReLU marker used as a stable operator identity."""
    raise NotImplementedError("PointwiseKind.RELU is a symbolic IR operator.")


def _sigmoid(x: object) -> object:
    """Placeholder scalar sigmoid marker used as a stable operator identity."""
    raise NotImplementedError("PointwiseKind.SIGMOID is a symbolic IR operator.")


def _where(condition: object, if_true: object, if_false: object) -> object:
    """Placeholder scalar where marker used as a stable operator identity."""
    raise NotImplementedError("PointwiseKind.WHERE is a symbolic IR operator.")


class PointwiseKind(Enum):
    """Elementwise scalar functions lifted to tensors by the IR."""

    IDENTITY = _identity
    ADD = operator.add
    SUB = operator.sub
    MUL = operator.mul
    TRUEDIV = operator.truediv
    FLOORDIV = operator.floordiv
    MOD = operator.mod
    POW = operator.pow
    NEG = operator.neg
    POS = operator.pos
    ABS = operator.abs

    EQ = operator.eq
    NE = operator.ne
    LT = operator.lt
    LE = operator.le
    GT = operator.gt
    GE = operator.ge

    AND = operator.and_
    OR = operator.or_
    XOR = operator.xor
    NOT = operator.invert

    CEIL = math.ceil
    FLOOR = math.floor
    TRUNC = math.trunc
    EXP = math.exp
    LOG = math.log
    SQRT = math.sqrt
    SIN = math.sin
    COS = math.cos
    TANH = math.tanh

    RELU = _relu
    SIGMOID = _sigmoid
    WHERE = _where
    MAXIMUM = max
    MINIMUM = min


class ShapeKind(Enum):
    """Tensor shape and layout operations."""

    SHAPE_OF = "shape_of"
    SIZE_OF = "size_of"
    RANK_OF = "rank_of"
    CAST = "cast"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"
    EXPAND = "expand"
    BROADCAST_TO = "broadcast_to"
    CONCAT = "concat"
    SPLIT = "split"
    SLICE = "slice"
    GATHER = "gather"
    GATHER_ND = "gather_nd"
    SCATTER = "scatter"
    SCATTER_ND = "scatter_nd"
    PAD = "pad"
    TILE = "tile"
    REPEAT = "repeat"
    FLATTEN = "flatten"


class ReduceKind(Enum):
    """Tensor reductions."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PROD = "prod"
    ARGMAX = "argmax"
    ARGMIN = "argmin"


class SequenceKind(Enum):
    """Operations over ONNX-style sequences."""

    EMPTY = "empty"
    CONSTRUCT = "construct"
    LENGTH = "length"
    GET = "get"
    INSERT = "insert"
    ERASE = "erase"
    CONCAT = "concat"


class OptionalKind(Enum):
    """Operations over optional values."""

    NONE = "none"
    SOME = "some"
    HAS_VALUE = "has_value"
    GET = "get"

