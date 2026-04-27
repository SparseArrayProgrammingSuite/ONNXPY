"""Per-op emitters that turn an ONNX ``NodeProto`` into a Python expression.

Each emitter receives an :class:`EmitContext` (defined as a Protocol so the
emit module can supply the concrete implementation) and the ONNX node, and
returns the right-hand side of the assignment that produces the node's
single output. The orchestrator in ``emit.py`` is responsible for emitting
``<lhs> = <rhs>`` and managing the surrounding state.

Multi-output ops are not part of the MVP and will raise here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from onnx import NodeProto


class EmitContext(Protocol):
    """Minimal surface that op emitters depend on."""

    def name(self, ssa_name: str) -> str:
        """Map an ONNX SSA name to its Python identifier."""

    def attr(self, node: NodeProto, name: str, default: Any = None) -> Any:
        """Return the Python value of ``node``'s attribute ``name``."""

    def attr_tensor(self, node: NodeProto, name: str) -> np.ndarray | None:
        """Return the TensorProto attribute ``name`` as a numpy array, or None."""

    def literal(self, value: Any) -> str:
        """Render a Python literal as source code (quotes, tuples, etc.)."""

    def register_constant(self, hint: str, array: np.ndarray) -> str:
        """Store ``array`` in the safetensors sidecar and return an expr that fetches it."""


Emitter = Callable[[EmitContext, NodeProto], str]


def _require_single_output(node: NodeProto) -> None:
    if len(node.output) != 1:
        raise NotImplementedError(
            f"{node.op_type} produces {len(node.output)} outputs; "
            "multi-output ops are not in the MVP",
        )


# Pointwise ops mapped to a Python operator. Each op has 2 inputs.
_BINARY_OPERATORS = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
}


def _emit_binary(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    op = _BINARY_OPERATORS[node.op_type]
    lhs = ctx.name(node.input[0])
    rhs = ctx.name(node.input[1])
    return f"({lhs} {op} {rhs})"


def _emit_relu(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return f"xp.maximum({ctx.name(node.input[0])}, 0)"


def _emit_sigmoid(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    return f"(1 / (1 + xp.exp(-{x})))"


def _emit_tanh(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return f"xp.tanh({ctx.name(node.input[0])})"


def _emit_identity(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return ctx.name(node.input[0])


def _emit_matmul(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    a = ctx.name(node.input[0])
    b = ctx.name(node.input[1])
    return f"({a} @ {b})"


def _emit_gemm(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    a = ctx.name(node.input[0])
    b = ctx.name(node.input[1])
    c = ctx.name(node.input[2]) if len(node.input) > 2 and node.input[2] else "None"
    alpha = float(ctx.attr(node, "alpha", 1.0))
    beta = float(ctx.attr(node, "beta", 1.0))
    trans_a = int(ctx.attr(node, "transA", 0))
    trans_b = int(ctx.attr(node, "transB", 0))
    return (
        f"_onnxpy_gemm({a}, {b}, {c}, alpha={alpha!r}, beta={beta!r}, "
        f"trans_a={trans_a}, trans_b={trans_b}, xp=xp)"
    )


def _emit_reshape(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    data = ctx.name(node.input[0])
    shape = ctx.name(node.input[1])
    allow_zero = int(ctx.attr(node, "allowzero", 0))
    return f"_onnxpy_reshape({data}, {shape}, allow_zero={allow_zero}, xp=xp)"


def _emit_flatten(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", 1))
    return f"_onnxpy_flatten({ctx.name(node.input[0])}, axis={axis}, xp=xp)"


def _emit_transpose(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    perm = ctx.attr(node, "perm", None)
    perm_expr = "None" if perm is None else ctx.literal(tuple(int(p) for p in perm))
    return f"_onnxpy_transpose({ctx.name(node.input[0])}, perm={perm_expr}, xp=xp)"


def _emit_concat(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", 0))
    inputs = ", ".join(ctx.name(i) for i in node.input)
    return f"_onnxpy_concat(({inputs},), axis={axis}, xp=xp)"


def _emit_softmax(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", -1))
    return f"_onnxpy_softmax({ctx.name(node.input[0])}, axis={axis}, xp=xp)"


def _emit_log_softmax(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", -1))
    return f"_onnxpy_log_softmax({ctx.name(node.input[0])}, axis={axis}, xp=xp)"


def _emit_global_average_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return f"_onnxpy_global_avg_pool({ctx.name(node.input[0])}, xp=xp)"


def _emit_average_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    kernel_shape = ctx.literal(tuple(int(k) for k in ctx.attr(node, "kernel_shape", ())))
    strides = _list_attr_expr(ctx, node, "strides")
    pads = _list_attr_expr(ctx, node, "pads")
    ceil_mode = int(ctx.attr(node, "ceil_mode", 0))
    auto_pad = ctx.literal(str(ctx.attr(node, "auto_pad", "NOTSET")))
    count_include_pad = int(ctx.attr(node, "count_include_pad", 0))
    return (
        f"_onnxpy_avg_pool({ctx.name(node.input[0])}, kernel_shape={kernel_shape}, "
        f"strides={strides}, pads={pads}, ceil_mode={ceil_mode}, "
        f"auto_pad={auto_pad}, count_include_pad={count_include_pad}, xp=xp)"
    )


def _emit_max_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    kernel_shape = ctx.literal(tuple(int(k) for k in ctx.attr(node, "kernel_shape", ())))
    strides = _list_attr_expr(ctx, node, "strides")
    pads = _list_attr_expr(ctx, node, "pads")
    dilations = _list_attr_expr(ctx, node, "dilations")
    ceil_mode = int(ctx.attr(node, "ceil_mode", 0))
    auto_pad = ctx.literal(str(ctx.attr(node, "auto_pad", "NOTSET")))
    return (
        f"_onnxpy_max_pool({ctx.name(node.input[0])}, kernel_shape={kernel_shape}, "
        f"strides={strides}, pads={pads}, dilations={dilations}, "
        f"ceil_mode={ceil_mode}, auto_pad={auto_pad}, xp=xp)"
    )


def _emit_conv(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    w = ctx.name(node.input[1])
    b = ctx.name(node.input[2]) if len(node.input) > 2 and node.input[2] else "None"
    kernel_shape = _list_attr_expr(ctx, node, "kernel_shape")
    strides = _list_attr_expr(ctx, node, "strides")
    pads = _list_attr_expr(ctx, node, "pads")
    dilations = _list_attr_expr(ctx, node, "dilations")
    group = int(ctx.attr(node, "group", 1))
    auto_pad = ctx.literal(str(ctx.attr(node, "auto_pad", "NOTSET")))
    return (
        f"_onnxpy_conv({x}, {w}, {b}, kernel_shape={kernel_shape}, "
        f"strides={strides}, pads={pads}, dilations={dilations}, "
        f"group={group}, auto_pad={auto_pad}, xp=xp)"
    )


def _emit_cast(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    to = int(ctx.attr(node, "to"))
    return f"_onnxpy_cast({ctx.name(node.input[0])}, to={to}, xp=xp)"


def _emit_constant(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    tensor = ctx.attr_tensor(node, "value")
    if tensor is not None:
        return ctx.register_constant(node.output[0], tensor)
    int_val = ctx.attr(node, "value_int", None)
    if int_val is not None:
        return f"_onnxpy_asarray({int(int_val)!r}, 'int64', xp)"
    ints = ctx.attr(node, "value_ints", None)
    if ints is not None:
        return f"_onnxpy_asarray({ctx.literal([int(i) for i in ints])}, 'int64', xp)"
    float_val = ctx.attr(node, "value_float", None)
    if float_val is not None:
        return f"_onnxpy_asarray({float(float_val)!r}, 'float32', xp)"
    floats = ctx.attr(node, "value_floats", None)
    if floats is not None:
        return f"_onnxpy_asarray({ctx.literal([float(f) for f in floats])}, 'float32', xp)"
    raise NotImplementedError("Constant op variant not supported in MVP (need 'value' or value_*)")


def _list_attr_expr(ctx: EmitContext, node: NodeProto, name: str) -> str:
    """Render an int-list attribute as a tuple literal, or ``None`` if absent."""

    value = ctx.attr(node, name, None)
    if value is None:
        return "None"
    return ctx.literal(tuple(int(v) for v in value))


EMITTERS: dict[str, Emitter] = {
    "Add": _emit_binary,
    "Sub": _emit_binary,
    "Mul": _emit_binary,
    "Div": _emit_binary,
    "Relu": _emit_relu,
    "Sigmoid": _emit_sigmoid,
    "Tanh": _emit_tanh,
    "Identity": _emit_identity,
    "MatMul": _emit_matmul,
    "Gemm": _emit_gemm,
    "Reshape": _emit_reshape,
    "Flatten": _emit_flatten,
    "Transpose": _emit_transpose,
    "Concat": _emit_concat,
    "Softmax": _emit_softmax,
    "LogSoftmax": _emit_log_softmax,
    "GlobalAveragePool": _emit_global_average_pool,
    "AveragePool": _emit_average_pool,
    "MaxPool": _emit_max_pool,
    "Conv": _emit_conv,
    "Cast": _emit_cast,
    "Constant": _emit_constant,
}


def supported_ops() -> frozenset[str]:
    """Return the set of ONNX op types this MVP knows how to emit."""

    return frozenset(EMITTERS)


__all__ = ["EMITTERS", "EmitContext", "Emitter", "supported_ops"]
