"""Tests for ONNX-to-IR import scaffolding."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from onnxpy.compiler.ir import CallOp, ConstantOp, IfElseOp, PointwiseOp
from onnxpy.compiler.opkinds import PointwiseKind
from onnxpy.compiler.parser import import_model


def test_imports_basic_pointwise_graph() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    summed = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])

    add_node = helper.make_node("Add", ["x", "y"], ["sum"])
    relu_node = helper.make_node("Relu", ["sum"], ["z"])

    graph = helper.make_graph(
        [add_node, relu_node],
        "basic_graph",
        [x, y],
        [z],
        value_info=[summed],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])

    function = import_model(model, infer_shapes=False).functions["basic_graph"]

    assert tuple(value.name for value in function.signature.inputs) == ("x", "y")
    assert len(function.body) == 2
    assert isinstance(function.body[0], PointwiseOp)
    assert isinstance(function.body[1], PointwiseOp)
    assert function.body[0].kind is PointwiseKind.ADD
    assert function.body[1].kind is PointwiseKind.RELU
    assert tuple(value.name for value in function.ret.values) == ("z",)


def test_imports_high_level_ops_as_call_ops() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])
    weight = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])

    conv_node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])
    graph = helper.make_graph([conv_node], "conv_graph", [x], [y], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])

    function = import_model(model, infer_shapes=False).functions["conv_graph"]

    assert len(function.body) == 2
    assert isinstance(function.body[0], ConstantOp)
    assert isinstance(function.body[1], CallOp)
    assert function.body[1].callee == "Conv"
    assert tuple(value.name for value in function.body[1].inputs) == ("x", "w")


def test_imports_if_as_structured_region() -> None:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])

    then_node = helper.make_node(
        "Constant",
        [],
        ["then_value"],
        value=helper.make_tensor("then_tensor", TensorProto.FLOAT, [1], [1.0]),
    )
    else_node = helper.make_node(
        "Constant",
        [],
        ["else_value"],
        value=helper.make_tensor("else_tensor", TensorProto.FLOAT, [1], [0.0]),
    )

    then_graph = helper.make_graph(
        [then_node],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_value", TensorProto.FLOAT, [1])],
    )
    else_graph = helper.make_graph(
        [else_node],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_value", TensorProto.FLOAT, [1])],
    )

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["out"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "if_graph", [cond], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])

    function = import_model(model, infer_shapes=False).functions["if_graph"]

    assert len(function.body) == 1
    assert isinstance(function.body[0], IfElseOp)
    assert function.body[0].cond.name == "cond"
    assert len(function.body[0].if_true.ops) == 1
    assert len(function.body[0].if_false.ops) == 1
    assert tuple(value.name for value in function.ret.values) == ("out",)
