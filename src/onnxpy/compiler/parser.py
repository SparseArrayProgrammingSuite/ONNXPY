"""Import ONNX models into the structured ONNXPY IR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import onnx
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)
from onnx import TensorProto as TensorProtoEnum
from onnx import numpy_helper, shape_inference

from .ir import (
    AttributeValue,
    CallOp,
    ConstantOp,
    Function,
    FunctionSignature,
    IfElseOp,
    LoopOp,
    Module,
    Op,
    OptionalOp,
    PointwiseOp,
    ReduceOp,
    Region,
    ReturnOp,
    ScanOp,
    SequenceOp,
    ShapeOp,
    Value,
)
from .opkinds import OptionalKind, PointwiseKind, ReduceKind, SequenceKind, ShapeKind
from .types import ONNXPYTypes, OptionalType, SequenceType, TensorType, UnknownType

_POINTWISE_BY_OPTYPE: dict[str, PointwiseKind] = {
    "Abs": PointwiseKind.ABS,
    "Add": PointwiseKind.ADD,
    "And": PointwiseKind.AND,
    "Ceil": PointwiseKind.CEIL,
    "Cos": PointwiseKind.COS,
    "Div": PointwiseKind.TRUEDIV,
    "Equal": PointwiseKind.EQ,
    "Exp": PointwiseKind.EXP,
    "Floor": PointwiseKind.FLOOR,
    "Greater": PointwiseKind.GT,
    "GreaterOrEqual": PointwiseKind.GE,
    "Identity": PointwiseKind.IDENTITY,
    "Less": PointwiseKind.LT,
    "LessOrEqual": PointwiseKind.LE,
    "Log": PointwiseKind.LOG,
    "Max": PointwiseKind.MAXIMUM,
    "Min": PointwiseKind.MINIMUM,
    "Mod": PointwiseKind.MOD,
    "Mul": PointwiseKind.MUL,
    "Neg": PointwiseKind.NEG,
    "Not": PointwiseKind.NOT,
    "Or": PointwiseKind.OR,
    "Pow": PointwiseKind.POW,
    "Relu": PointwiseKind.RELU,
    "Sigmoid": PointwiseKind.SIGMOID,
    "Sin": PointwiseKind.SIN,
    "Sqrt": PointwiseKind.SQRT,
    "Sub": PointwiseKind.SUB,
    "Tanh": PointwiseKind.TANH,
    "Where": PointwiseKind.WHERE,
    "Xor": PointwiseKind.XOR,
}

_SHAPE_BY_OPTYPE: dict[str, ShapeKind] = {
    "Cast": ShapeKind.CAST,
    "Concat": ShapeKind.CONCAT,
    "Expand": ShapeKind.EXPAND,
    "Flatten": ShapeKind.FLATTEN,
    "Gather": ShapeKind.GATHER,
    "GatherND": ShapeKind.GATHER_ND,
    "Pad": ShapeKind.PAD,
    "Reshape": ShapeKind.RESHAPE,
    "ScatterElements": ShapeKind.SCATTER,
    "ScatterND": ShapeKind.SCATTER_ND,
    "Shape": ShapeKind.SHAPE_OF,
    "Size": ShapeKind.SIZE_OF,
    "Slice": ShapeKind.SLICE,
    "Split": ShapeKind.SPLIT,
    "Squeeze": ShapeKind.SQUEEZE,
    "Tile": ShapeKind.TILE,
    "Transpose": ShapeKind.TRANSPOSE,
    "Unsqueeze": ShapeKind.UNSQUEEZE,
}

_REDUCE_BY_OPTYPE: dict[str, ReduceKind] = {
    "ArgMax": ReduceKind.ARGMAX,
    "ArgMin": ReduceKind.ARGMIN,
    "ReduceMax": ReduceKind.MAX,
    "ReduceMean": ReduceKind.MEAN,
    "ReduceMin": ReduceKind.MIN,
    "ReduceProd": ReduceKind.PROD,
    "ReduceSum": ReduceKind.SUM,
}

_SEQUENCE_BY_OPTYPE: dict[str, SequenceKind] = {
    "ConcatFromSequence": SequenceKind.CONCAT,
    "SequenceAt": SequenceKind.GET,
    "SequenceConstruct": SequenceKind.CONSTRUCT,
    "SequenceEmpty": SequenceKind.EMPTY,
    "SequenceErase": SequenceKind.ERASE,
    "SequenceInsert": SequenceKind.INSERT,
    "SequenceLength": SequenceKind.LENGTH,
}

_OPTIONAL_BY_OPTYPE: dict[str, OptionalKind] = {
    "OptionalGetElement": OptionalKind.GET,
    "OptionalHasElement": OptionalKind.HAS_VALUE,
}


@dataclass(frozen=True, slots=True)
class _GraphImportResult:
    """Imported graph body plus its explicit inputs and yielded outputs."""

    inputs: tuple[Value, ...]
    ops: tuple[Op, ...]
    outputs: tuple[Value, ...]


def load_model(model_path: str | Path) -> ModelProto:
    """Load an ONNX model from disk."""

    return onnx.load(Path(model_path))


def import_model(
    model_or_path: ModelProto | str | Path,
    *,
    function_name: str | None = None,
    infer_shapes: bool = True,
) -> Module:
    """Import an ONNX model or model path into an ONNXPY module."""

    model = load_model(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    importer = _Importer(_maybe_infer_shapes(model) if infer_shapes else model)
    return importer.import_module(function_name=function_name)


__all__ = ["import_model", "load_model"]


class _Importer:
    """Internal ONNX-to-IR importer."""

    def __init__(self, model: ModelProto) -> None:
        self._model = model

    def import_module(self, *, function_name: str | None = None) -> Module:
        """Import the model's main graph as a module with one function."""

        name = function_name or self._model.graph.name or "main"
        function = self._import_function(self._model.graph, name=name)
        opset_imports = {
            opset.domain: int(opset.version) for opset in self._model.opset_import
        }
        return Module(
            functions={function.name: function},
            attrs={
                "ir_version": int(self._model.ir_version),
                "producer_name": self._model.producer_name,
                "producer_version": self._model.producer_version,
                "domain": self._model.domain,
                "model_version": int(self._model.model_version),
                "doc_string": self._model.doc_string,
                "opset_imports": opset_imports,
            },
        )

    def _import_function(self, graph: GraphProto, *, name: str) -> Function:
        body = self._import_graph(graph, outer_scope={})
        signature = FunctionSignature(
            inputs=body.inputs,
            outputs=tuple(output.type for output in body.outputs),
        )
        return Function(
            name=name,
            signature=signature,
            body=body.ops,
            ret=ReturnOp(values=body.outputs),
            attrs=(("source_graph_name", graph.name),),
        )

    def _import_region(
        self,
        graph: GraphProto,
        *,
        outer_scope: Mapping[str, Value],
    ) -> Region:
        imported = self._import_graph(graph, outer_scope=outer_scope)
        return Region(inputs=imported.inputs, ops=imported.ops, yields=imported.outputs)

    def _import_graph(
        self,
        graph: GraphProto,
        *,
        outer_scope: Mapping[str, Value],
    ) -> _GraphImportResult:
        env: dict[str, Value] = dict(outer_scope)
        type_by_name = self._collect_types(graph)
        initializer_names = {initializer.name for initializer in graph.initializer}
        inputs: list[Value] = []
        ops: list[Op] = []

        for input_proto in graph.input:
            value = self._value_for_name(input_proto.name, type_by_name)
            env[input_proto.name] = value
            if input_proto.name not in initializer_names:
                inputs.append(value)

        for initializer in graph.initializer:
            value = env.get(initializer.name)
            if value is None:
                value = Value(
                    name=initializer.name,
                    type=type_by_name.get(
                        initializer.name,
                        self._type_from_tensor_proto(initializer),
                    ),
                )
                env[initializer.name] = value
            ops.append(ConstantOp(outputs=(value,), value=numpy_helper.to_array(initializer)))

        for node in graph.node:
            op = self._import_node(node=node, env=env, type_by_name=type_by_name)
            ops.append(op)
            for output in op.outputs:
                env[output.name] = output

        outputs = tuple(
            env.get(output.name, self._value_for_name(output.name, type_by_name))
            for output in graph.output
        )
        return _GraphImportResult(inputs=tuple(inputs), ops=tuple(ops), outputs=outputs)

    def _import_node(
        self,
        *,
        node: NodeProto,
        env: dict[str, Value],
        type_by_name: Mapping[str, ONNXPYTypes],
    ) -> Op:
        outputs = self._node_outputs(node, type_by_name)
        attrs = self._attribute_map(node)

        if node.op_type == "Constant":
            return ConstantOp(outputs=outputs, value=self._constant_value(attrs))

        if node.op_type == "If":
            cond = self._maybe_input(env, node.input[0])
            if cond is None:
                raise ValueError("If node is missing its condition input.")
            then_graph = self._graph_attr(attrs, "then_branch")
            else_graph = self._graph_attr(attrs, "else_branch")
            return IfElseOp(
                outputs=outputs,
                cond=cond,
                if_true=self._import_region(then_graph, outer_scope=env),
                if_false=self._import_region(else_graph, outer_scope=env),
            )

        if node.op_type == "Loop":
            body_graph = self._graph_attr(attrs, "body")
            max_trip_count = self._maybe_input(env, node.input[0]) if len(node.input) > 0 else None
            initial_cond = self._maybe_input(env, node.input[1]) if len(node.input) > 1 else None
            carried = tuple(
                input_value
                for input_name in node.input[2:]
                if (input_value := self._maybe_input(env, input_name)) is not None
            )
            return LoopOp(
                outputs=outputs,
                max_trip_count=max_trip_count,
                initial_cond=initial_cond,
                carried=carried,
                body=self._import_region(body_graph, outer_scope=env),
            )

        if node.op_type == "Scan":
            body_graph = self._graph_attr(attrs, "body")
            num_scan_inputs = cast(int, attrs["num_scan_inputs"])
            resolved_inputs = tuple(
                value
                for input_name in node.input
                if (value := self._maybe_input(env, input_name)) is not None
            )
            if num_scan_inputs < 0 or num_scan_inputs > len(resolved_inputs):
                raise ValueError(
                    f"Invalid num_scan_inputs={num_scan_inputs} for Scan node {node.name!r}.",
                )
            split_index = len(resolved_inputs) - num_scan_inputs
            scan_attrs = tuple(
                (key, value)
                for key, value in attrs.items()
                if key != "body"
            )
            return ScanOp(
                outputs=outputs,
                initial_state=resolved_inputs[:split_index],
                scan_inputs=resolved_inputs[split_index:],
                body=self._import_region(body_graph, outer_scope=env),
                attrs=scan_attrs,
            )

        if node.op_type == "Optional":
            kind = OptionalKind.SOME if any(node.input) else OptionalKind.NONE
            return OptionalOp(outputs=outputs, kind=kind, inputs=self._node_inputs(node, env))

        pointwise_kind = _POINTWISE_BY_OPTYPE.get(node.op_type)
        if pointwise_kind is not None:
            return PointwiseOp(outputs=outputs, kind=pointwise_kind, inputs=self._node_inputs(node, env))

        shape_kind = _SHAPE_BY_OPTYPE.get(node.op_type)
        if shape_kind is not None:
            return ShapeOp(
                outputs=outputs,
                kind=shape_kind,
                inputs=self._node_inputs(node, env),
                attrs=self._plain_attrs(attrs),
            )

        reduce_kind = _REDUCE_BY_OPTYPE.get(node.op_type)
        if reduce_kind is not None:
            reduce_inputs = self._node_inputs(node, env)
            if not reduce_inputs:
                raise ValueError(f"Reduce node {node.name!r} has no input.")
            return ReduceOp(
                outputs=outputs,
                kind=reduce_kind,
                input=reduce_inputs[0],
                attrs=self._plain_attrs(attrs),
            )

        sequence_kind = _SEQUENCE_BY_OPTYPE.get(node.op_type)
        if sequence_kind is not None:
            return SequenceOp(
                outputs=outputs,
                kind=sequence_kind,
                inputs=self._node_inputs(node, env),
                attrs=self._plain_attrs(attrs),
            )

        optional_kind = _OPTIONAL_BY_OPTYPE.get(node.op_type)
        if optional_kind is not None:
            return OptionalOp(outputs=outputs, kind=optional_kind, inputs=self._node_inputs(node, env))

        callee = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        fallback_attrs = (("onnx_name", node.name),) + self._plain_attrs(attrs)
        return CallOp(
            outputs=outputs,
            callee=callee,
            inputs=self._node_inputs(node, env),
            attrs=fallback_attrs,
        )

    def _collect_types(self, graph: GraphProto) -> dict[str, ONNXPYTypes]:
        type_by_name: dict[str, ONNXPYTypes] = {}
        for value_info in (*graph.input, *graph.output, *graph.value_info):
            value_type = self._type_from_value_info(value_info)
            if value_type is not None:
                type_by_name[value_info.name] = value_type
        for initializer in graph.initializer:
            type_by_name.setdefault(initializer.name, self._type_from_tensor_proto(initializer))
        return type_by_name

    def _type_from_value_info(self, value_info: ValueInfoProto) -> ONNXPYTypes | None:
        if value_info.type.WhichOneof("value") is None:
            return None
        return self._type_from_type_proto(value_info.type)

    def _type_from_type_proto(self, type_proto: TypeProto) -> ONNXPYTypes:
        value_kind = type_proto.WhichOneof("value")
        if value_kind == "tensor_type":
            tensor_type = type_proto.tensor_type
            dtype = TensorProtoEnum.DataType.Name(tensor_type.elem_type).lower()
            dims = tuple(self._shape_dim(dim) for dim in tensor_type.shape.dim)
            return TensorType(dtype=dtype, shape=dims)
        if value_kind == "sequence_type":
            return SequenceType(elem_type=self._type_from_type_proto(type_proto.sequence_type.elem_type))
        if value_kind == "optional_type":
            return OptionalType(elem_type=self._type_from_type_proto(type_proto.optional_type.elem_type))
        return UnknownType(reason=f"unsupported_type:{value_kind}")

    def _type_from_tensor_proto(self, tensor: TensorProto) -> TensorType:
        dtype = TensorProtoEnum.DataType.Name(tensor.data_type).lower()
        return TensorType(dtype=dtype, shape=tuple(int(dim) for dim in tensor.dims))

    def _shape_dim(self, dim: TensorShapeProto.Dimension) -> int | str | None:
        if dim.HasField("dim_value"):
            return int(dim.dim_value)
        if dim.dim_param:
            return dim.dim_param
        return None

    def _node_inputs(self, node: NodeProto, env: dict[str, Value]) -> tuple[Value, ...]:
        return tuple(
            value
            for input_name in node.input
            if (value := self._maybe_input(env, input_name)) is not None
        )

    def _node_outputs(
        self,
        node: NodeProto,
        type_by_name: Mapping[str, ONNXPYTypes],
    ) -> tuple[Value, ...]:
        return tuple(
            self._value_for_name(output_name, type_by_name)
            for output_name in node.output
            if output_name
        )

    def _value_for_name(
        self,
        name: str,
        type_by_name: Mapping[str, ONNXPYTypes],
    ) -> Value:
        return Value(name=name, type=type_by_name.get(name, UnknownType(reason=f"missing_type:{name}")))

    def _maybe_input(self, env: dict[str, Value], name: str) -> Value | None:
        if not name:
            return None
        value = env.get(name)
        if value is None:
            value = Value(name=name, type=UnknownType(reason=f"unresolved_value:{name}"))
            env[name] = value
        return value

    def _attribute_map(self, node: NodeProto) -> dict[str, AttributeValue]:
        return {attribute.name: self._attribute_value(attribute) for attribute in node.attribute}

    def _attribute_value(self, attribute: AttributeProto) -> AttributeValue:
        if attribute.type == AttributeProto.FLOAT:
            return float(attribute.f)
        if attribute.type == AttributeProto.INT:
            return int(attribute.i)
        if attribute.type == AttributeProto.STRING:
            return attribute.s.decode("utf-8")
        if attribute.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(attribute.t)
        if attribute.type == AttributeProto.GRAPH:
            return attribute.g
        if attribute.type == AttributeProto.FLOATS:
            return tuple(float(value) for value in attribute.floats)
        if attribute.type == AttributeProto.INTS:
            return tuple(int(value) for value in attribute.ints)
        if attribute.type == AttributeProto.STRINGS:
            return tuple(value.decode("utf-8") for value in attribute.strings)
        if attribute.type == AttributeProto.TENSORS:
            return tuple(numpy_helper.to_array(tensor) for tensor in attribute.tensors)
        if attribute.type == AttributeProto.GRAPHS:
            return tuple(attribute.graphs)
        if attribute.type == AttributeProto.TYPE_PROTO:
            return self._type_from_type_proto(attribute.tp)
        if attribute.type == AttributeProto.TYPE_PROTOS:
            return tuple(self._type_from_type_proto(type_proto) for type_proto in attribute.type_protos)
        return None

    def _graph_attr(self, attrs: Mapping[str, AttributeValue], name: str) -> GraphProto:
        value = attrs.get(name)
        if not isinstance(value, GraphProto):
            raise ValueError(f"Expected graph attribute {name!r}.")
        return value

    def _plain_attrs(
        self,
        attrs: Mapping[str, AttributeValue],
    ) -> tuple[tuple[str, AttributeValue], ...]:
        return tuple(
            (key, value)
            for key, value in attrs.items()
            if not isinstance(value, GraphProto) and not _is_graph_sequence(value)
        )

    def _constant_value(self, attrs: Mapping[str, AttributeValue]) -> AttributeValue:
        if "value" in attrs:
            return attrs["value"]
        if len(attrs) == 1:
            return next(iter(attrs.values()))
        return dict(attrs)


def _is_graph_sequence(value: AttributeValue) -> bool:
    return isinstance(value, tuple) and any(isinstance(item, GraphProto) for item in value)


def _maybe_infer_shapes(model: ModelProto) -> ModelProto:
    try:
        return shape_inference.infer_shapes(model)
    except Exception:
        return model
