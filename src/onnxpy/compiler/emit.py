"""Direct ONNX → Python source generator.

Walks an :class:`onnx.ModelProto` in topological order and emits a
self-contained Python module that, when imported, exposes a callable
matching the ONNX graph's signature. Initializers and inline ``Constant``
op tensors are written to a ``.safetensors`` sidecar that the generated
module loads at import time.
"""

from __future__ import annotations

import keyword
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, numpy_helper
from safetensors.numpy import save_file as _save_safetensors

from .helpers import HELPERS_SOURCE
from .ops import EMITTERS

_RESERVED_NAMES = frozenset(
    {
        "np",
        "math",
        "xp",
        "Path",
        "_WEIGHTS",
        "_WEIGHTS_PATH",
        "_onnxpy_load_safetensors",
    }
)


@dataclass
class _NameAllocator:
    """Sanitize ONNX SSA names into unique Python identifiers."""

    used: set[str] = field(default_factory=set)
    mapping: dict[str, str] = field(default_factory=dict)

    def allocate(self, name: str, *, prefix: str = "v") -> str:
        if name in self.mapping:
            return self.mapping[name]
        cleaned = re.sub(r"\W+", "_", name).strip("_").lower()
        if not cleaned:
            cleaned = prefix
        if cleaned[0].isdigit():
            cleaned = f"{prefix}_{cleaned}"
        if keyword.iskeyword(cleaned) or cleaned in _RESERVED_NAMES:
            cleaned = f"{cleaned}_"
        candidate = cleaned
        counter = 1
        while candidate in self.used:
            candidate = f"{cleaned}_{counter}"
            counter += 1
        self.used.add(candidate)
        self.mapping[name] = candidate
        return candidate


@dataclass
class _EmitContextImpl:
    """Concrete implementation of :class:`onnxpy.compiler.ops.EmitContext`."""

    names: _NameAllocator
    weights: dict[str, np.ndarray]
    _const_counter: int = 0

    def name(self, ssa_name: str) -> str:
        if not ssa_name:
            raise ValueError("Empty SSA name encountered while emitting an op.")
        return self.names.allocate(ssa_name)

    def attr(self, node: NodeProto, name: str, default: Any = None) -> Any:
        for a in node.attribute:
            if a.name == name:
                return _attr_value(a)
        return default

    def attr_tensor(self, node: NodeProto, name: str) -> np.ndarray | None:
        for a in node.attribute:
            if a.name == name and a.type == AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t)
        return None

    def literal(self, value: Any) -> str:
        return _literal(value)

    def register_constant(self, hint: str, array: np.ndarray) -> str:
        key = hint or f"_const_{self._const_counter}"
        if key in self.weights:
            self._const_counter += 1
            key = f"{hint}__const_{self._const_counter}"
        self.weights[key] = array
        return f'xp.asarray(_WEIGHTS[{key!r}])'


def _attr_value(a: AttributeProto) -> Any:
    t = a.type
    if t == AttributeProto.INT:
        return int(a.i)
    if t == AttributeProto.FLOAT:
        return float(a.f)
    if t == AttributeProto.STRING:
        return a.s.decode("utf-8")
    if t == AttributeProto.INTS:
        return [int(v) for v in a.ints]
    if t == AttributeProto.FLOATS:
        return [float(v) for v in a.floats]
    if t == AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in a.strings]
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    type_name = AttributeProto.AttributeType.Name(t)
    raise NotImplementedError(f"Attribute type {type_name} not supported")


def _literal(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        inner = ", ".join(_literal(v) for v in value)
        return f"[{inner}]"
    return repr(value)


def compile_model(
    onnx_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    weights_path: str | os.PathLike[str] | None = None,
    function_name: str = "model",
    xp_default: str = "np",
) -> Path:
    """Compile an ONNX file to a Python module + safetensors sidecar.

    Returns the path to the generated ``.py`` file.
    """

    onnx_file = Path(onnx_path)
    out_file = Path(out_path)
    weights_file = (
        Path(weights_path) if weights_path is not None else out_file.with_suffix(".safetensors")
    )

    model = onnx.load(str(onnx_file))
    source, weights = compile_to_source(
        model,
        function_name=function_name,
        xp_default=xp_default,
        weights_path=weights_file,
        relative_to=out_file.parent,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(source, encoding="utf-8")

    weights_file.parent.mkdir(parents=True, exist_ok=True)
    if weights:
        _save_safetensors(
            {k: np.ascontiguousarray(v) for k, v in weights.items()},
            str(weights_file),
        )
    elif weights_file.exists():
        weights_file.unlink()
    return out_file


def compile_to_source(
    model: ModelProto,
    *,
    function_name: str = "model",
    xp_default: str = "np",
    weights_path: Path,
    relative_to: Path | None = None,
) -> tuple[str, dict[str, np.ndarray]]:
    """Return the generated Python source and the dict of tensors to persist."""

    graph = model.graph
    _validate_op_coverage(graph)

    names = _NameAllocator()
    weights: dict[str, np.ndarray] = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer
    }
    initializer_names = set(weights)

    real_inputs = [vi for vi in graph.input if vi.name not in initializer_names]
    arg_names = [names.allocate(vi.name, prefix="x") for vi in real_inputs]
    init_local_names = {
        tensor.name: names.allocate(tensor.name, prefix="w") for tensor in graph.initializer
    }

    ctx = _EmitContextImpl(names=names, weights=weights)

    body: list[str] = []
    for tensor in graph.initializer:
        local = init_local_names[tensor.name]
        body.append(f'    {local} = xp.asarray(_WEIGHTS[{tensor.name!r}])')

    for node in graph.node:
        if not node.output:
            raise ValueError(f"Node of type {node.op_type!r} has no outputs.")
        if len(node.output) > 1:
            raise NotImplementedError(
                f"Multi-output op {node.op_type!r} not supported in MVP",
            )
        emitter = EMITTERS[node.op_type]
        rhs = emitter(ctx, node)
        lhs = ctx.name(node.output[0])
        body.append(f"    {lhs} = {rhs}")

    if not graph.output:
        raise ValueError("Graph has no outputs.")
    output_names = [ctx.name(o.name) for o in graph.output]
    if len(output_names) == 1:
        body.append(f"    return {output_names[0]}")
    else:
        body.append(f"    return ({', '.join(output_names)},)")

    signature = f"def {function_name}({', '.join(arg_names)}, *, xp={xp_default}):"
    weights_rel = _format_relative_path(weights_path, relative_to or weights_path.parent)
    source = _assemble_module(weights_rel, signature, body, has_weights=bool(weights))
    return source, weights


def _assemble_module(
    weights_rel: str, signature: str, body: list[str], *, has_weights: bool
) -> str:
    header: list[str] = [
        '"""Generated by onnxpy — do not edit by hand."""',
        "from __future__ import annotations",
        "",
    ]
    if has_weights:
        header += [
            "from pathlib import Path",
            "",
            "import numpy as np",
            "from safetensors.numpy import load_file as _onnxpy_load_safetensors",
            "",
            f"_WEIGHTS_PATH = Path(__file__).resolve().parent / {weights_rel!r}",
            "_WEIGHTS = _onnxpy_load_safetensors(str(_WEIGHTS_PATH))",
        ]
    else:
        header += ["import numpy as np"]
    header += ["", "", HELPERS_SOURCE.rstrip(), "", signature]
    if not body:
        body = ["    pass"]
    return "\n".join(header + body).rstrip() + "\n"


def _format_relative_path(path: Path, relative_to: Path) -> str:
    try:
        rel = os.path.relpath(path.resolve(), relative_to.resolve())
    except ValueError:
        rel = str(path.resolve())
    return rel.replace(os.sep, "/")


def _validate_op_coverage(graph: GraphProto) -> None:
    seen = {n.op_type for n in graph.node}
    unsupported = sorted(seen - set(EMITTERS))
    if unsupported:
        raise NotImplementedError(
            f"Unsupported ONNX op types in graph: {', '.join(unsupported)}",
        )


__all__ = ["compile_model", "compile_to_source"]
