"""Shared test helpers for ONNXPY round-trip tests."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import AttributeProto, ModelProto, numpy_helper

from onnxpy import compile_model


def _load_generated(py_path: Path):
    """Import a freshly generated module under a unique name."""

    spec = importlib.util.spec_from_file_location(f"onnxpy_gen_{uuid.uuid4().hex}", py_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_inputs(model: ModelProto) -> dict[str, np.ndarray]:
    values = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    const_counter = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        tensor_attr = next(
            (
                attr
                for attr in node.attribute
                if attr.name == "value" and attr.type == AttributeProto.TENSOR
            ),
            None,
        )
        if tensor_attr is None:
            continue
        key = node.output[0] if node.output else f"_const_{const_counter}"
        if key in values:
            const_counter += 1
            key = f"{key}__const_{const_counter}"
        values[key] = numpy_helper.to_array(tensor_attr.t)
    return values


def _compile_and_run(model: ModelProto, feed: dict[str, np.ndarray], tmp_path: Path):
    onnx_path = tmp_path / "m.onnx"
    onnx.save(model, str(onnx_path))
    py_path = tmp_path / "m.py"
    compile_model(str(onnx_path), str(py_path))
    module = _load_generated(py_path)
    values = {**feed, **_tensor_inputs(model)}
    args = [values[name] for name in module.model_inputs]
    return module.model(*args)


def _run_ort(model: ModelProto, feed: dict[str, np.ndarray]):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {k: np.asarray(v) for k, v in feed.items()})
    return res[0] if len(res) == 1 else tuple(res)


@pytest.fixture
def assert_matches_ort(tmp_path):
    """Compile, execute, and compare against onnxruntime within tolerance."""

    def _do(
        model: ModelProto,
        feed: dict[str, np.ndarray],
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        actual = _compile_and_run(model, feed, tmp_path)
        expected = _run_ort(model, feed)
        if isinstance(expected, tuple):
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_allclose(np.asarray(a), e, atol=atol, rtol=rtol)
        else:
            np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)

    return _do
