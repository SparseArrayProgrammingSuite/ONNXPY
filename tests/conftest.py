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
from onnx import ModelProto

from onnxpy import compile_model


def _load_generated(py_path: Path):
    """Import a freshly generated module under a unique name."""

    spec = importlib.util.spec_from_file_location(f"onnxpy_gen_{uuid.uuid4().hex}", py_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arg_order(model: ModelProto) -> list[str]:
    init_names = {t.name for t in model.graph.initializer}
    return [vi.name for vi in model.graph.input if vi.name not in init_names]


def _compile_and_run(model: ModelProto, feed: dict[str, np.ndarray], tmp_path: Path):
    onnx_path = tmp_path / "m.onnx"
    onnx.save(model, str(onnx_path))
    py_path = tmp_path / "m.py"
    compile_model(str(onnx_path), str(py_path))
    module = _load_generated(py_path)
    args = [feed[name] for name in _arg_order(model)]
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
