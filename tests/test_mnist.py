"""End-to-end round-trip test against ``tmp/models/mnist-8.onnx``.

This is the project's definition of done: compile the real ONNX model,
import the generated module, run it, and require the output to match
onnxruntime within tolerance on a handful of random inputs.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper

from onnxpy import compile_model

_MNIST_PATH = Path(__file__).resolve().parents[1] / "tmp" / "models" / "mnist-8.onnx"


@pytest.fixture(scope="module")
def mnist_module(tmp_path_factory):
    if not _MNIST_PATH.exists():
        pytest.skip(f"mnist-8 model not present at {_MNIST_PATH}")
    out_dir = tmp_path_factory.mktemp("mnist")
    py_path = out_dir / "mnist_8.py"
    compile_model(_MNIST_PATH, py_path)
    spec = importlib.util.spec_from_file_location(f"mnist_gen_{uuid.uuid4().hex}", py_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = onnx.load(str(_MNIST_PATH))
    tensor_inputs = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer
    }
    return module, tensor_inputs


@pytest.fixture(scope="module")
def ort_session():
    if not _MNIST_PATH.exists():
        pytest.skip(f"mnist-8 model not present at {_MNIST_PATH}")
    return ort.InferenceSession(str(_MNIST_PATH), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_mnist_matches_ort(mnist_module, ort_session, seed, array_namespace):
    rng = np.random.RandomState(seed)
    x = rng.randn(1, 1, 28, 28).astype(np.float32)
    expected = ort_session.run(["Plus214_Output_0"], {"Input3": x})[0]
    module, tensor_inputs = mnist_module
    values = {"Input3": x, **tensor_inputs}
    args = [values[name] for name in module.model_inputs]
    actual = np.asarray(module.model(*args, xp=array_namespace))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert int(np.argmax(actual)) == int(np.argmax(expected))
