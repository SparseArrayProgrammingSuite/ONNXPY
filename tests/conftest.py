"""Shared test helpers for ONNXPY round-trip tests."""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort
import pytest
from onnx import ModelProto

from onnxpy import compile_to_source


class _NumpyWithUnfold:
    """Test array namespace implementing unfold with NumPy views."""

    def __getattr__(self, name):
        return getattr(np, name)

    def unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        rank = len(kernel_shape)
        axes = tuple(range(x.ndim - rank, x.ndim)) if axes is None else tuple(axes)
        strides = (1,) * rank if strides is None else strides
        dilations = (1,) * rank if dilations is None else dilations
        padding = ((0, 0),) * rank if padding is None else padding
        pad_width = [(0, 0)] * x.ndim
        for axis, pair in zip(axes, padding, strict=True):
            pad_width[axis] = pair
        padded = np.pad(x, pad_width, constant_values=fill_value)
        effective = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_shape, dilations, strict=True)
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded, effective, axis=axes)
        slices = [slice(None)] * windows.ndim
        for axis, stride in zip(axes, strides, strict=True):
            slices[axis] = slice(None, None, stride)
        for axis, dilation in enumerate(dilations, start=x.ndim):
            slices[axis] = slice(None, None, dilation)
        return windows[tuple(slices)]


@pytest.fixture(scope="session")
def array_namespace():
    return _NumpyWithUnfold()


def _compile_and_run(model: ModelProto, feed: dict[str, np.ndarray], xp):
    source, tensor_inputs = compile_to_source(model)
    scope = {"math": math}
    exec(source, scope)
    args = [feed[vi.name] for vi in model.graph.input if vi.name not in tensor_inputs]
    return scope["model"](xp, *args, *tensor_inputs.values())


def _run_ort(model: ModelProto, feed: dict[str, np.ndarray]):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {k: np.asarray(v) for k, v in feed.items()})
    return res[0] if len(res) == 1 else tuple(res)


@pytest.fixture
def assert_matches_ort(array_namespace):
    """Compile, execute, and compare against onnxruntime within tolerance."""

    def _do(
        model: ModelProto,
        feed: dict[str, np.ndarray],
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        actual = _compile_and_run(model, feed, array_namespace)
        expected = _run_ort(model, feed)
        if isinstance(expected, tuple):
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_allclose(np.asarray(a), e, atol=atol, rtol=rtol)
        else:
            np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)

    return _do
