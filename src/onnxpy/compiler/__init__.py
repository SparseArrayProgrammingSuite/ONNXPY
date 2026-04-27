"""ONNXPY compiler: direct ONNX → Python+NumPy source generation."""

from .emit import compile_model, compile_to_source
from .ops import supported_ops

__all__ = ["compile_model", "compile_to_source", "supported_ops"]
