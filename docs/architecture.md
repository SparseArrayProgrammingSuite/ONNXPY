# Architecture Notes

This template is intended for an ONNX-to-NumPy compiler with a staged pipeline:

1. Load ONNX model
2. Normalize ONNX graph into internal IR
3. Run analysis and rewrite passes
4. Lower into NumPy-friendly operations
5. Generate Python code or execute directly

Potential future modules:

- `parser.py`
- `ir.py`
- `normalize.py`
- `lower_numpy.py`
- `codegen.py`

