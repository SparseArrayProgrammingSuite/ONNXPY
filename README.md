# onnxpy

Compile an ONNX model directly to a Python file backed by NumPy. Each
ONNX node becomes one Python statement. Weights are written to a
`.safetensors` sidecar alongside the generated module.

## Install

Requires Python 3.11+ and Poetry.

```bash
poetry install
```

## Usage

Compile:

```bash
poetry run onnxpy compile model.onnx -o out/model.py
```

Outputs `out/model.py` and `out/model.safetensors`.

Flags:

- `--weights PATH` write the sidecar somewhere other than next to the `.py`
- `--function-name NAME` rename the generated callable (default `model`)
- `--xp NAME` change the default array module baked into the signature (default `np`)

List supported ops:

```bash
poetry run onnxpy ops
```

Library form:

```python
from onnxpy import compile_model

compile_model("model.onnx", "out/model.py")
```

## Generated module

```python
from out.model import model
import numpy as np

y = model(np.zeros((1, 1, 28, 28), dtype=np.float32))
```

The generated function takes an `xp=np` keyword argument. Any
array-API-conforming module is accepted. `Conv`, `MaxPool`,
`AveragePool`, and ND padding fall back to NumPy internally and convert
the result via `xp.asarray`.

## Supported ops

`Add, AveragePool, Cast, Concat, Constant, Conv, Div, Flatten, Gemm,
GlobalAveragePool, Identity, LogSoftmax, MatMul, MaxPool, Mul, Relu,
Reshape, Sigmoid, Softmax, Sub, Tanh, Transpose`.

Unsupported op types raise `NotImplementedError` at compile time.

## Tests

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

`tests/test_ops.py` builds tiny graphs with `onnx.helper`, compiles
them, and compares the generated output against `onnxruntime`.
`tests/test_mnist.py` round-trips `tmp/models/mnist-8.onnx` against
`onnxruntime` on five seeds within `atol=rtol=1e-4`.

## Layout

```
src/onnxpy/
  cli.py
  compiler/
    emit.py     # graph walk, source assembly, safetensors sidecar
    ops.py      # per-op emitters
    helpers.py  # runtime preamble inlined into generated modules
tests/
docs/
  architecture.md
```

## Limitations

- No control flow (`If`, `Loop`, `Scan`).
- No sequence or optional types.
- No multi-output ops.
- Conv and pooling use `sliding_window_view`; correct but not fast.
- Modern op semantics only; pre-opset-13 quirks are not normalized.
