# onnxpy

Compile an ONNX model directly to a Python file backed by an Array
API-compatible namespace. Each ONNX node becomes one Python statement.
Initializers and inline tensor constants are exposed as ordinary
computation inputs.

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

Outputs `out/model.py`.

Flags:

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
from out.model import model, model_inputs
import numpy as np

values = {
    "input": np.zeros((1, 1, 28, 28), dtype=np.float32),
    # Add initializer and Constant tensor values here.
}
args = [values[name] for name in model_inputs]
y = model(*args)
```

The generated module exports `model_inputs`, the full call order, and
`tensor_inputs`, the subset that came from ONNX initializers or tensor
`Constant` nodes. The generated function takes an `xp=np` keyword
argument. Any Array API-conforming namespace is accepted.

## SAPS Conv-2 helper

The Conv-2 SAPS benchmark can be regenerated as a single module with
the compiled ONNXPY model inlined:

```bash
poetry run python scripts/print_conv2lth_numpy.py --write-saps-benchmark
```

Without `--write-saps-benchmark`, the same script prints the generated
benchmark source to stdout. Use `--model-only` to print just the
generated Conv-2 model source.

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
    emit.py     # graph walk and source assembly
    ops.py      # per-op emitters
    helpers.py  # runtime preamble inlined into generated modules
scripts/
  print_conv2lth_numpy.py
tests/
docs/
  architecture.md
```

## Limitations

- No control flow (`If`, `Loop`, `Scan`).
- No sequence or optional types.
- No multi-output ops.
- Conv and pooling lower through `xp.unfold` when the backend exposes it.
- Modern op semantics only; pre-opset-13 quirks are not normalized.
