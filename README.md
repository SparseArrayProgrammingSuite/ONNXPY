# ONNXPY

Template scaffold for an ONNX-to-NumPy compiler project with both a library and CLI layout.

## Project Goals

- Python library under `src/onnxpy/`
- CLI entrypoint exposed as `onnxpy`
- Test, lint, and type-check tooling
- Room for parser, IR, lowering, and code generation modules

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/)

## Getting Started

```bash
poetry install
poetry shell
```

Or run commands without activating a shell:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Installing Additional Libraries

Runtime dependency:

```bash
poetry add <package-name>
```

Development dependency:

```bash
poetry add --group dev <package-name>
```

## CLI

The CLI entrypoint is reserved as:

```bash
poetry run onnxpy --help
```

The implementation is intentionally left as a placeholder in this template.

## Suggested Layout

```text
src/onnxpy/
  cli.py
  compiler/
    parser.py
    ir.py
    normalize.py
    lower_numpy.py
    codegen.py
tests/
docs/
```

## Common Commands

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

