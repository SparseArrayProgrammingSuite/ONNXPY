"""``onnxpy`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .compiler import compile_model, supported_ops


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnxpy",
        description="Compile ONNX models directly to Python + NumPy source.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compile_p = sub.add_parser(
        "compile",
        help="Compile an ONNX file to a Python module + safetensors sidecar.",
    )
    compile_p.add_argument("model", type=Path, help="Path to the input .onnx file.")
    compile_p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path of the .py file to write.",
    )
    compile_p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path of the .safetensors sidecar (default: alongside the .py).",
    )
    compile_p.add_argument(
        "--function-name",
        default="model",
        help="Name of the generated callable (default: 'model').",
    )
    compile_p.add_argument(
        "--xp",
        default="np",
        help="Default xp module name to bake into the signature (default: 'np').",
    )

    sub.add_parser(
        "ops",
        help="List the ONNX op types this build of onnxpy supports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ops":
        for op in sorted(supported_ops()):
            print(op)
        return 0

    if args.command == "compile":
        out = compile_model(
            args.model,
            args.output,
            weights_path=args.weights,
            function_name=args.function_name,
            xp_default=args.xp,
        )
        print(f"wrote {out}")
        if args.weights is not None:
            print(f"weights {args.weights}")
        else:
            print(f"weights {out.with_suffix('.safetensors')}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse exits before this


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
