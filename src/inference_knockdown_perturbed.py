"""Backward-compatible entry point for portable shRNA batch inference."""

import sys

from aethercell.batch_inference import main, parse_args
from aethercell.legacy_outputs import write_legacy_expression


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--mode" not in arguments:
        arguments = ["--mode", "shrna", *arguments]
    parsed = parse_args(arguments)
    status = main(arguments)
    if status == 0:
        write_legacy_expression(parsed.output_dir)
    raise SystemExit(status)
