"""Backward-compatible entry point for portable drug batch inference.

All file locations are CLI arguments; no server or workstation paths are used.
See the root README and run ``python -m aethercell.doctor`` for asset help.
"""

import sys

from aethercell.batch_inference import main, parse_args
from aethercell.legacy_outputs import write_legacy_expression


if __name__ == "__main__":
    arguments = sys.argv[1:]
    parsed = parse_args(arguments)
    status = main(arguments)
    if status == 0:
        write_legacy_expression(parsed.output_dir)
    raise SystemExit(status)
