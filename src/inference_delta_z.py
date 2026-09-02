"""Backward-compatible entry point for portable drug batch inference.

The historical script contained workstation-specific paths.  This wrapper now
uses the tested ``aethercell-batch-infer`` CLI and writes expression, delta and
delta-z outputs together.  Run ``python -m aethercell.doctor`` if assets are
missing.
"""

import sys

from aethercell.batch_inference import main, parse_args
from aethercell.legacy_outputs import write_legacy_delta_z


if __name__ == "__main__":
    arguments = sys.argv[1:]
    parsed = parse_args(arguments)
    status = main(arguments)
    if status == 0:
        write_legacy_delta_z(parsed.output_dir)
    raise SystemExit(status)
