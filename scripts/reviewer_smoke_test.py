#!/usr/bin/env python3
"""Run the reviewer-facing AetherCell workflows without external model/data assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


TASK_FILES = {
    "ac-rp": "ac_rp_predictions.csv",
    "synergy": "synergy_predictions.csv",
    "cdx": "cdx_predictions.csv",
    "tcga": "tcga_predictions.csv",
    "ac-dr": "ac_dr_predictions.csv",
}


def _run(command: list[str], *, root: Path, environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the two-epoch training integration test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    examples = root / "examples" / "data"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source) if not existing else os.pathsep.join((str(source), existing))

    with tempfile.TemporaryDirectory(prefix="aethercell-reviewer-") as temporary:
        output = Path(temporary)
        _run(
            [sys.executable, "-m", "aethercell.doctor", "--project-root", str(root)],
            root=root,
            environment=environment,
        )
        if not args.quick:
            _run(
                [sys.executable, "-m", "aethercell.train", "--smoke-test", "--output-dir", str(output / "training")],
                root=root,
                environment=environment,
            )
        for task, filename in TASK_FILES.items():
            _run(
                [
                    sys.executable,
                    "-m",
                    "aethercell.benchmarks",
                    task,
                    "--input",
                    str(examples / filename),
                    "--output-dir",
                    str(output / task),
                ],
                root=root,
                environment=environment,
            )

    print("PASS: reviewer smoke test completed; temporary outputs were cleaned up.")
    print("For full assets, run: aethercell-doctor --full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
