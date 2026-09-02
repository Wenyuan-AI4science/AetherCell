#!/usr/bin/env python3
"""Apply the reviewed deterministic-input patch to the released model package.

The pinned Hugging Face archive predates the reviewer fix and contains API/Skill
examples that silently synthesize cell profiles. This patch is deliberately
fail-closed: it updates only files whose SHA-256 matches the known release (or
this patch), and refuses to overwrite an unknown/customized package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PATCH_VERSION = "2026-09-02-deterministic-cell-context-v3"
FILES = {
    "models/transcriptome_prediction/transcriptome_inference.py": {
        "accepted": [
            "21881d186d0d9040fc07b767bca03fdfe04f34871b1ccbc87e9f9e800ff499c3",
            "aa822efab27d2a01958c6e33f8b244a09370aad67f13e5be9c139d395c935f91",
        ],
        "patched": "88e6434345947cbc5f3aadbac48abe8c2caec884d0f6c115806a2b6b78afec37",
    },
    "models/ic50_prediction/ic50_inference.py": {
        "accepted": [
            "e58507269d5a05b2919cbada33035fbceaeb6dc4b2c49c91b725ecd53e7e7037",
            "7e448207ccbae920ad68547a75d335c9cc6f1bebb1b7721a2dff947b28c47a12",
        ],
        "patched": "f9818bc50349186ef7469d34986c6ec98291e70d90410b57146aa9f07e6e58ea",
    },
    ".claude/skills/transcriptome-prediction/SKILL.md": {
        "accepted": [
            "e294598613e9f1d39addc11fe596394551f84a75c80ac4069bc7582c05bf7a10",
            "15b877a880191d9d95390463ce6ab6b4070ccc1988d7ba605a98d7c0cf6141eb",
        ],
        "patched": "825fe0db0251b370d704b4aa1b71d9421e9b0909528418a5c832b4e1bde33182",
    },
    ".claude/skills/ic50-prediction/SKILL.md": {
        "accepted": [
            "bba8b156955c0cbbf454e96b7bda7507f64551dff502141cd808922264765615",
            "a78c0e0168a0a7b237df5121f581dbf7a500f66473adb7a6d3220c73cf60d311",
        ],
        "patched": "bc15a510aa61bc3e3efc8513ab1bda4e526d331c9669775c60148f8a5782c9de",
    },
    ".claude/skills/drug-repurposing/SKILL.md": {
        "accepted": [
            "2ad66866346e8d113701902727a620aa475904c580a6be6a648bd1e8154809c8",
            "f33dd1a52a6880d45b97a0cc13db1710107d5f57ce677bbc81e6ce54132c84cf",
        ],
        "patched": "73ce4e13ba8b5298387ae7584286eab8cfb495953d572966249c8fbbe870b349",
    },
}
EXAMPLE_TARGET = "examples/api_context_examples.npz"
EXAMPLE_SHA256 = "84eb649700d6678819fb782d7ac20b97a55cd919d6117aec3d5914b141d391c6"
MANIFEST = "agent_ready_patch_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_package_root(path: Path) -> Path:
    candidates = [path, path / "aethercell-drug-discovery-v1.0.0"]
    for candidate in candidates:
        if (candidate / "models" / "transcriptome_prediction").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not find the extracted model package under {path}; expected "
        "aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction"
    )


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".aethercell-patch-tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def apply_patch(package_dir: Path, repository_root: Path | None = None) -> dict:
    repository_root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    patch_root = repository_root / "patches" / "agent_ready"
    package_root = find_package_root(package_dir)

    plan = []
    for relative, expected in FILES.items():
        source = patch_root / relative
        target = package_root / relative
        if not source.is_file() or sha256(source) != expected["patched"]:
            raise RuntimeError(f"repository patch asset is missing or corrupt: {source}")
        if not target.is_file():
            raise FileNotFoundError(f"released package file is missing: {target}")
        actual = sha256(target)
        if actual == expected["patched"]:
            action = "already_patched"
        elif actual in expected["accepted"]:
            action = "replace"
        else:
            raise RuntimeError(
                f"refusing to overwrite unknown/customized file {target}: sha256={actual}"
            )
        plan.append((relative, source, target, action))

    example_source = repository_root / "examples" / "data" / "api_context_examples.npz"
    if not example_source.is_file() or sha256(example_source) != EXAMPLE_SHA256:
        raise RuntimeError(f"real example context is missing or corrupt: {example_source}")
    example_target = package_root / EXAMPLE_TARGET
    if example_target.exists() and sha256(example_target) != EXAMPLE_SHA256:
        raise RuntimeError(f"refusing to overwrite unknown example context: {example_target}")

    for relative, source, target, action in plan:
        if action == "replace":
            _atomic_copy(source, target)
        if sha256(target) != FILES[relative]["patched"]:
            raise RuntimeError(f"post-copy verification failed: {target}")
    if not example_target.exists():
        _atomic_copy(example_source, example_target)
    if sha256(example_target) != EXAMPLE_SHA256:
        raise RuntimeError(f"post-copy verification failed: {example_target}")

    report = {
        "patch_version": PATCH_VERSION,
        "package_root": str(package_root),
        "policy": "real cell expression required; no random profile fallback",
        "files": {relative: FILES[relative]["patched"] for relative in FILES},
        EXAMPLE_TARGET: EXAMPLE_SHA256,
    }
    manifest = package_root / MANIFEST
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="extracted package itself, or its parent directory",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(json.dumps(apply_patch(args.package_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
