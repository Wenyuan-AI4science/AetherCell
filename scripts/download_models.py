#!/usr/bin/env python3
"""Download pinned models and apply the reviewed deterministic-input API patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

REPO_ID = "liwenyuan99/AetherCell"
PACKAGE = "aethercell-drug-discovery-v1.0.0.zip"
PACKAGE_SIZE = 4_027_434_886
PACKAGE_SHA256 = "6e4c9407ce2c6235442a8341c26e5c8beb36f4b59d30a1a976d1892d9b08706f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in model archive: {info.filename}")
        archive.extractall(root)


def verify_live_metadata() -> dict:
    url = f"https://huggingface.co/api/models/{REPO_ID}/tree/main?recursive=true&expand=false"
    request = urllib.request.Request(url, headers={"User-Agent": "AetherCell-model-downloader/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    matches = [entry for entry in entries if entry.get("path") == PACKAGE]
    if len(matches) != 1:
        raise RuntimeError(f"Hugging Face repository does not contain exactly one {PACKAGE}")
    entry = matches[0]
    if int(entry.get("size", -1)) != PACKAGE_SIZE or entry.get("lfs", {}).get("oid") != PACKAGE_SHA256:
        raise RuntimeError("live model-package metadata differs from the pinned release")
    return {"repo_id": REPO_ID, "filename": PACKAGE, "size": PACKAGE_SIZE, "sha256": PACKAGE_SHA256}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="validate the live Hugging Face entry without downloading 4.0 GB")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = verify_live_metadata()
    print(json.dumps(metadata, indent=2))
    if args.metadata_only:
        return 0
    package = args.output_dir / PACKAGE
    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise SystemExit("install huggingface-hub or use `pip install -e .[model]`") from error
        downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=PACKAGE, local_dir=args.output_dir))
        if downloaded != package:
            package = downloaded
    if not package.is_file():
        raise FileNotFoundError(package)
    if package.stat().st_size != PACKAGE_SIZE:
        raise RuntimeError(f"model package size mismatch: expected {PACKAGE_SIZE}, found {package.stat().st_size}")
    actual = sha256(package)
    if actual != PACKAGE_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {PACKAGE_SHA256}, found {actual}")
    print(f"verified {package.name}: sha256={actual}")
    if args.extract:
        safe_extract(package, args.output_dir)
        from apply_agent_ready_patch import apply_patch

        report = apply_patch(args.output_dir)
        print(f"applied Agent-Ready patch: {report['patch_version']}")
        if not args.keep_archive:
            package.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
