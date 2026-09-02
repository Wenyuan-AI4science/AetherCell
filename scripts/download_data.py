#!/usr/bin/env python3
"""Download, resume, verify, and optionally extract the AetherCell Zenodo archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "18295255"
ZENODO_API = f"https://zenodo.org/api/records/{RECORD_ID}"
EXPECTED_FILE = "data4train.zip"
EXPECTED_SIZE = 7_343_624_829
EXPECTED_MD5 = "0302f6e032112f80af230315fc7469d9"


def record_file() -> dict:
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        record = json.load(response)
    files = {entry["key"]: entry for entry in record.get("files", [])}
    if EXPECTED_FILE not in files:
        raise RuntimeError(f"Zenodo record {RECORD_ID} does not contain {EXPECTED_FILE}")
    entry = files[EXPECTED_FILE]
    checksum = str(entry.get("checksum", "")).removeprefix("md5:")
    if int(entry.get("size", -1)) != EXPECTED_SIZE or checksum != EXPECTED_MD5:
        raise RuntimeError("Zenodo metadata differs from the version pinned by this repository")
    return entry


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, size: int) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise RuntimeError(f"partial file is larger than expected: {part}")
    free = shutil.disk_usage(destination.parent).free
    if free < size - offset:
        raise RuntimeError(f"insufficient free space: need {size - offset:,} additional bytes, have {free:,}")
    headers = {"User-Agent": "AetherCell-data-downloader/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"download request failed: HTTP {error.code}") from error
    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        response.close()
        raise RuntimeError("server ignored the Range request; remove the .part file to restart explicitly")
    mode = "ab" if offset else "wb"
    completed = offset
    with response, part.open(mode) as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
            completed += len(chunk)
            print(f"\r{completed:,}/{size:,} bytes ({completed / size:.1%})", end="", flush=True)
    print()
    if completed != size:
        raise RuntimeError(f"download is incomplete: expected {size}, found {completed}")
    os.replace(part, destination)


def safe_extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
        zipped.extractall(root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="validate the live record without downloading 7.3 GB")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entry = record_file()
    url = entry.get("links", {}).get("self")
    if not url:
        raise RuntimeError("Zenodo API did not return a content URL")
    metadata = {"record_id": RECORD_ID, "filename": EXPECTED_FILE, "size": EXPECTED_SIZE, "md5": EXPECTED_MD5, "url": url}
    (args.output_dir / "zenodo_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if args.metadata_only:
        return 0
    archive = args.output_dir / EXPECTED_FILE
    if not args.verify_only and not archive.exists():
        download(url, archive, EXPECTED_SIZE)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found for verification: {archive}")
    if archive.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"size mismatch for {archive}")
    actual = md5(archive)
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, found {actual}")
    print(f"verified {archive.name}: md5={actual}")
    if args.extract:
        safe_extract(archive, args.output_dir / "data4train")
        if not args.keep_archive:
            archive.unlink()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; the .part file can be resumed", file=sys.stderr)
        raise SystemExit(130)
