"""Verify raw inputs against data/raw/SHA256SUMS before any stage runs.

Every derived table claims provenance from specific raw files. If a raw file is silently
replaced (a newer DepMap release, a re-download), that claim breaks, so we check first and
fail loudly rather than build on different data.
"""

import hashlib
import sys
from pathlib import Path

from oncora.config import DATA_RAW

MANIFEST = DATA_RAW / "SHA256SUMS"
_CHUNK = 8 * 1024 * 1024


class RawDataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest: Path) -> dict[str, str]:
    """Parse `shasum -a 256` output: '<hex>  <relative path>' per line."""
    entries = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        entries[rel.lstrip("*")] = digest
    return entries


def verify_raw(manifest: Path = MANIFEST) -> dict[str, str]:
    """Return {relative path: digest} if every listed file matches; raise otherwise."""
    if not manifest.exists():
        raise RawDataError(f"Checksum manifest not found: {manifest}")
    root = manifest.parent
    expected = read_manifest(manifest)
    problems = []
    for rel, digest in expected.items():
        path = root / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {rel}")
    if problems:
        raise RawDataError(
            "Raw data does not match SHA256SUMS:\n  " + "\n  ".join(problems)
        )
    return expected


def main() -> None:
    try:
        verified = verify_raw()
    except RawDataError as e:
        sys.exit(str(e))
    print(f"OK: {len(verified)} raw files match {MANIFEST.relative_to(DATA_RAW.parent.parent)}")


if __name__ == "__main__":
    main()
