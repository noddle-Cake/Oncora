"""Provenance-tagged Parquet I/O and fail-loudly table assertions (NFR-6, NFR-7).

Each derived table records, in its Parquet footer, which raw files (by SHA256) and which
filters produced it. Months later you can still answer "where did this number come from?"
without rerunning anything.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from oncora.config import DATA_RAW, ROOT
from oncora.io.checksums import MANIFEST, read_manifest

_KEY = b"oncora_provenance"


class DataIntegrityError(RuntimeError):
    pass


def _code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def source_entry(path: Path) -> dict[str, str | None]:
    """Describe a source file. Raw files take their digest from SHA256SUMS (already verified);
    derived files point at their own provenance instead."""
    path = Path(path)
    try:
        rel = str(path.resolve().relative_to(DATA_RAW.resolve()))
        return {"path": f"data/raw/{rel}", "sha256": read_manifest(MANIFEST).get(rel)}
    except ValueError:
        return {"path": str(path.resolve().relative_to(ROOT.resolve())), "sha256": None}


def write_parquet(
    df: pl.DataFrame,
    path: Path,
    *,
    sources: list[Path],
    created_by: str,
    filters: list[str] | None = None,
    config: dict | None = None,
) -> Path:
    meta = {
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "sources": [source_entry(s) for s in sources],
        "filters": filters or [],
        "config": config or {},
        "shape": list(df.shape),
    }
    table = df.to_arrow()
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), _KEY: json.dumps(meta).encode()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_provenance(path: Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    if _KEY not in md:
        raise DataIntegrityError(f"{path} has no Oncora provenance metadata")
    return json.loads(md[_KEY])


def assert_table(
    df: pl.DataFrame,
    *,
    name: str,
    required_cols: list[str],
    min_rows: int = 1,
    max_null_rate: dict[str, float] | None = None,
    unique_key: list[str] | None = None,
) -> None:
    """Raise DataIntegrityError instead of letting a broken table flow downstream."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataIntegrityError(f"{name}: missing columns {missing}")
    if df.height < min_rows:
        raise DataIntegrityError(f"{name}: {df.height} rows, expected at least {min_rows}")
    for col, bound in (max_null_rate or {}).items():
        rate = df[col].null_count() / df.height
        if rate > bound:
            raise DataIntegrityError(f"{name}: column {col!r} null rate {rate:.2%} > {bound:.2%}")
    if unique_key:
        n_dup = df.height - df.select(unique_key).unique().height
        if n_dup:
            raise DataIntegrityError(f"{name}: {n_dup} duplicate rows on key {unique_key}")
