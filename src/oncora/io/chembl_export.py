"""One-time ChEMBL 37 restore → Parquet export (spec 02, Req 1; DR-5).

Technique: relational → columnar ETL. ChEMBL ships as a PostgreSQL dump, but Oncora needs only
8 tables and only reads them. Restoring those tables (data only: no indexes or foreign keys),
copying them to Parquet, and dropping the database leaves a Postgres-free pipeline that DuckDB
can query directly.

Prerequisites (one-time, see README):
    brew install postgresql@16
    pg_ctl -D /opt/homebrew/var/postgresql@16 -l data/interim/chembl/postgres.log start

Run:
    uv run python -m oncora.io.chembl_export all        # extract → restore → export → spot-check
    pg_ctl -D /opt/homebrew/var/postgresql@16 stop

Connection settings come from the standard libpq environment variables (PGHOST, PGPORT, PGUSER,
PGPASSWORD); with none set, the local Unix socket and current user are used.
"""

import argparse
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import duckdb
import polars as pl

from oncora.config import DATA_INTERIM, raw_path
from oncora.data_prep.provenance import source_entry

DB_NAME = "chembl_37"
DUMP_MEMBER = "chembl_37/chembl_37_postgresql/chembl_37_postgresql.dmp"
WORK_DIR = DATA_INTERIM / "chembl"
DUMP_PATH = WORK_DIR / "chembl_37_postgresql.dmp"

# Tables Oncora reads, with columns to leave out (large structure blobs nothing downstream uses).
TABLES: dict[str, tuple[str, ...]] = {
    "activities": (),
    "assays": (),
    "target_dictionary": (),
    "target_components": (),
    "component_sequences": ("sequence",),
    "component_synonyms": (),
    "molecule_dictionary": (),
    "compound_structures": ("molfile",),
}

_PG_BIN_CANDIDATES = ("/opt/homebrew/opt/postgresql@16/bin", "/usr/local/opt/postgresql@16/bin")


class ChemblExportError(RuntimeError):
    pass


def _pg_tool(name: str) -> str:
    found = shutil.which(name) or next(
        (str(Path(d) / name) for d in _PG_BIN_CANDIDATES if (Path(d) / name).exists()), None
    )
    if not found:
        raise ChemblExportError(
            f"`{name}` not found. Install PostgreSQL once with `brew install postgresql@16`."
        )
    return found


def extract_dump(tar_path: Path | None = None, out: Path = DUMP_PATH) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path or raw_path("chembl_tar")) as tar:
        member = tar.getmember(DUMP_MEMBER)
        with tar.extractfile(member) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out


_TOC_LINE = re.compile(r"^\d+; \d+ \d+ (TABLE|TABLE DATA) public (\w+) ")


def restore_list(dump: Path, tables=TABLES) -> Path:
    """Write a pg_restore -L list holding only the table definitions and data we need.
    Skipping indexes, constraints and every other table keeps the restore small and fast."""
    toc = subprocess.run(
        [_pg_tool("pg_restore"), "-l", str(dump)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    keep = [line for line in toc if (m := _TOC_LINE.match(line)) and m.group(2) in tables]
    found = {_TOC_LINE.match(line).group(2) for line in keep}
    if missing := set(tables) - found:
        raise ChemblExportError(f"Tables not in dump: {sorted(missing)}")
    path = dump.with_suffix(".list")
    path.write_text("\n".join(keep) + "\n")
    return path


def restore(dump: Path = DUMP_PATH, db: str = DB_NAME, jobs: int = 8) -> None:
    subprocess.run([_pg_tool("dropdb"), "--if-exists", db], check=True)
    subprocess.run([_pg_tool("createdb"), db], check=True)
    subprocess.run(
        [_pg_tool("pg_restore"), "--no-owner", "--no-privileges", f"--jobs={jobs}",
         "-L", str(restore_list(dump)), "-d", db, str(dump)],
        check=True,
    )


def _connect(db: str = DB_NAME) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("INSTALL postgres; LOAD postgres")
    parts = [f"dbname={db}"] + [
        f"{k}={os.environ[v]}"
        for k, v in [("host", "PGHOST"), ("port", "PGPORT"), ("user", "PGUSER"), ("password", "PGPASSWORD")]
        if v in os.environ
    ]
    con.sql(f"ATTACH '{' '.join(parts)}' AS pg (TYPE postgres, READ_ONLY)")
    return con


def export_tables(out_dir: Path = WORK_DIR, db: str = DB_NAME, tables=TABLES) -> dict[str, int]:
    con = _connect(db)
    counts = {}
    for table, drop in tables.items():
        cols = [r[0] for r in con.sql(f"DESCRIBE pg.public.{table}").fetchall() if r[0] not in drop]
        out = out_dir / f"{table}.parquet"
        select = ", ".join(f'"{c}"' for c in cols)
        con.sql(
            f"COPY (SELECT {select} FROM pg.public.{table}) TO '{out}' "
            "(FORMAT parquet, COMPRESSION zstd)"
        )
        counts[table] = con.sql(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
        if counts[table] == 0:
            raise ChemblExportError(f"Exported {table} is empty")
    manifest = pl.DataFrame({"table": list(counts), "rows": list(counts.values())})
    manifest = manifest.with_columns(pl.lit(source_entry(raw_path("chembl_tar"))["sha256"]).alias("source_sha256"))
    manifest.write_csv(out_dir / "export_manifest.csv")
    return counts


def spot_check(chembl_id: str = "CHEMBL941", out_dir: Path = WORK_DIR) -> pl.DataFrame:
    """Imatinib (CHEMBL941) should show potent activity on ABL1, KIT and PDGFR family targets."""
    p = {t: str(out_dir / f"{t}.parquet") for t in TABLES}
    return duckdb.sql(
        f"""
        SELECT td.chembl_id AS target, td.pref_name, COUNT(*) AS n,
               ROUND(MEDIAN(a.pchembl_value), 2) AS median_pchembl
        FROM '{p["molecule_dictionary"]}' md
        JOIN '{p["activities"]}' a USING (molregno)
        JOIN '{p["assays"]}' s USING (assay_id)
        JOIN '{p["target_dictionary"]}' td ON td.tid = s.tid
        WHERE md.chembl_id = '{chembl_id}' AND a.pchembl_value IS NOT NULL
              AND td.target_type = 'SINGLE PROTEIN' AND td.organism = 'Homo sapiens'
        GROUP BY ALL HAVING COUNT(*) >= 3
        ORDER BY median_pchembl DESC LIMIT 10
        """
    ).pl()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["extract", "restore", "export", "spot-check", "cleanup", "all"])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if args.step in ("extract", "all"):
        print(f"Extracted {extract_dump()}")
    if args.step in ("restore", "all"):
        restore(jobs=args.jobs)
        print(f"Restored {len(TABLES)} tables into {DB_NAME}")
    if args.step in ("export", "all"):
        for t, n in export_tables().items():
            print(f"  {t:22s} {n:>12,}")
    if args.step in ("spot-check", "all"):
        print(spot_check())
    if args.step in ("cleanup", "all"):
        DUMP_PATH.unlink(missing_ok=True)
        DUMP_PATH.with_suffix(".list").unlink(missing_ok=True)
        subprocess.run([_pg_tool("dropdb"), "--if-exists", DB_NAME], check=True)
        print("Removed extracted dump and dropped the database; Parquet files kept")


if __name__ == "__main__":
    main()
