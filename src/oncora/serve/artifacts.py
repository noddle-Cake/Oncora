"""Serving tables and the downloadable artifact bundle (spec 09; FR-23, NFR-2).

    uv run python -m oncora.serve.artifacts tables     # data/serving/*.parquet from interim/processed
    uv run python -m oncora.serve.artifacts package    # dist/oncora-artifacts-<runs>.tar.gz + .sha256
    uv run python -m oncora.serve.artifacts verify     # check an extracted bundle against its manifest

Inference reads only small, purpose-built tables in `data/serving/`, never the multi-GB build
tables. Each is cut to exactly what the evidence bundle uses, so the outputs are unchanged:

| table | from | kept |
|---|---|---|
| `compound_bridge` | interim/compound_bridge | the 5 annotation columns, rows with a structure |
| `chembl_actives` | interim/chembl_activities | actives only (pChEMBL ≥ 6), target gene symbol pre-joined |
| `ccle_expression` + `ccle_genes` | interim CCLE | the genes of the target model's targets |
| `l1000_*` | interim signature extract | copied when present (optional, D-09) |

The bundle packs the two model runs plus these tables (~400 MB compressed) with a manifest of
SHA256 digests. It is published as a GitHub Release asset and fetched with
`scripts/fetch_artifacts.sh`, so a fresh clone can serve predictions without the ~87 GB of raw data.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from oncora.config import (
    ACTIVE_PCHEMBL,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_SERVING,
    MODELS,
    ROOT,
)
from oncora.data_prep.provenance import _code_version, assert_table, write_parquet
from oncora.models.target.predict import latest_run

DIST = ROOT / "dist"
MANIFEST = "MANIFEST.json"
LICENSES = [
    "ChEMBL 37 (EMBL-EBI): CC BY-SA 3.0. Derived: chembl_actives, target model.",
    "LINCS L1000 / CLUE annotations and Level 4 signatures (Broad Institute): see clue.io terms. "
    "Derived: compound_bridge, l1000_*.",
    "DepMap / CCLE expression (Broad Institute): see depmap.org terms. Derived: ccle_*, sensitivity model.",
    "PharmacoDB DSS3 sensitivity: see pharmacodb.ca terms. Derived: sensitivity model.",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build_tables(out: Path = DATA_SERVING, interim: Path = DATA_INTERIM, processed: Path = DATA_PROCESSED) -> dict[str, int]:
    """Write the serving tables; returns rows per table."""
    out.mkdir(parents=True, exist_ok=True)
    rows = {}

    bridge_src = interim / "compound_bridge.parquet"
    bridge = (pl.read_parquet(bridge_src, columns=["inchi_key", "connectivity", "cmap_names", "moas", "targets"])
              .filter(pl.col("inchi_key").is_not_null()))
    assert_table(bridge, name="serving/compound_bridge", required_cols=bridge.columns)
    write_parquet(bridge, out / "compound_bridge.parquet", sources=[bridge_src], created_by=__name__)
    rows["compound_bridge"] = bridge.height

    tp_src, act_src = processed / "target_pairs.parquet", interim / "chembl_activities.parquet"
    genes = pl.read_parquet(tp_src, columns=["target_chembl_id", "target_gene_symbol"]).unique("target_chembl_id")
    actives = (pl.read_parquet(act_src, columns=["inchi_key", "connectivity", "target_chembl_id", "target_name",
                                                 "pchembl_median"])
               .filter(pl.col("pchembl_median") >= ACTIVE_PCHEMBL)
               .join(genes, on="target_chembl_id", how="left"))
    assert_table(actives, name="serving/chembl_actives", required_cols=actives.columns)
    write_parquet(actives, out / "chembl_actives.parquet", sources=[act_src, tp_src], created_by=__name__,
                  filters=[f"pchembl_median >= {ACTIVE_PCHEMBL}"])
    rows["chembl_actives"] = actives.height

    target_genes = set(genes["target_gene_symbol"].drop_nulls())
    cg_src, ccle_src = interim / "ccle_genes.parquet", interim / "ccle_expression.parquet"
    cg = pl.read_parquet(cg_src).filter(pl.col("gene_symbol").is_in(list(target_genes)))
    ccle = pl.read_parquet(ccle_src, columns=["model_id", *cg["gene_col"].to_list()])
    assert_table(ccle, name="serving/ccle_expression", required_cols=["model_id"], min_rows=100)
    write_parquet(cg, out / "ccle_genes.parquet", sources=[cg_src, tp_src], created_by=__name__,
                  filters=["genes of the target model's targets"])
    write_parquet(ccle, out / "ccle_expression.parquet", sources=[ccle_src, tp_src], created_by=__name__,
                  filters=["genes of the target model's targets"])
    rows["ccle_genes"] = cg.height

    for name in ("l1000_consensus.parquet", "l1000_genes.parquet"):
        src = interim / name
        if src.exists():
            shutil.copyfile(src, out / name)   # already provenance-tagged by oncora.data_prep.signatures
            rows[name.removesuffix(".parquet")] = pl.scan_parquet(src).select(pl.len()).collect().item()
    return rows


def bundle_files(target_run: Path, sens_run: Path, serving: Path = DATA_SERVING, root: Path = ROOT) -> list[Path]:
    files = sorted(Path(dirpath) / f for d in (target_run, sens_run, serving)
                   for dirpath, _, names in os.walk(d, followlinks=True) for f in names)
    if not any(p.name == "chembl_actives.parquet" for p in files):
        raise FileNotFoundError(f"{serving} has no serving tables: run `python -m {__name__} tables` first")
    return [p.relative_to(root) if p.is_absolute() else p for p in files]


def package(target_run: Path | None = None, sens_run: Path | None = None, out_dir: Path = DIST,
            root: Path = ROOT, serving: Path = DATA_SERVING) -> Path:
    target_run = target_run or latest_run("target")
    sens_run = sens_run or latest_run("sensitivity")
    rel = bundle_files(target_run, sens_run, serving, root=root)
    manifest = {
        "bundle": "oncora-artifacts",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "model_runs": {"target": target_run.name, "sensitivity": sens_run.name},
        "files": {str(p): {"bytes": (root / p).stat().st_size, "sha256": sha256(root / p)} for p in rel},
        "licenses": LICENSES,
        "disclaimer": "These are computational research predictions, not validated clinical conclusions.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"oncora-artifacts-t{target_run.name}-s{sens_run.name}.tar.gz"
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1))
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        tar.add(out_dir / MANIFEST, arcname=MANIFEST)
        for p in rel:
            tar.add(root / p, arcname=str(p))
    (out_dir / (archive.name + ".sha256")).write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


def verify(root: Path = ROOT) -> dict:
    """Check every manifest entry exists with the recorded digest; raise on any mismatch."""
    manifest = json.loads((root / MANIFEST).read_text())
    bad = [p for p, m in manifest["files"].items()
           if not (root / p).is_file() or sha256(root / p) != m["sha256"]]
    if bad:
        raise RuntimeError(f"{len(bad)} bundle files are missing or corrupted, e.g. {bad[:3]}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tables")
    pk = sub.add_parser("package")
    pk.add_argument("--target-run")
    pk.add_argument("--sensitivity-run")
    sub.add_parser("verify")
    args = ap.parse_args()
    if args.cmd == "tables":
        for name, n in build_tables().items():
            print(f"{name}: {n:,} rows")
        total = sum(p.stat().st_size for p in DATA_SERVING.glob("*.parquet"))
        print(f"data/serving: {total / 1e6:.0f} MB")
    elif args.cmd == "package":
        t = MODELS / "target" / args.target_run if args.target_run else None
        s = MODELS / "sensitivity" / args.sensitivity_run if args.sensitivity_run else None
        archive = package(t, s)
        print(f"{archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    else:
        m = verify()
        print(f"OK: {len(m['files'])} files match the manifest (target {m['model_runs']['target']}, "
              f"sensitivity {m['model_runs']['sensitivity']})")


if __name__ == "__main__":
    main()
