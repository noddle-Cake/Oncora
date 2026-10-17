"""Consensus L1000 signatures per compound from the Level 4 GCTX (spec 01 task 5.3; spec 07 Req 1.4).

    uv run python -m oncora.data_prep.signatures

**Evidence only (D-09).** These are *measured* expression responses. A new SMILES has none, so
nothing here may ever become a model feature (`assert_structure_only` enforces that). They feed
the evidence bundle as database-style annotations for compounds LINCS actually profiled.

The GCTX is an ~89 GB HDF5 matrix (1.8M profiles × 12,328 genes, float32 plate-normalized
z-scores) and is never loaded whole. One sequential pass reads row blocks of ~0.8 GB, keeps the
eligible profiles and the 978 landmark genes (measured directly; the other 11,350 are inferred),
and accumulates sums per (compound, cell line). Peak memory is ~3 GB.

Consensus, in two steps:
1. **cell-line signature:** mean z over a compound's eligible profiles in one cell line;
2. **compound signature:** mean of its cell-line signatures, so every cell line counts once and a
   compound profiled mostly in one line is not dominated by it.

Eligible profiles: QC pass, dose in µM ≥ `SIG_MIN_DOSE_UM`, and a structure in the compound bridge.

**Reproducibility:** the compound's cell lines are split into two fixed halves (alternating in
name order), each half is averaged, and the two half-signatures are correlated. A real drug
response repeats across cell lines; noise does not. It is null with a single cell line.
"""

import argparse
import hashlib
import time
from pathlib import Path

import duckdb
import h5py
import numpy as np
import polars as pl

from oncora.config import DATA_INTERIM, NULL_TOKENS, SIG_MIN_DOSE_UM, raw_path
from oncora.data_prep.provenance import assert_table, write_parquet

GCTX = raw_path("l1000_level4")
BRIDGE = DATA_INTERIM / "compound_bridge.parquet"
OUT = DATA_INTERIM / "l1000_consensus.parquet"
GENES_OUT = DATA_INTERIM / "l1000_genes.parquet"
MATRIX, COL_IDS, ROW_IDS = "0/DATA/0/matrix", "0/META/COL/id", "0/META/ROW/id"
BLOCK_ROWS = 16_384


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_ids(gctx: Path) -> tuple[list[str], list[str]]:
    with h5py.File(gctx, "r") as f:
        return [x.decode() for x in f[COL_IDS][:]], [x.decode() for x in f[ROW_IDS][:]]


def landmark_genes(row_ids: list[str], geneinfo: Path = raw_path("geneinfo")) -> pl.DataFrame:
    """Landmark genes in GCTX row order: row (matrix column index), gene_id, gene_symbol."""
    gi = pl.read_csv(geneinfo, separator="\t", infer_schema=False)
    lm = gi.filter(pl.col("feature_space") == "landmark").select("gene_id", "gene_symbol")
    rows = pl.DataFrame({"gene_id": row_ids}).with_row_index("row")
    return rows.join(lm, on="gene_id", how="inner", maintain_order="left")


def profile_metadata(col_ids: list[str], instinfo: Path = raw_path("instinfo"), bridge: Path = BRIDGE) -> pl.DataFrame:
    """One row per GCTX profile with its compound and condition (never loads instinfo into pandas)."""
    cols = ["sample_id", "pert_id", "cell_iname", "pert_dose", "pert_dose_unit", "pert_time", "qc_pass"]
    inst = duckdb.sql(f"""
        SELECT {', '.join(cols)} FROM read_csv('{instinfo}', delim='\t', header=true, all_varchar=true)
    """).pl().with_columns(pl.when(pl.col(c).is_in(list(NULL_TOKENS))).then(None).otherwise(pl.col(c)).alias(c)
                           for c in cols)
    b = pl.read_parquet(bridge, columns=["pert_id", "inchi_key", "connectivity"]).filter(pl.col("inchi_key").is_not_null())
    return (pl.DataFrame({"sample_id": col_ids}).with_row_index("col")
            .join(inst, on="sample_id", how="left", maintain_order="left")
            .join(b, on="pert_id", how="left", maintain_order="left")
            .with_columns(pl.col("pert_dose").cast(pl.Float64, strict=False).alias("dose_um")))


def eligible(meta: pl.DataFrame, min_dose: float = SIG_MIN_DOSE_UM) -> pl.DataFrame:
    return meta.filter((pl.col("qc_pass") == "1") & pl.col("inchi_key").is_not_null()
                       & (pl.col("pert_dose_unit") == "uM") & (pl.col("dose_um") >= min_dose))


def stream_group_sums(gctx: Path, cols: np.ndarray, group: np.ndarray, n_groups: int, gene_rows: np.ndarray,
                      block_rows: int = BLOCK_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Sum the selected genes of the selected profiles per group, in one sequential pass.

    cols: matrix row index of each eligible profile; group: its group id. Returns (sums, counts).
    """
    order = np.argsort(cols, kind="stable")
    cols, group = cols[order], group[order]
    sums = np.zeros((n_groups, len(gene_rows)), dtype=np.float64)
    counts = np.bincount(group, minlength=n_groups)
    with h5py.File(gctx, "r") as f:
        ds = f[MATRIX]
        n = ds.shape[0]
        for start in range(0, n, block_rows):
            lo, hi = np.searchsorted(cols, [start, min(start + block_rows, n)])
            if lo == hi:
                continue
            block = ds[start:min(start + block_rows, n)]
            x = block[cols[lo:hi] - start][:, gene_rows].astype(np.float64)
            g = group[lo:hi]
            o = np.argsort(g, kind="stable")
            g, x = g[o], x[o]
            starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            sums[g[starts]] += np.add.reduceat(x, starts, axis=0)
    return sums, counts


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def consensus(groups: pl.DataFrame, means: np.ndarray) -> pl.DataFrame:
    """groups: one row per (inchi_key, cell_iname) group in `means` order, with connectivity, pert_ids, n."""
    g = groups.with_row_index("gi").sort("inchi_key", "cell_iname")
    rows = []
    for (ik,), part in g.group_by(["inchi_key"], maintain_order=True):
        idx = part["gi"].to_numpy()
        sig = means[idx].mean(axis=0)
        rep = None
        if len(idx) >= 2:
            rep = _corr(means[idx[0::2]].mean(axis=0), means[idx[1::2]].mean(axis=0))
        rows.append({
            "inchi_key": ik, "connectivity": part["connectivity"][0],
            "pert_ids": sorted({p for ps in part["pert_ids"].to_list() for p in ps}),
            "n_profiles": int(part["n"].sum()), "n_cell_lines": len(idx),
            "reproducibility": None if rep is None or not np.isfinite(rep) else rep,
            "rms_z": float(np.sqrt(np.mean(sig ** 2))),
            "signature": sig.astype(np.float32),
        })
    n_genes = means.shape[1]
    return pl.DataFrame(rows, schema_overrides={"signature": pl.Array(pl.Float32, n_genes)})


def build(gctx: Path = GCTX, out: Path = OUT, genes_out: Path = GENES_OUT, *, instinfo: Path = raw_path("instinfo"),
          bridge: Path = BRIDGE, geneinfo: Path = raw_path("geneinfo"), min_dose: float = SIG_MIN_DOSE_UM,
          block_rows: int = BLOCK_ROWS, digest: bool = True) -> pl.DataFrame:
    if not gctx.exists():
        raise FileNotFoundError(f"{gctx} not found: the L1000 Level 4 GCTX is optional (D-09); "
                                "without it the evidence bundle simply omits measured signatures.")
    t0 = time.time()
    col_ids, row_ids = read_ids(gctx)
    genes = landmark_genes(row_ids, geneinfo)
    meta = profile_metadata(col_ids, instinfo, bridge)
    el = eligible(meta, min_dose)
    groups = (el.group_by("inchi_key", "cell_iname", maintain_order=True)
              .agg(pl.col("connectivity").first(), pl.col("pert_id").unique().sort().alias("pert_ids"), pl.len().alias("n"))
              .with_row_index("gid"))
    gid = el.join(groups.select("inchi_key", "cell_iname", "gid"), on=["inchi_key", "cell_iname"],
                  how="left", maintain_order="left")["gid"].to_numpy()
    sums, counts = stream_group_sums(gctx, el["col"].to_numpy(), gid, groups.height, genes["row"].to_numpy(), block_rows)
    means = sums / counts[:, None]
    table = consensus(groups.drop("gid"), means)
    assert_table(table, name="l1000_consensus", required_cols=["inchi_key", "connectivity", "signature"],
                 unique_key=["inchi_key"])
    if not np.isfinite(np.stack(table["signature"].to_numpy())).all():
        raise ValueError("l1000_consensus: non-finite values in a consensus signature")

    config = {"min_dose_um": min_dose, "qc_pass": True, "genes": "LINCS landmark (978)",
              "consensus": "mean over cell lines of per-cell-line mean z",
              "reproducibility": "Pearson r between half-signatures from alternating cell lines (name order)",
              "gctx_sha256": sha256_file(gctx) if digest else None,
              "profiles": {"total": meta.height, "eligible": el.height}}
    sources = [gctx, instinfo, bridge, geneinfo]
    write_parquet(table, out, sources=sources, created_by="oncora.data_prep.signatures",
                  filters=["qc_pass == 1", f"pert_dose_unit == uM and dose >= {min_dose}", "structure in compound bridge"],
                  config=config)
    write_parquet(genes.drop("row"), genes_out, sources=[gctx, geneinfo], created_by="oncora.data_prep.signatures",
                  config={"order": "signature vector order"})
    print(f"{table.height:,} compounds from {el.height:,} of {meta.height:,} profiles "
          f"in {time.time() - t0:.0f} s → {out}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-digest", action="store_true", help="skip hashing the 89 GB file (~1 min)")
    args = ap.parse_args()
    build(digest=not args.no_digest)


if __name__ == "__main__":
    main()
