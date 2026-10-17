"""Compound feature table: featurize every modeling compound once, cache by InChIKey (spec 03, Req 3.5).

Rows are keyed by the InChIKey used in the modeling tables (the source's key), so joins stay
exact. The table also records the standardized SMILES and RDKit InChIKey actually featurized,
so differences between source keys and standardized structures stay visible.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from oncora.chem.featurize import CompoundFeatureConfig, featurize_smiles_batch, unpack_fingerprints
from oncora.data_prep.provenance import read_provenance, write_parquet

BATCH = 5000


def _featurize_parallel(smiles: list[str], cfg: CompoundFeatureConfig, n_jobs: int) -> dict[str, list]:
    batches = [smiles[i: i + BATCH] for i in range(0, len(smiles), BATCH)]
    merged: dict[str, list] = {k: [] for k in ("ok", "std_smiles", "std_inchi_key", "scaffold", "fp", "desc", "n_nonfinite")}
    if n_jobs <= 1 or len(batches) == 1:
        results = [featurize_smiles_batch(b, cfg.to_dict()) for b in batches]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(featurize_smiles_batch, batches, [cfg.to_dict()] * len(batches)))
    for r in results:
        for k, v in r.items():
            merged.setdefault(k, []).extend(v)
    return merged


def _to_frame(keys: list[str], res: dict[str, list], cfg: CompoundFeatureConfig) -> pl.DataFrame:
    zeros = np.zeros(len(cfg.descriptors), dtype=np.float32)
    desc = np.vstack([d if d is not None else zeros for d in res["desc"]]) if keys else np.zeros((0, len(cfg.descriptors)))
    return pl.DataFrame({
        "inchi_key": keys,
        "ok": res["ok"],
        "std_smiles": res["std_smiles"],
        "std_inchi_key": res["std_inchi_key"],
        "scaffold": res["scaffold"],
        "fp": pl.Series(res["fp"], dtype=pl.Binary),
        "n_nonfinite": pl.Series(res["n_nonfinite"], dtype=pl.Int16),
        **{f"desc_{name}": pl.Series(desc[:, j], dtype=pl.Float32) for j, name in enumerate(cfg.descriptors)},
    })


def featurize_table(
    compounds: pl.DataFrame,
    cfg: CompoundFeatureConfig,
    cache_path: Path,
    *,
    n_jobs: int | None = None,
    sources: list[Path] | None = None,
) -> pl.DataFrame:
    """compounds: columns inchi_key, canonical_smiles (one row per key). Featurizes keys missing
    from the cache, or everything if the cached config differs, and rewrites the cache."""
    compounds = compounds.unique("inchi_key").sort("inchi_key")
    cached = None
    if cache_path.exists() and read_provenance(cache_path).get("config", {}).get("features") == cfg.to_dict():
        cached = pl.read_parquet(cache_path)
    todo = compounds if cached is None else compounds.join(cached.select("inchi_key"), on="inchi_key", how="anti")

    if cached is not None and todo.is_empty():
        return cached.join(compounds.select("inchi_key"), on="inchi_key", how="semi")

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    res = _featurize_parallel(todo["canonical_smiles"].to_list(), cfg, n_jobs)
    fresh = _to_frame(todo["inchi_key"].to_list(), res, cfg)
    table = fresh if cached is None else pl.concat([cached, fresh]).sort("inchi_key")
    write_parquet(table, cache_path, sources=sources or [], created_by=__name__,
                  config={"features": cfg.to_dict()},
                  filters=["standardize (cleanup, largest fragment, uncharge)", "ECFP + descriptors"])
    return table.join(compounds.select("inchi_key"), on="inchi_key", how="semi")


def feature_matrix(table: pl.DataFrame, keys: list[str], cfg: CompoundFeatureConfig) -> np.ndarray:
    """Dense float32 (len(keys), n_features) in the order of `keys`; raises on unknown or failed keys."""
    sub = pl.DataFrame({"inchi_key": keys}).with_row_index("_pos").join(table, on="inchi_key", how="left")
    if (bad := sub.filter(pl.col("ok").is_null() | ~pl.col("ok"))).height:
        raise KeyError(f"{bad.height} compounds lack features, e.g. {bad['inchi_key'].head(3).to_list()}")
    sub = sub.sort("_pos")
    fp = unpack_fingerprints(sub["fp"].to_list(), cfg.n_bits).astype(np.float32)
    desc = sub.select([f"desc_{d}" for d in cfg.descriptors]).to_numpy().astype(np.float32)
    return np.hstack([fp, desc])
