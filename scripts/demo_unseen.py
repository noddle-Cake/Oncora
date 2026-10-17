#!/usr/bin/env python
"""Unseen-compound demo (spec 09, Req 1; AC-6, AC-7).

    uv run python scripts/demo_unseen.py

Runs the full pipeline for a compound held out of **both** models' training data, asserts that
holdout before trusting the output, and compares the predicted cell-line ranking against the
compound's real measured DSS3 values, which were set aside at table-build time.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from oncora.chem.featurize import featurize_mols
from oncora.chem.similarity import TrainingIndex
from oncora.chem.validate import parse_smiles
from oncora.config import DATA_PROCESSED, DEMO_HOLDOUT, MODELS, REPORTS
from oncora.data_prep.modeling import FEATURES

OUT_DIR = REPORTS / "demo"
HOLDOUT_SENS = DATA_PROCESSED / "demo_holdout_sensitivity.parquet"
HOLDOUT_TARGETS = DATA_PROCESSED / "demo_holdout_targets.parquet"


class NotUnseen(AssertionError):
    pass


def assert_unseen(inchi_key: str, run_dirs: dict[str, Path]) -> None:
    """Both models' training sets must exclude this compound's connectivity block."""
    conn = inchi_key[:14]
    std = pl.read_parquet(FEATURES, columns=["inchi_key", "std_inchi_key"])
    for name, run_dir in run_dirs.items():
        idx = TrainingIndex.load(run_dir / "train_index.npz")
        keys = pl.DataFrame({"inchi_key": list(idx.keys)}).join(std, on="inchi_key", how="left")
        blocks = set(keys["std_inchi_key"].drop_nulls().str.slice(0, 14).to_list())
        if conn in blocks or inchi_key in set(idx.keys):
            raise NotUnseen(f"{inchi_key} is in the {name} model's training set")


def ground_truth_comparison(inchi_key: str, scored: pl.DataFrame) -> dict | None:
    """Spearman ρ between predicted and measured sensitivity for this compound's held-out lines."""
    if not HOLDOUT_SENS.exists():
        return None
    truth = pl.read_parquet(HOLDOUT_SENS).filter(pl.col("inchi_key") == inchi_key)
    if truth.is_empty():
        return None
    joined = truth.join(scored, on="model_id", how="inner")
    if joined.height < 5:
        return None
    rho = spearmanr(joined["dss3"].to_numpy(), joined["predicted_percentile"].to_numpy())
    top20 = joined.sort("predicted_percentile", descending=True).head(20)
    overall_median = float(np.median(joined["dss3"].to_numpy()))
    return {
        "n_cell_lines_compared": joined.height,
        "spearman_rho": round(float(rho.statistic), 4),
        "p_value": float(rho.pvalue),
        "median_measured_dss3_all": round(overall_median, 4),
        "median_measured_dss3_top20_predicted": round(float(np.median(top20["dss3"].to_numpy())), 4),
    }


def known_targets(inchi_key: str) -> list[dict]:
    if not HOLDOUT_TARGETS.exists():
        return []
    t = pl.read_parquet(HOLDOUT_TARGETS).filter((pl.col("inchi_key") == inchi_key) & (pl.col("label") == 1))
    return (t.sort("pchembl_median", descending=True).head(10)
            .select("target_chembl_id", "target_name", "pchembl_median").to_dicts())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smiles", default=None, help="defaults to the configured demo compound")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    from oncora.serve.pipeline import OncoraPipeline

    pipe = OncoraPipeline()
    smiles = args.smiles
    if smiles is None:
        conn = next(iter(DEMO_HOLDOUT))
        row = pl.read_parquet(HOLDOUT_SENS).filter(pl.col("connectivity") == conn).head(1)
        if row.is_empty():
            raise SystemExit(f"No held-out sensitivity rows for {conn}; pass --smiles explicitly")
        smiles = row["canonical_smiles"][0]

    result = pipe.predict(smiles)
    if not result["valid"]:
        raise SystemExit(f"Demo SMILES is invalid: {result['error']}")
    inchi_key = result["inchi_key"]

    run_dirs = {"target": MODELS / "target" / result["model_versions"]["target"],
                "sensitivity": MODELS / "sensitivity" / result["model_versions"]["sensitivity"]}
    assert_unseen(inchi_key, run_dirs)

    scored = pipe.sensitivity.predict_all(featurize_mols([parse_smiles(smiles).mol], pipe.cfg)[0])
    comparison = ground_truth_comparison(inchi_key, scored)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "compound": DEMO_HOLDOUT.get(inchi_key[:14], "unknown"), "smiles": smiles,
               "unseen_verified": True, "prediction": result,
               "ground_truth_comparison": comparison, "known_targets_held_out": known_targets(inchi_key)}
    (args.out / f"{inchi_key}.json").write_text(json.dumps(payload, indent=1, default=str))

    print(f"Demo compound: {payload['compound']} ({inchi_key}) — verified unseen by both models")
    print("\nTop predicted targets:")
    for t in result["targets"][:5]:
        print(f"  {t['rank']:2d} {t['score']:.3f}  {t['gene_symbol'] or '':8s} {t['pref_name']}")
    if kt := payload["known_targets_held_out"]:
        print("\nHeld-out ChEMBL actives (ground truth, never trained on):")
        for t in kt[:5]:
            print(f"      pChEMBL {t['pchembl_median']:.2f}  {t['target_name']}")
    if comparison:
        print(f"\nCell-line ranking vs {comparison['n_cell_lines_compared']} measured lines: "
              f"Spearman ρ = {comparison['spearman_rho']:.3f} (p={comparison['p_value']:.2g})")
        print(f"  median measured DSS3: all lines {comparison['median_measured_dss3_all']:.4f}, "
              f"top-20 predicted {comparison['median_measured_dss3_top20_predicted']:.4f}")
    print(f"\nEvidence summary ({result['evidence_summary']['renderer']}):")
    for c in result["evidence_summary"]["claims"]:
        print(f"  [{c['section']}] {c['text']}")
    print(f"\nWrote {args.out / (inchi_key + '.json')}")


if __name__ == "__main__":
    main()
