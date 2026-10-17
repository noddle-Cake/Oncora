"""Load a trained target model and rank targets for a compound (spec 05, Req 2 and 4.4; FR-23)."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from oncora.chem.featurize import CompoundFeatureConfig
from oncora.chem.similarity import NearestNeighbor, TrainingIndex
from oncora.config import MODELS, TARGET_TOP_N
from oncora.models.target.attribution import Attribution, explain
from oncora.models.target.calibration import apply_calibrator, ranking_score


@dataclass
class TargetPrediction:
    target_chembl_id: str
    pref_name: str | None
    gene_symbol: str | None
    uniprot: str | None
    score: float          # calibrated probability of activity (pChEMBL ≥ 6)
    rank: int


class TargetModel:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.card = json.loads((run_dir / "model_card.json").read_text())
        self.cfg = CompoundFeatureConfig.from_dict(self.card["features"]["compound"])
        meta = pl.read_parquet(run_dir / "targets.parquet").filter(pl.col("skipped").is_null()).sort("target_chembl_id")
        self.targets = meta["target_chembl_id"].to_list()
        self.meta = {r["target_chembl_id"]: r for r in meta.iter_rows(named=True)}
        cals = json.loads((run_dir / "calibrators.json").read_text())
        self.calibrators = [cals.get(t, {"method": "identity"}) for t in self.targets]
        with ThreadPoolExecutor(8) as ex:
            self.boosters = list(ex.map(lambda t: lgb.Booster(model_file=str(run_dir / "lgbm" / f"{t}.txt")), self.targets))
        self.index = TrainingIndex.load(run_dir / "train_index.npz")

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def raw_matrix(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features).astype(np.float32)
        with ThreadPoolExecutor(8) as ex:
            return np.column_stack(list(ex.map(lambda b: b.predict(features, num_threads=1), self.boosters)))

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calibrated probabilities, shape (n_compounds, n_targets)."""
        raw = self.raw_matrix(features)
        return np.column_stack([apply_calibrator(c, raw[:, j]) for j, c in enumerate(self.calibrators)])

    def predict(self, features: np.ndarray, top_n: int = TARGET_TOP_N) -> list[TargetPrediction]:
        raw = self.raw_matrix(features)[0]
        scores = np.array([apply_calibrator(c, raw[j:j + 1])[0] for j, c in enumerate(self.calibrators)])
        order = np.argsort(-ranking_score(scores, raw), kind="stable")[:top_n]
        out = []
        for rank, i in enumerate(order, start=1):
            m = self.meta[self.targets[i]]
            out.append(TargetPrediction(self.targets[i], m.get("target_name"), m.get("target_gene_symbol"),
                                        m.get("uniprot_accession"), float(scores[i]), rank))
        return out

    def nearest_training(self, packed_fp: np.ndarray) -> NearestNeighbor:
        return self.index.nearest(packed_fp)

    def explain(self, mol, target_chembl_id: str, k: int = 5) -> list[Attribution]:
        """Top substructures pushing this target's score up for `mol` (a standardized RDKit Mol)."""
        return explain(self.boosters[self.targets.index(target_chembl_id)], mol, self.cfg, k)


def latest_run(kind: str) -> Path:
    runs = sorted(p for p in (MODELS / kind).glob("*") if (p / "model_card.json").exists())
    if not runs:
        raise FileNotFoundError(f"No trained {kind} model under {MODELS / kind}; run the trainer first")
    return runs[-1]


def load_target_model(run_id: str | None = None) -> TargetModel:
    run_dir = MODELS / "target" / run_id if run_id else latest_run("target")
    if not (run_dir / "model_card.json").exists():
        available = [p.name for p in (MODELS / "target").glob("*")]
        raise FileNotFoundError(f"No target model at {run_dir}; available runs: {available}")
    return TargetModel(run_dir)
