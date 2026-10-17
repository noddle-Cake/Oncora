"""Load a trained sensitivity model and rank cell lines for a compound (spec 06, Req 3, 5.2; FR-9)."""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from oncora.chem.featurize import CompoundFeatureConfig
from oncora.chem.similarity import NearestNeighbor, TrainingIndex
from oncora.config import MODELS, SENS_TOP_N
from oncora.models.target.predict import latest_run


@dataclass
class CellLinePrediction:
    model_id: str
    cell_line_name: str | None
    predicted_percentile: float   # predicted DSS3 percentile under the training distribution
    predicted_dss3: float         # back-transformed through the training distribution
    p_sensitive: float            # P(line is in this compound's top 10%)
    rank: int


class SensitivityModel:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.card = json.loads((run_dir / "model_card.json").read_text())
        self.cfg = CompoundFeatureConfig.from_dict(self.card["features"]["compound"])
        self.reg = lgb.Booster(model_file=str(run_dir / "regressor.txt"))
        self.clf = lgb.Booster(model_file=str(run_dir / "classifier.txt"))
        self.sorted_train = np.load(run_dir / "rank_transform.npy")
        cells = pl.read_parquet(run_dir / "cellline_matrix.parquet")
        self.model_ids = cells["model_id"].to_list()
        self.cell_values = cells.drop("model_id").to_numpy().astype(np.float32)
        self.names: dict[str, str] = {}
        self.index = TrainingIndex.load(run_dir / "train_index.npz")

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def set_cell_line_names(self, names: dict[str, str]) -> None:
        self.names = names

    def predict_all(self, compound_vec: np.ndarray) -> pl.DataFrame:
        """Score one compound against every persisted cell line; sorted by predicted sensitivity."""
        x = np.hstack([np.broadcast_to(compound_vec.astype(np.float32), (len(self.model_ids), len(compound_vec))),
                       self.cell_values])
        pct = np.clip(self.reg.predict(x), 0.0, 1.0)
        prob = self.clf.predict(x)
        dss3 = np.quantile(self.sorted_train, pct)
        return (pl.DataFrame({"model_id": self.model_ids, "predicted_percentile": pct,
                              "predicted_dss3": dss3, "p_sensitive": prob})
                .sort(["predicted_percentile", "model_id"], descending=[True, False])
                .with_row_index("rank", offset=1))

    def predict(self, compound_vec: np.ndarray, top_n: int = SENS_TOP_N) -> list[CellLinePrediction]:
        return [CellLinePrediction(r["model_id"], self.names.get(r["model_id"]), float(r["predicted_percentile"]),
                                   float(r["predicted_dss3"]), float(r["p_sensitive"]), int(r["rank"]))
                for r in self.predict_all(compound_vec).head(top_n).iter_rows(named=True)]

    def nearest_training(self, packed_fp: np.ndarray) -> NearestNeighbor:
        return self.index.nearest(packed_fp)


def load_sensitivity_model(run_id: str | None = None) -> SensitivityModel:
    run_dir = MODELS / "sensitivity" / run_id if run_id else latest_run("sensitivity")
    if not (run_dir / "model_card.json").exists():
        raise FileNotFoundError(f"No sensitivity model at {run_dir}")
    return SensitivityModel(run_dir)
