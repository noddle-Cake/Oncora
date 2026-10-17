"""Cell-line expression features (spec 03, Req 5; DR-7, decision D-06).

MVP strategy: the 978 LINCS landmark genes, z-scored with means and stds from the *training*
cell lines only. The gene list is fixed in advance, so no gene-selection step can see test
lines. The scaling statistics are the only fitted part, so they are fit on training lines and
persisted with the model (inference parity).

The "variance" strategy (top-k most variable genes, also fit on training lines) exists for the
STRETCH comparison in spec 06.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import polars as pl

from oncora.config import DATA_INTERIM, raw_path

CCLE_PARQUET = DATA_INTERIM / "ccle_expression.parquet"
CCLE_GENES = DATA_INTERIM / "ccle_genes.parquet"

Strategy = Literal["landmark", "variance"]


@dataclass(frozen=True)
class CellLineFeatureConfig:
    strategy: Strategy
    genes: tuple[str, ...]          # CCLE column names, e.g. "EGFR (1956)": the persisted truth
    means: tuple[float, ...]
    stds: tuple[float, ...]

    @property
    def feature_names(self) -> list[str]:
        return [f"expr_{g}" for g in self.genes]

    def to_dict(self) -> dict:
        return asdict(self) | {k: list(getattr(self, k)) for k in ("genes", "means", "stds")}

    @classmethod
    def from_dict(cls, d: dict) -> "CellLineFeatureConfig":
        return cls(d["strategy"], tuple(d["genes"]), tuple(d["means"]), tuple(d["stds"]))


@lru_cache(maxsize=1)
def landmark_gene_columns() -> tuple[str, ...]:
    """CCLE gene columns whose Entrez id is a LINCS landmark gene, in CCLE column order."""
    gi = pl.read_csv(raw_path("geneinfo"), separator="\t", quote_char=None, infer_schema=False)
    landmark = set(gi.filter(pl.col("feature_space") == "landmark")["gene_id"].cast(pl.Int64))
    genes = pl.read_parquet(CCLE_GENES)
    return tuple(genes.filter(pl.col("entrez_id").is_in(list(landmark)))["gene_col"])


def _expression(model_ids: list[str], genes: list[str] | tuple[str, ...]) -> np.ndarray:
    df = pl.read_parquet(CCLE_PARQUET, columns=["model_id", *genes])
    idx = pl.DataFrame({"model_id": model_ids}).with_row_index("_pos").join(df, on="model_id", how="left")
    if (missing := idx.filter(pl.col(genes[0]).is_null())["model_id"].to_list()):
        raise KeyError(f"{len(missing)} cell lines have no CCLE expression, e.g. {missing[:3]}")
    return idx.sort("_pos").select(genes).to_numpy().astype(np.float64)


def fit_cellline_features(train_model_ids: list[str], strategy: Strategy = "landmark",
                          k: int = 2000) -> CellLineFeatureConfig:
    if strategy == "landmark":
        genes = landmark_gene_columns()
    elif strategy == "variance":
        all_genes = tuple(pl.read_parquet(CCLE_GENES)["gene_col"])
        sd = _expression(train_model_ids, all_genes).std(axis=0)
        order = np.argsort(-sd, kind="stable")
        genes = tuple(all_genes[i] for i in order[:k] if sd[i] > 0)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    x = _expression(train_model_ids, genes)
    means, stds = x.mean(axis=0), x.std(axis=0)
    stds = np.where(stds > 0, stds, 1.0)  # a gene constant on training lines contributes zeros
    return CellLineFeatureConfig(strategy, tuple(genes), tuple(means.tolist()), tuple(stds.tolist()))


def transform(model_ids: list[str], cfg: CellLineFeatureConfig) -> np.ndarray:
    """float32 (len(model_ids), n_genes) z-scores in the order of `model_ids`."""
    x = _expression(model_ids, cfg.genes)
    return ((x - np.asarray(cfg.means)) / np.asarray(cfg.stds)).astype(np.float32)
