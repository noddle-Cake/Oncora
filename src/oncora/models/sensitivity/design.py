"""Design matrix for the pairwise (compound × cell line) sensitivity model (spec 06, Req 1).

A row is [compound features ‖ cell-line features]. Both halves are built per entity once (~940
compounds, ~1k lines) and gathered by index, so the ~470k-row matrix is assembled with two
fancy-index operations instead of a joined DataFrame.

`assert_structure_only` enforces FR-10: every feature must come from the compound structure or the
precomputed cell-line expression config. Nothing derived from a measured perturbation
signature may enter, because a new SMILES would not have one at inference.
"""

import numpy as np
import polars as pl

from oncora.chem.featurize import CompoundFeatureConfig
from oncora.features.cellline import CellLineFeatureConfig


class StructureOnlyViolation(AssertionError):
    pass


def assert_structure_only(feature_names: list[str], compound_cfg: CompoundFeatureConfig,
                          cell_cfg: CellLineFeatureConfig | None) -> None:
    allowed = set(compound_cfg.feature_names) | (set(cell_cfg.feature_names) if cell_cfg else set())
    if extra := [f for f in feature_names if f not in allowed]:
        raise StructureOnlyViolation(f"features outside the structure/expression configs: {extra[:5]}")


class EntityMatrix:
    """Feature rows for a set of entity keys, gathered by key."""

    def __init__(self, keys: list[str], values: np.ndarray, names: list[str]):
        self.keys, self.values, self.names = list(keys), values.astype(np.float32), names
        self._idx = pl.DataFrame({"k": self.keys}).with_row_index("i")

    def index(self, keys) -> np.ndarray:
        idx = pl.DataFrame({"k": keys}).join(self._idx, on="k", how="left", maintain_order="left")["i"]
        if idx.null_count():
            raise KeyError(f"{idx.null_count()} keys missing from the entity matrix")
        return idx.to_numpy()


def build_design_matrix(rows: pl.DataFrame, compounds: EntityMatrix, cells: EntityMatrix | None) -> np.ndarray:
    xc = compounds.values[compounds.index(rows["inchi_key"])]
    if cells is None:
        return xc
    return np.hstack([xc, cells.values[cells.index(rows["model_id"])]])
