"""Measured L1000 signatures as evidence (spec 07, Req 1.4; FR-16; D-09).

For a query that LINCS actually profiled (matched by exact InChIKey, then by connectivity block,
like the other annotations), the bundle gets:

- the strongest up- and down-regulated landmark genes of its consensus signature;
- **"responds like" neighbors:** other profiled compounds whose measured signatures correlate
  most with it, with their LINCS names, MoAs and targets. Neighbors are drawn only from
  compounds with a reproducible response, never share the query's skeleton (stereoisomers and
  salts would trivially match), and are only computed when the query's own response is
  reproducible. Otherwise the correlations would rank noise.

A similar response suggests a shared mechanism; it does not prove one. For a brand-new SMILES
there is no measured signature and this evidence is simply absent. It never feeds a model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from oncora.config import (
    DATA_INTERIM,
    SIG_MIN_REPRODUCIBILITY,
    SIG_NEIGHBORS,
    SIG_TOP_GENES,
)
from oncora.interpret.bundle import (
    GeneResponse,
    MeasuredSignature,
    SignatureNeighbor,
    r3,
)

CONSENSUS = DATA_INTERIM / "l1000_consensus.parquet"
GENES = DATA_INTERIM / "l1000_genes.parquet"
BRIDGE = DATA_INTERIM / "compound_bridge.parquet"


def _unit_rows(x: np.ndarray) -> np.ndarray:
    """Center and L2-normalize rows, so a dot product is a Pearson correlation."""
    x = x - x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def reproducible(rep: float | None, threshold: float = SIG_MIN_REPRODUCIBILITY) -> bool:
    return rep is not None and np.isfinite(rep) and rep >= threshold


@dataclass
class SignatureIndex:
    table: pl.DataFrame        # inchi_key, connectivity, n_profiles, n_cell_lines, reproducibility + names/moa/targets
    signatures: np.ndarray     # (n, genes) consensus z
    genes: list[str]

    def __post_init__(self):
        self._unit = _unit_rows(self.signatures.astype(np.float64))
        rep = self.table["reproducibility"].to_list()
        self._ref = np.array([reproducible(r) for r in rep])
        self._conn = self.table["connectivity"].to_numpy()

    @classmethod
    def load(cls, consensus: Path = CONSENSUS, genes: Path = GENES, bridge: Path = BRIDGE) -> "SignatureIndex":
        t = pl.read_parquet(consensus)
        names = (pl.read_parquet(bridge, columns=["inchi_key", "cmap_names", "moas", "targets"])
                 .filter(pl.col("inchi_key").is_not_null())
                 .group_by("inchi_key")
                 .agg(*(pl.col(c).list.explode(keep_nulls=False, empty_as_null=False).unique().sort().alias(a)
                        for c, a in (("cmap_names", "names"), ("moas", "moa"), ("targets", "targets")))))
        t = t.join(names, on="inchi_key", how="left", maintain_order="left").with_columns(
            pl.col(c).fill_null([]) for c in ("names", "moa", "targets"))
        return cls(t.drop("signature"), np.stack(t["signature"].to_numpy()), pl.read_parquet(genes)["gene_symbol"].to_list())

    @classmethod
    def load_optional(cls) -> "SignatureIndex | None":
        """None when signatures were never extracted (the GCTX is optional)."""
        return cls.load() if CONSENSUS.exists() and GENES.exists() else None

    def _row(self, inchi_key: str) -> tuple[int, str] | None:
        exact = np.flatnonzero(self.table["inchi_key"].to_numpy() == inchi_key)
        if len(exact):
            return int(exact[0]), "exact"
        same = np.flatnonzero(self._conn == inchi_key[:14])
        if len(same):   # the best-profiled stereoisomer / salt form of the same skeleton
            return int(same[np.argmax(self.table["n_profiles"].to_numpy()[same])]), "connectivity"
        return None

    def neighbors(self, i: int, k: int = SIG_NEIGHBORS) -> list[SignatureNeighbor]:
        r = self._unit @ self._unit[i]
        ok = self._ref & (self._conn != self._conn[i])
        idx = np.flatnonzero(ok)
        top = idx[np.lexsort((self.table["inchi_key"].to_numpy()[idx], -r[idx]))][:k]   # ties broken by key
        rows = self.table[top.tolist()]
        return [SignatureNeighbor(inchi_key=row["inchi_key"], names=row["names"][:3], moa=row["moa"],
                                  targets=row["targets"], correlation=r3(float(r[j])))
                for j, row in zip(top, rows.iter_rows(named=True))]

    def lookup(self, inchi_key: str, top_genes: int = SIG_TOP_GENES) -> MeasuredSignature | None:
        hit = self._row(inchi_key)
        if hit is None:
            return None
        i, level = hit
        row = self.table.row(i, named=True)
        z = self.signatures[i]
        order = np.argsort(-z, kind="stable")
        up = [GeneResponse(gene_symbol=self.genes[j], z=r3(float(z[j]))) for j in order[:top_genes] if z[j] > 0]
        down = [GeneResponse(gene_symbol=self.genes[j], z=r3(float(z[j]))) for j in order[::-1][:top_genes] if z[j] < 0]
        rep = row["reproducibility"]
        is_rep = reproducible(rep)
        return MeasuredSignature(match_level=level, n_profiles=row["n_profiles"], n_cell_lines=row["n_cell_lines"],
                                 reproducibility=r3(rep), reproducible=is_rep, up=up, down=down,
                                 neighbors=self.neighbors(i) if is_rep else [])


def signature_target_overlap(predicted_genes: list[str | None], sig: MeasuredSignature | None) -> list[str]:
    if sig is None:
        return []
    known = {t for n in sig.neighbors for t in n.targets}
    return sorted({g for g in predicted_genes if g and g in known})
