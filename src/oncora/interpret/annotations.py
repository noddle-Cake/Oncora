"""Database annotations and cross-evidence for the bundle (spec 07, Req 1.2, 1.4, 2).

Annotations are looked up by **exact InChIKey** first. If that fails, the connectivity block is
tried (same skeleton, stereo possibly different), and the match level is recorded so the
reader knows which it was. Annotations describe what databases already say about the compound.
They are never model output and never training labels for this query.
"""

from dataclasses import dataclass

import polars as pl

from oncora.config import ACTIVE_PCHEMBL, DATA_INTERIM, DATA_PROCESSED
from oncora.interpret.bundle import Annotations, ChemblActive, TargetExpression, r3

COMPOUND_BRIDGE = DATA_INTERIM / "compound_bridge.parquet"
CHEMBL_ACTIVITIES = DATA_INTERIM / "chembl_activities.parquet"
TARGET_PAIRS = DATA_PROCESSED / "target_pairs.parquet"
CCLE = DATA_INTERIM / "ccle_expression.parquet"
CCLE_GENES = DATA_INTERIM / "ccle_genes.parquet"


@dataclass
class AnnotationIndex:
    lincs: pl.DataFrame      # inchi_key, connectivity, names, moa, targets (lists)
    chembl: pl.DataFrame     # inchi_key, connectivity, target_chembl_id, target_name, gene, pchembl

    @classmethod
    def load(cls) -> "AnnotationIndex":
        b = pl.read_parquet(COMPOUND_BRIDGE, columns=["inchi_key", "connectivity", "cmap_names", "moas", "targets"])
        lincs = (b.filter(pl.col("inchi_key").is_not_null())
                 .group_by("inchi_key")
                 .agg(pl.col("connectivity").first(),
                      *(pl.col(c).list.explode(keep_nulls=False, empty_as_null=False).unique().sort().alias(a)
                        for c, a in (("cmap_names", "names"), ("moas", "moa"), ("targets", "targets")))))
        genes = (pl.read_parquet(TARGET_PAIRS, columns=["target_chembl_id", "target_gene_symbol"])
                 .unique("target_chembl_id"))
        chembl = (pl.read_parquet(CHEMBL_ACTIVITIES, columns=["inchi_key", "connectivity", "target_chembl_id",
                                                              "target_name", "pchembl_median"])
                  .filter(pl.col("pchembl_median") >= ACTIVE_PCHEMBL)
                  .join(genes, on="target_chembl_id", how="left"))
        return cls(lincs, chembl)

    def lookup(self, inchi_key: str, max_chembl: int = 10) -> Annotations:
        for level, col, key in (("exact", "inchi_key", inchi_key), ("connectivity", "connectivity", inchi_key[:14])):
            lin = self.lincs.filter(pl.col(col) == key)
            ch = self.chembl.filter(pl.col(col) == key)
            if lin.height or ch.height:
                names = sorted({n for row in lin["names"].to_list() for n in row})
                moa = sorted({n for row in lin["moa"].to_list() for n in row})
                targets = sorted({n for row in lin["targets"].to_list() for n in row})
                ch = (ch.group_by("target_chembl_id")
                      .agg(pl.col("target_name").first(), pl.col("target_gene_symbol").first(),
                           pl.col("pchembl_median").max())
                      .sort(["pchembl_median", "target_chembl_id"], descending=[True, False]).head(max_chembl))
                actives = [ChemblActive(target_chembl_id=r["target_chembl_id"], name=r["target_name"],
                                        gene_symbol=r["target_gene_symbol"], pchembl=r3(r["pchembl_median"]))
                           for r in ch.iter_rows(named=True)]
                return Annotations(matched=True, match_level=level, names=names, moa=moa, targets=targets,
                                   chembl_actives=actives)
        return Annotations(matched=False, match_level="none")


def target_annotation_overlap(predicted_genes: list[str | None], ann: Annotations) -> list[str]:
    known = set(ann.targets) | {a.gene_symbol for a in ann.chembl_actives if a.gene_symbol}
    return sorted({g for g in predicted_genes if g and g in known})


class ExpressionLookup:
    """Expression percentile of a gene across all CCLE lines (FR-16, STRETCH)."""

    def __init__(self):
        genes = pl.read_parquet(CCLE_GENES)
        self.col_of = dict(zip(genes["gene_symbol"], genes["gene_col"]))
        self._ccle = None

    def _frame(self) -> pl.DataFrame:
        if self._ccle is None:
            self._ccle = pl.read_parquet(CCLE)
        return self._ccle

    def target_expression(self, genes: list[str], top_lines: list[str]) -> list[TargetExpression]:
        out = []
        df = self._frame()
        for g in genes:
            col = self.col_of.get(g)
            if col is None or not top_lines:
                continue
            ranks = df.select("model_id", (pl.col(col).rank("average") / pl.len()).alias("pct"))
            top = ranks.filter(pl.col("model_id").is_in(top_lines))["pct"]
            if top.len():
                out.append(TargetExpression(gene_symbol=g, mean_percentile_in_top_lines=r3(float(top.mean())),
                                            n_top_lines=top.len()))
        return out
