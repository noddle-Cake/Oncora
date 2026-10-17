"""Single source of truth for paths, seeds, thresholds, and feature config.

Other modules import from here instead of hard-coding values, so a run is fully described by
this file plus the raw data checksums (NFR-2).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

_LINCS = "lincs/StandardAnnotations"

# Logical name -> path relative to DATA_RAW.
RAW = {
    "ccle": "ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    "dss3": "pharmacodb/L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv",
    "chembl_tar": "chembl/chembl_37_postgresql.tar",
    "compoundinfo": f"{_LINCS}/compoundinfo.txt",
    "pertinfo": f"{_LINCS}/pertinfo.txt",
    "cellinfo": f"{_LINCS}/cellinfo.txt",
    "geneinfo": f"{_LINCS}/geneinfo.txt",
    "instinfo": f"{_LINCS}/instinfo.txt",
    "sm_metadata": f"{_LINCS}/Small_Molecule_Metadata.txt",
    "field_definitions": f"{_LINCS}/DataFieldDefinitions.xlsx",
}


def raw_path(name: str) -> Path:
    return DATA_RAW / RAW[name]


SEED = 42

# --- Spec 01 checkpoint decisions (reports/eda_summary.md) ---------------------------------
# D-01: splits group on the InChIKey connectivity block (first 14 chars), not the full key.
SPLIT_GROUP_KEY = "connectivity"
# D-03: compounds with any DSS3 >= this are mixed-scale; excluded from headline training.
DSS3_SCALE_BREAK = 1.0
DSS3_RESCALE_FACTOR = 0.01  # sensitivity analysis only: values >= break are multiplied by this
# D-04 / D-05: regression on ranks; "sensitive" = top fraction of a compound's cell lines.
DSS3_TRANSFORM = "rank"
SENSITIVE_LABEL = "within_compound_top_frac"
SENSITIVE_TOP_FRAC = 0.10
# D-06: cell-line features = LINCS landmark genes (geneinfo feature_space == "landmark").
CELLLINE_FEATURES = "landmark"

# LINCS annotation files write missing values as a literal pair of double quotes.
NULL_TOKENS = ("", '""')

DISCLAIMER = (
    "These are computational research predictions, not validated clinical conclusions."
)
