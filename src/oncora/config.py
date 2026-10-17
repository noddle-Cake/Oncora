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

# LINCS annotation files write missing values as a literal pair of double quotes.
NULL_TOKENS = ("", '""')

DISCLAIMER = (
    "These are computational research predictions, not validated clinical conclusions."
)
