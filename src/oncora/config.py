"""Single source of truth for paths, seeds, thresholds, and feature config.

Other modules import from here instead of hard-coding values, so a run is fully described by
this file plus the raw data checksums (NFR-2).
"""

import os
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

# --- Spec 02: ChEMBL label policy (DR-5, DR-6) --------------------------------------------
CHEMBL_MIN_CONFIDENCE = 8        # direct (9) or homologous (8) single-protein assignment
ACTIVE_PCHEMBL = 6.0             # active ⇔ pChEMBL ≥ 6 (≤ 1 µM)
NEG_RATIO = 3                    # target negatives : positives, topped up by sampling
MIN_POSITIVES_PER_TARGET = 30

# --- Spec 09: unseen-compound demo -------------------------------------------------------
# Connectivity blocks removed from every modeling table before training, so the demo compound
# is truly unseen by both models (stereo variants included). Its labels are kept aside for
# comparison in processed/demo_holdout_*.parquet.
DEMO_HOLDOUT = {"XGALLCVXEZPNRQ": "gefitinib"}

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

# --- Spec 03 ------------------------------------------------------------------------------
NEAR_SEEN_TANIMOTO = 0.99        # FR-4: flag queries this similar to a training compound

# --- Spec 04: evaluation protocol --------------------------------------------------------
TEST_FRAC = 0.2                  # group holdout share for test
VAL_FRAC = 0.1                   # share of the remaining groups used for early stopping/calibration
BOOTSTRAP_N = 200                # compound-level bootstrap resamples for CIs
TOPK_MAX_COMPOUNDS = 20_000      # test compounds sampled for the target top-k metric
GLOBAL_SENSITIVE_QUANTILE = 0.90 # comparison label (D-05)
REPORTS_EVAL = REPORTS / "eval"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"

# --- Spec 05: target model ----------------------------------------------------------------
TARGET_LGBM = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "colsample_bytree": 0.5,
    "min_child_samples": 10, "subsample": 0.8, "subsample_freq": 1, "verbose": -1,
}
TARGET_EARLY_STOPPING = 30
TARGET_TOP_N = 20

# --- Spec 06: sensitivity model -----------------------------------------------------------
SENS_LGBM = {
    "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63, "colsample_bytree": 0.3,
    "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "verbose": -1,
}
SENS_EARLY_STOPPING = 50
SENS_TOP_N = 25
CANCER_TOP_K = 3            # aggregate a disease by the mean of its top-k predicted lines
CANCER_MIN_LINES = 3        # fewer lines than this -> flagged low-support

# --- Spec 07: interpretation / LLM (D5, NFR-9, NFR-10) ------------------------------------
LLM_ENABLED = os.environ.get("ONCORA_LLM_ENABLED", "1") == "1"
LLM_MODEL = os.environ.get("ONCORA_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = "low"          # restating a small bundle needs little reasoning
LLM_MAX_TOKENS = 2000
LLM_TIMEOUT_S = 30.0
LLM_USAGE_LOG = REPORTS / "llm_usage.jsonl"
# Applicability domain: below this heavy-atom count, ECFP features are too sparse to place a
# compound's targets reliably (measured: paracetamol/aspirin rank their known targets 128-458 of
# 1,360, while imatinib/dasatinib rank theirs 2-6). See reports/submission.md.
MIN_RELIABLE_HEAVY_ATOMS = 20
EVIDENCE_TOP_TARGETS = 10
EVIDENCE_TOP_LINES = 25

# LINCS annotation files write missing values as a literal pair of double quotes.
NULL_TOKENS = ("", '""')

DISCLAIMER = (
    "These are computational research predictions, not validated clinical conclusions."
)
