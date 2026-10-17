"""One query, end to end: SMILES → targets, cell lines, cancer types, evidence summary (spec 08).

Models and lookup tables load once (FR-23); each call is pure inference.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np

from oncora.chem.featurize import featurize_mols
from oncora.chem.similarity import pack_bits
from oncora.chem.validate import parse_smiles
from oncora.config import DISCLAIMER, SENS_TOP_N, TARGET_TOP_N
from oncora.interpret import explain
from oncora.interpret.annotations import AnnotationIndex, ExpressionLookup
from oncora.interpret.builder import build_bundle
from oncora.interpret.signatures import SignatureIndex
from oncora.models.sensitivity.aggregate import (
    DiseaseMapMissing,
    aggregate,
    load_disease_map,
)
from oncora.models.sensitivity.predict import CellLinePrediction, load_sensitivity_model
from oncora.models.target.predict import load_target_model

log = logging.getLogger(__name__)
NO_DISEASE_MAP_NOTE = ("Cancer-type ranking is unavailable: no cell line → disease map is installed "
                       "(add DepMap Model.csv, decision D-07). Cell-line rankings are unaffected.")


class OncoraPipeline:
    def __init__(self, target_run: str | None = None, sensitivity_run: str | None = None):
        self.target = load_target_model(target_run)
        self.sensitivity = load_sensitivity_model(sensitivity_run)
        if self.target.cfg != self.sensitivity.cfg:
            raise RuntimeError("target and sensitivity models were trained with different compound feature configs")
        self.cfg = self.target.cfg
        self.annotations = AnnotationIndex.load()
        self.expression = ExpressionLookup()
        self.signatures = SignatureIndex.load_optional()   # None without the optional L1000 extract (D-09)
        try:
            self.disease_map = load_disease_map()
            self.sensitivity.set_cell_line_names(
                dict(zip(self.disease_map["model_id"], self.disease_map.get_column("cell_line_name")))
                if "cell_line_name" in self.disease_map.columns else {})
        except DiseaseMapMissing:
            self.disease_map = None

    @property
    def model_versions(self) -> dict[str, str]:
        return {"target": self.target.run_id, "sensitivity": self.sensitivity.run_id}

    def predict(self, smiles: str, *, explain_with_llm: bool = True,
                top_targets: int = TARGET_TOP_N, top_lines: int = SENS_TOP_N) -> dict:
        timings: dict[str, float] = {}

        @contextmanager
        def timed(name):
            t = time.perf_counter()
            yield
            timings[name] = round((time.perf_counter() - t) * 1000, 1)

        with timed("parse"):
            parsed = parse_smiles(smiles)
        if not parsed.valid:
            return {"valid": False, "input_smiles": smiles, "error": parsed.error, "disclaimer": DISCLAIMER}

        with timed("featurize"):
            x = featurize_mols([parsed.mol], self.cfg)
            packed = pack_bits(x[0, : self.cfg.n_bits].astype(np.uint8))
        with timed("targets"):
            targets = self.target.predict(x, top_n=top_targets)
        with timed("sensitivity"):
            all_lines = self.sensitivity.predict_all(x[0])
            lines = [CellLinePrediction(r["model_id"], self.sensitivity.names.get(r["model_id"]),
                                        float(r["predicted_percentile"]), float(r["predicted_dss3"]),
                                        float(r["p_sensitive"]), int(r["rank"]))
                     for r in all_lines.head(top_lines).iter_rows(named=True)]
        with timed("cancer_types"):
            cancer, note = [], NO_DISEASE_MAP_NOTE
            if self.disease_map is not None:
                cancer, unmapped = aggregate(all_lines, self.disease_map)
                note = f"{unmapped} scored cell lines have no disease mapping and are not aggregated." if unmapped else None
        with timed("near_seen"):
            near = {"target": self.target.nearest_training(packed), "sensitivity": self.sensitivity.nearest_training(packed)}
        with timed("evidence"):
            bundle = build_bundle(
                input_smiles=smiles, canonical_smiles=parsed.canonical_smiles, inchi_key=parsed.inchi_key,
                # Without a disease map, cancer types are out of scope rather than "missing": keep the
                # API's note, but leave it out of the evidence the explanation is written from.
                targets=targets, cell_lines=lines, cancer_types=cancer,
                cancer_types_note=note if self.disease_map is not None else None,
                near_seen=near, model_runs=self.model_versions,
                selectivity_learned=self.sensitivity.card.get("learned_selectivity"),
                annotations=self.annotations, expression=self.expression,
                heavy_atoms=parsed.mol.GetNumHeavyAtoms(), signatures=self.signatures)
        with timed("explain"):
            summary = explain(bundle, use_llm=explain_with_llm)

        return {
            "valid": True, "input_smiles": smiles, "canonical_smiles": parsed.canonical_smiles,
            "inchi_key": parsed.inchi_key,
            "targets": [asdict(t) for t in targets],
            "sensitive_cell_lines": [asdict(c) for c in lines],
            "cancer_types": [asdict(c) for c in cancer], "cancer_types_note": note,
            "evidence_summary": summary.model_dump(),
            "evidence_bundle": bundle.model_dump(mode="json"),
            "near_seen": {k: {"flag": v.near_seen, "max_tanimoto": round(v.max_tanimoto, 3),
                              "nearest_inchi_key": v.inchi_key} for k, v in near.items()},
            "selectivity_learned": self.sensitivity.card.get("learned_selectivity"),
            "model_versions": self.model_versions, "timings_ms": timings, "disclaimer": DISCLAIMER,
        }
