import h5py
import numpy as np
import polars as pl
import pytest

from oncora.data_prep import signatures as sig

GENES = ["1", "2", "3", "4"]          # GCTX row ids (Entrez); 1, 3, 4 are landmarks


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.provenance.ROOT", tmp_path)
    #           sample   pert  cell  dose  unit  qc
    profiles = [("s0", "P1", "A", "10", "uM", "1"),
                ("s1", "P1", "A", "10", "uM", "1"),
                ("s2", "P1", "B", "10", "uM", "1"),
                ("s3", "P1", "B", "0.1", "uM", "1"),   # below the dose floor
                ("s4", "P2", "A", "10", "uM", "0"),    # fails QC
                ("s5", "P2", "A", "10", "uM", "1"),
                ("s6", "P9", "A", "10", "uM", "1")]    # no structure in the bridge
    m = np.arange(len(profiles) * len(GENES), dtype=np.float32).reshape(len(profiles), len(GENES))
    gctx = tmp_path / "x.gctx"
    with h5py.File(gctx, "w") as f:
        f[sig.MATRIX] = m
        f[sig.COL_IDS] = np.array([p[0] for p in profiles], dtype="S")
        f[sig.ROW_IDS] = np.array(GENES, dtype="S")
    inst = tmp_path / "instinfo.txt"
    inst.write_text("sample_id\tpert_id\tcell_iname\tpert_dose\tpert_dose_unit\tpert_time\tqc_pass\n"
                    + "".join(f"{s}\t{p}\t{c}\t{d}\t{u}\t24\t{q}\n" for s, p, c, d, u, q in profiles))
    gi = tmp_path / "geneinfo.txt"
    gi.write_text("gene_id\tgene_symbol\tfeature_space\n1\tG1\tlandmark\n2\tG2\tinferred\n"
                  "3\tG3\tlandmark\n4\tG4\tlandmark\n")
    bridge = tmp_path / "bridge.parquet"
    pl.DataFrame({"pert_id": ["P1", "P2"], "inchi_key": ["AAAAAAAAAAAAAA-XX-N", "BBBBBBBBBBBBBB-YY-N"],
                  "connectivity": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]}).write_parquet(bridge)
    return {"gctx": gctx, "instinfo": inst, "geneinfo": gi, "bridge": bridge, "m": m}


def _build(raw, tmp_path, block_rows=2):
    return sig.build(raw["gctx"], tmp_path / "out.parquet", tmp_path / "genes.parquet", instinfo=raw["instinfo"],
                     bridge=raw["bridge"], geneinfo=raw["geneinfo"], block_rows=block_rows, digest=False)


def test_consensus_filters_and_averages_per_cell_line_first(raw, tmp_path):
    t = _build(raw, tmp_path).sort("inchi_key")
    m, lm = raw["m"], [0, 2, 3]
    a = t.row(0, named=True)
    # P1: cell A = mean(s0, s1); cell B = s2 only (s3 is below the dose floor); then mean over cells
    expect = (m[[0, 1]][:, lm].mean(axis=0) + m[2, lm]) / 2
    np.testing.assert_allclose(np.asarray(a["signature"]), expect, rtol=1e-6)
    assert (a["n_profiles"], a["n_cell_lines"]) == (3, 2)
    b = t.row(1, named=True)
    np.testing.assert_allclose(np.asarray(b["signature"]), m[5, lm], rtol=1e-6)   # s4 failed QC
    assert b["reproducibility"] is None                                            # one cell line
    assert t.height == 2                                                           # P9 has no structure


def test_block_size_does_not_change_result(raw, tmp_path):
    small = _build(raw, tmp_path, block_rows=1).sort("inchi_key")
    big = _build(raw, tmp_path, block_rows=1000).sort("inchi_key")
    np.testing.assert_allclose(np.stack(small["signature"].to_numpy()), np.stack(big["signature"].to_numpy()))


def test_gene_order_is_persisted(raw, tmp_path):
    _build(raw, tmp_path)
    assert pl.read_parquet(tmp_path / "genes.parquet")["gene_symbol"].to_list() == ["G1", "G3", "G4"]


def test_missing_gctx_is_an_explanatory_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="optional"):
        sig.build(tmp_path / "absent.gctx", tmp_path / "o.parquet", tmp_path / "g.parquet", digest=False)


def _index(reps=(0.9, 0.8, 0.95, 0.1)):
    from oncora.interpret.signatures import SignatureIndex
    base = np.array([3.0, -2.0, 1.0, 0.0, -0.5])
    sigs = np.stack([base,                                # Q: the query
                     base * 0.9 + 0.01,                   # same skeleton as Q (a stereoisomer)
                     base * 0.5 + np.array([0, 0, 0, 0.2, 0]),   # a different compound responding alike
                     base * 1.1])                         # alike but irreproducible -> not a reference
    table = pl.DataFrame({
        "inchi_key": ["QQQQQQQQQQQQQQ-AA-N", "QQQQQQQQQQQQQQ-BB-N", "RRRRRRRRRRRRRR-AA-N", "SSSSSSSSSSSSSS-AA-N"],
        "connectivity": ["QQQQQQQQQQQQQQ", "QQQQQQQQQQQQQQ", "RRRRRRRRRRRRRR", "SSSSSSSSSSSSSS"],
        "n_profiles": [40, 10, 30, 5], "n_cell_lines": [6, 2, 4, 2], "reproducibility": list(reps),
        "names": [["q"], ["q-stereo"], ["erlotinib"], ["noise"]], "moa": [[], [], ["EGFR inhibitor"], []],
        "targets": [[], [], ["EGFR"], []]})
    return SignatureIndex(table, sigs.astype(np.float32), ["G1", "G2", "G3", "G4", "G5"])


def test_lookup_reports_top_genes_and_reproducible_neighbors_only():
    s = _index().lookup("QQQQQQQQQQQQQQ-AA-N")
    assert s.match_level == "exact" and s.reproducible
    assert [g.gene_symbol for g in s.up] == ["G1", "G3"] and [g.gene_symbol for g in s.down] == ["G2", "G5"]
    # the stereoisomer shares the skeleton and the irreproducible compound is not a reference
    assert [n.names[0] for n in s.neighbors] == ["erlotinib"]
    assert s.neighbors[0].correlation > 0.9


def test_connectivity_match_and_absent_compound():
    idx = _index()
    assert idx.lookup("QQQQQQQQQQQQQQ-ZZ-N").match_level == "connectivity"
    assert idx.lookup("XXXXXXXXXXXXXX-AA-N") is None


def test_irreproducible_query_gets_no_neighbors():
    s = _index(reps=(0.05, 0.8, 0.95, 0.1)).lookup("QQQQQQQQQQQQQQ-AA-N")
    assert not s.reproducible and s.neighbors == []


def test_bundle_and_template_cite_the_measured_signature():
    from types import SimpleNamespace

    from oncora.chem.similarity import NearestNeighbor
    from oncora.interpret import render_template
    from oncora.interpret.annotations import AnnotationIndex
    from oncora.interpret.builder import build_bundle

    ik = "QQQQQQQQQQQQQQ-AA-N"
    ann = AnnotationIndex(pl.DataFrame({"inchi_key": [ik], "connectivity": [ik[:14]], "names": [["q"]],
                                        "moa": [[]], "targets": [[]]}),
                          pl.DataFrame(schema={"inchi_key": pl.String, "connectivity": pl.String,
                                               "target_chembl_id": pl.String, "target_name": pl.String,
                                               "pchembl_median": pl.Float64, "target_gene_symbol": pl.String}))
    t = [SimpleNamespace(rank=1, target_chembl_id="CHEMBL203", pref_name="EGFR", gene_symbol="EGFR", score=0.9)]
    kw = {"input_smiles": "s", "canonical_smiles": "s", "inchi_key": ik, "targets": t, "cell_lines": [],
          "cancer_types": [], "cancer_types_note": None, "near_seen": {"target": NearestNeighbor(0.5, "O", False)},
          "model_runs": {"target": "a", "sensitivity": "b"}, "selectivity_learned": None, "annotations": ann}
    with_sig, without = build_bundle(**kw, signatures=_index()), build_bundle(**kw)
    assert without.annotations.measured_signature is None
    assert with_sig.annotations.measured_signature.kind == "annotation"
    assert with_sig.derived.signature_target_overlap == ["EGFR"]
    assert with_sig.to_canonical_json() == build_bundle(**kw, signatures=_index()).to_canonical_json()
    claims = render_template.render(with_sig).claims
    assert any("most resembles: erlotinib" in c.text for c in claims)
    assert all(with_sig.has(p) for c in claims for p in c.cites)


def test_reproducibility_is_half_split_correlation():
    groups = pl.DataFrame({"inchi_key": ["K"] * 4, "cell_iname": ["a", "b", "c", "d"], "connectivity": ["K"] * 4,
                           "pert_ids": [["P"]] * 4, "n": [1] * 4})
    base = np.array([1.0, -1.0, 2.0, 0.0])
    means = np.stack([base, base * 2, base + 0.1, base * 3])
    r = sig.consensus(groups, means)["reproducibility"][0]
    assert r == pytest.approx(1.0, abs=1e-3)   # halves (a, c) and (b, d) are both ~proportional to base
