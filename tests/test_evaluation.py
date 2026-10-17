import numpy as np
import polars as pl
import pytest

from oncora.data_prep.modeling import collapse_target_pairs
from oncora.evaluation.baselines import CellLineMeanBaseline, GlobalMeanBaseline, PopularityRanker, PriorBaseline
from oncora.evaluation.labels import RankTransform, global_threshold, within_compound_top
from oncora.evaluation.metrics import (
    bootstrap_ci,
    group_index,
    per_group_spearman,
    sensitivity_metrics,
    target_metrics,
    top_k_hit_rate,
)


# --- labels ---------------------------------------------------------------------------------

def test_rank_transform_uses_training_distribution():
    rt = RankTransform().fit(np.array([0, 0, 1, 2, 3.0]))
    assert rt.transform(np.array([0.0]))[0] == pytest.approx(0.2)     # midpoint of the two zeros
    assert rt.transform(np.array([10.0]))[0] == pytest.approx(1.0)
    assert rt.transform(np.array([-1.0]))[0] == pytest.approx(0.0)


def test_within_compound_top_ignores_zero_ties():
    df = pl.DataFrame({
        "inchi_key": ["A"] * 10 + ["B"] * 10,
        "dss3": [0.9, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.05, 0.4, 0.5] + [0.0] * 10,
    })
    lab = within_compound_top(df, 0.10).to_list()
    assert lab[:10] == [True] + [False] * 9          # top 10% of A = its single best line
    assert not any(lab[10:])                         # all-zero compound labels nothing sensitive
    assert global_threshold(np.arange(11.0), 0.9) == pytest.approx(9.0)


# --- baselines ------------------------------------------------------------------------------

def test_baselines_fit_on_training_rows_only():
    train = pl.DataFrame({"model_id": ["L1", "L1", "L2"], "target_chembl_id": ["T1", "T1", "T2"],
                          "label": [1, 0, 1]})
    y = np.array([0.2, 0.4, 0.9])
    clm = CellLineMeanBaseline().fit(train, y)
    test = pl.DataFrame({"model_id": ["L1", "L2", "L9"]})
    assert clm.predict(test).tolist() == pytest.approx([0.3, 0.9, 0.5])
    assert GlobalMeanBaseline().fit(train, y).predict(test).tolist() == pytest.approx([0.5] * 3)
    prior = PriorBaseline().fit(train)
    assert prior.predict(pl.DataFrame({"target_chembl_id": ["T1", "T2", "T9"]})).tolist() == pytest.approx([0.5, 1.0, 2 / 3])
    pop = PopularityRanker().fit(train).score_matrix(2, ["T2", "T1", "T9"])
    assert pop.shape == (2, 3) and pop[0].tolist() == [1, 1, 0]


# --- metrics --------------------------------------------------------------------------------

def test_target_metrics_perfect_and_random():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    t = np.array(["A"] * 4 + ["B"] * 4)
    perfect = target_metrics(y, y.astype(float), t)
    assert perfect["pr_auc_macro"] == pytest.approx(1.0) and perfect["roc_auc_macro"] == pytest.approx(1.0)
    const = target_metrics(y, np.zeros(8), t)
    assert const["roc_auc_macro"] == pytest.approx(0.5)


def test_top_k():
    scores = np.array([[0.9, 0.1, 0.5], [0.1, 0.2, 0.3]])
    out = top_k_hit_rate(scores, [{0}, {0}, set()], ks=(1, 2))
    assert out == {"top1": 0.5, "top2": 0.5}


def test_sensitivity_metrics_separate_potency_and_selectivity():
    # 3 compounds x 4 lines; prediction = compound potency only (no selectivity)
    potency = np.repeat([0.1, 0.5, 0.9], 4)
    selectivity = np.tile([0.0, 0.01, 0.02, 0.03], 3)
    y = potency + selectivity
    comp, line = np.repeat(["a", "b", "c"], 4), np.tile(["l1", "l2", "l3", "l4"], 3)
    m = sensitivity_metrics(y, potency, comp, line, min_group=3)
    assert m["spearman_per_cell_line"] == pytest.approx(1.0)   # potency ranking perfect
    assert np.isnan(m["spearman_per_compound"])                 # constant within compound: undefined
    assert per_group_spearman(y, y, comp, min_size=4).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_bootstrap_resamples_groups():
    groups = np.repeat(["a", "b", "c", "d"], 5)
    vals = np.repeat([1.0, 2.0, 3.0, 4.0], 5)
    lo, hi = bootstrap_ci(lambda idx: vals[idx].mean(), group_index(groups), n=300, seed=0)
    assert 1.0 <= lo < 2.5 < hi <= 4.0


# --- collapse rule (spec 05, Req 1.6) -------------------------------------------------------

def test_collapse_target_pairs():
    df = pl.DataFrame({
        "group": ["G1", "G1", "G1", "G2", "G2", "G3"],
        "inchi_key": ["k1b", "k1a", "k1c", "k2", "k2s", "k3"],
        "scaffold": ["s"] * 6,
        "target_chembl_id": ["T"] * 6,
        "pchembl_median": [7.0, 5.0, None, None, 6.5, None],
        "label": [1, 0, 0, 0, 1, 0],
        "negative_source": [None, "measured", "sampled", "sampled", None, "sampled"],
    })
    out = {r["group"]: r for r in collapse_target_pairs(df).iter_rows(named=True)}
    assert out["G1"]["pchembl"] == pytest.approx(6.0) and out["G1"]["label"] == 1   # median(7,5); sampled dropped
    assert out["G1"]["inchi_key"] == "k1a" and out["G1"]["n_merged"] == 2
    assert out["G2"]["label"] == 1 and out["G2"]["negative_source"] is None         # measured beats sampled
    assert out["G3"]["label"] == 0 and out["G3"]["negative_source"] == "sampled"
