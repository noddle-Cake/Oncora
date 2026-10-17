import numpy as np
import polars as pl

from oncora.evaluation.protocol import make_partition
from oncora.evaluation.report import EvalReport, EvalResult


def _report(model_lo):
    return EvalReport(
        "t", "sensitivity",
        [EvalResult("model", "compound", {"rho": 0.5, "auc": 0.6}, {"rho": (model_lo, 0.6), "auc": (0.55, 0.65)}),
         EvalResult("cell_line_mean", "compound", {"rho": 0.3, "auc": 0.62})],
        primary=["rho", "auc"],
    )


def test_verdict_uses_ci_lower_bound():
    ok = _report(model_lo=0.35)
    assert ok.beats_baselines()["compound"]["cell_line_mean"] == {"rho": True, "auc": False}
    assert ok.accepted()
    assert not _report(model_lo=0.25).accepted()     # point estimate higher, but CI overlaps
    md = ok.to_markdown()
    assert "Accepted: yes" in md and "**0.500** [0.350, 0.600]" in md


def test_partition_is_group_disjoint_and_covers_all_rows():
    rng = np.random.default_rng(0)
    groups = [f"g{i}" for i in rng.integers(0, 400, 5000)]
    df = pl.DataFrame({"group": groups, "scaffold": [f"s{int(g[1:]) % 60}" for g in groups]})
    for strategy in ("compound", "scaffold"):
        p = make_partition(df, strategy, seed=1)
        assert (p.train.astype(int) + p.val + p.test == 1).all()
        g = np.asarray(groups)
        assert not set(g[p.train]) & set(g[p.test]) and not set(g[p.val]) & set(g[p.test])
        assert 0.1 < p.test.mean() < 0.3 and p.val.mean() > 0
