"""L1000 signature profile and validation (spec 01 task 5.3) → reports/eda_signatures.md.

    uv run python -m oncora.eda.signatures

Reads only the extracted consensus table (`oncora.data_prep.signatures`), never the 89 GB GCTX.
Before signatures may be shown as evidence, three questions must come out right:

1. **Mechanism coherence:** do compounds with the same annotated MoA have more similar measured
   responses than compounds with different MoAs? (pairwise ROC-AUC; top-5 neighbor hit rate
   against a label-shuffled baseline), and how does that depend on the reproducibility filter?
2. **Beyond structure:** does it still hold for structurally dissimilar pairs (ECFP4 Tanimoto
   < 0.3)? If not, "responds like" would only rediscover chemical analogs.
3. **Link to sensitivity:** do compounds with similar responses kill the same cell lines? Each
   compound's DSS3 is ranked within the compound, and each line's average rank is subtracted, so
   generally fragile lines do not create agreement. Pairwise similarity of these residual profiles is
   compared with signature correlation (Mantel test: Spearman over pairs, compound-label permutations).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from oncora.config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    REPORTS,
    SEED,
    SIG_MIN_REPRODUCIBILITY,
)
from oncora.data_prep.provenance import read_provenance
from oncora.data_prep.signatures import GENES_OUT as GENES
from oncora.data_prep.signatures import OUT as CONSENSUS
from oncora.eda.common import plt, save_fig
from oncora.interpret.signatures import SignatureIndex, _unit_rows

THRESHOLDS = (None, 0.1, 0.2, 0.3, 0.5)
DISSIMILAR_TANIMOTO = 0.3
MIN_SHARED_LINES = 100
N_PERM = 200
SHOWCASE = ("gefitinib", "erlotinib", "vorinostat", "bortezomib", "trametinib", "palbociclib")


def _fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def one_per_skeleton(t: pl.DataFrame) -> pl.DataFrame:
    """The best-profiled form of each connectivity block, so stereoisomers don't pair with themselves."""
    return t.sort(["n_profiles", "inchi_key"], descending=[True, False]).unique("connectivity", keep="first",
                                                                                maintain_order=True)


def ecfp(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    with rdBase.BlockLogs():
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is not None:
                out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


def tanimoto_matrix(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.float32)
    inter = b @ b.T
    cnt = b.sum(axis=1)
    union = cnt[:, None] + cnt[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _shares_label(labels: list[list[str]]) -> np.ndarray:
    """Boolean matrix: do items i and j share at least one label?"""
    vocab = {m: k for k, m in enumerate(sorted({m for ms in labels for m in ms}))}
    hot = np.zeros((len(labels), len(vocab)), dtype=np.float32)
    for i, ms in enumerate(labels):
        hot[i, [vocab[m] for m in ms]] = 1
    return (hot @ hot.T) > 0


def moa_analysis(t: pl.DataFrame, sig: np.ndarray, smiles: dict[str, str]) -> dict:
    """Pairwise same-MoA AUC of signature correlation, per reproducibility threshold."""
    ann = t.with_row_index("i").filter(pl.col("moa").list.len() > 0)
    rows = []
    rng = np.random.default_rng(SEED)
    detail = {}
    for thr in THRESHOLDS:
        sub = ann.filter(pl.col("reproducibility").is_not_null()) if thr is None else \
            ann.filter(pl.col("reproducibility") >= thr)
        idx = sub["i"].to_numpy()
        if len(idx) < 20:
            continue
        u = _unit_rows(sig[idx].astype(np.float64))
        r = u @ u.T
        same = _shares_label(sub["moa"].to_list())
        iu = np.triu_indices(len(idx), 1)
        auc = roc_auc_score(same[iu], r[iu]) if 0 < same[iu].sum() < len(iu[0]) else float("nan")
        np.fill_diagonal(r, -np.inf)
        top5 = np.argsort(-r, axis=1)[:, :5]
        hit = float(np.mean([same[i, top5[i]].any() for i in range(len(idx))]))
        shuffled = []
        for _ in range(20):
            p = rng.permutation(len(idx))
            shuffled.append(np.mean([same[i, p[top5[i]]].any() for i in range(len(idx))]))
        rows.append({"threshold": thr, "compounds": len(idx), "same_moa_pairs": int(same[iu].sum()),
                     "auc": auc, "top5_hit": hit, "top5_shuffled": float(np.mean(shuffled))})
        if thr == SIG_MIN_REPRODUCIBILITY:
            fp = ecfp([smiles.get(k) for k in sub["inchi_key"].to_list()])
            tan = tanimoto_matrix(fp)[iu]
            rr, ss = (u @ u.T)[iu], same[iu]
            dis = tan < DISSIMILAR_TANIMOTO
            detail = {
                "auc_structure": roc_auc_score(ss, tan),
                "auc_signature_dissimilar": roc_auc_score(ss[dis], rr[dis]),
                "dissimilar_pairs": int(dis.sum()), "dissimilar_same": int(ss[dis].sum()),
                "same_r": rr[ss], "diff_r": rr[~ss],
            }
    return {"rows": rows, **detail}


def sensitivity_residuals() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Compound × line residual selectivity: within-compound DSS3 percentile minus the line's mean."""
    s = (pl.read_parquet(DATA_PROCESSED / "sensitivity.parquet", columns=["inchi_key", "model_id", "dss3", "mixed_scale"])
         .filter(~pl.col("mixed_scale"))
         .with_columns((pl.col("dss3").rank("average").over("inchi_key") / pl.len().over("inchi_key")).alias("pct")))
    s = s.with_columns((pl.col("pct") - pl.col("pct").mean().over("model_id")).alias("res"))
    w = s.pivot(on="model_id", index="inchi_key", values="res").sort("inchi_key")
    keys = w["inchi_key"].to_list()
    m = w.drop("inchi_key").to_numpy().astype(np.float64)
    mask = np.isfinite(m)
    return keys, np.where(mask, m, 0.0), mask.astype(np.float64)


def sensitivity_link(index: SignatureIndex) -> dict:
    keys, res, mask = sensitivity_residuals()
    rows = [index._row(k) for k in keys]
    keep = [i for i, h in enumerate(rows) if h is not None]
    sig_rows = np.array([rows[i][0] for i in keep])
    res, mask = res[keep], mask[keep]
    norms = np.sqrt((res ** 2) @ mask.T)        # per-pair norms over shared lines
    sens = (res @ res.T) / np.where(norms * norms.T > 0, norms * norms.T, np.nan)
    shared = mask @ mask.T
    u = index._unit[sig_rows]
    r = u @ u.T
    rep = index._ref[sig_rows]
    out = {"compounds": len(keep)}
    rng = np.random.default_rng(SEED)
    for name, sel in (("all", np.ones(len(keep), bool)), ("reproducible", rep)):
        idx = np.flatnonzero(sel)
        iu = np.triu_indices(len(idx), 1)
        a = sens[np.ix_(idx, idx)][iu]
        ok = (shared[np.ix_(idx, idx)][iu] >= MIN_SHARED_LINES) & np.isfinite(a)
        ra = rankdata(a[ok])
        rsig = r[np.ix_(idx, idx)]

        def stat(perm):
            b = rsig[np.ix_(perm, perm)][iu][ok]
            return np.corrcoef(ra, rankdata(b))[0, 1]

        obs = stat(np.arange(len(idx)))
        null = np.array([stat(rng.permutation(len(idx))) for _ in range(N_PERM)])
        out[name] = {"compounds": len(idx), "pairs": int(ok.sum()), "rho": float(obs),
                     "p": float((np.sum(null >= obs) + 1) / (N_PERM + 1)), "null_p95": float(np.quantile(null, 0.95))}
    return out


def showcase(index: SignatureIndex, bridge: pl.DataFrame) -> list[str]:
    lines = []
    for name in SHOWCASE:
        hit = bridge.filter(pl.col("cmap_names").list.contains(name) & pl.col("inchi_key").is_not_null())
        if not hit.height:
            continue
        s = index.lookup(hit["inchi_key"][0])
        if s is None:
            continue
        nb = "; ".join(f"{(n.names or [n.inchi_key])[0]} ({', '.join(n.moa) or 'no MoA'}, r {n.correlation:.2f})"
                       for n in s.neighbors) or "— (response not reproducible)"
        lines.append(f"| {name} | {s.n_cell_lines} | {_fmt(s.reproducibility, 2)} | "
                     f"{', '.join(g.gene_symbol for g in s.down[:4])} | {nb} |")
    return lines


def build_report() -> str:
    prov = read_provenance(CONSENSUS)
    t = pl.read_parquet(CONSENSUS)
    index = SignatureIndex.load(CONSENSUS, GENES, DATA_INTERIM / "compound_bridge.parquet")
    bridge = pl.read_parquet(DATA_INTERIM / "compound_bridge.parquet")
    smiles = dict(zip(bridge["inchi_key"], bridge["canonical_smiles"]))
    sk = one_per_skeleton(index.table.with_columns(pl.Series("signature", t["signature"])))
    sig = np.stack(sk["signature"].to_numpy())
    moa = moa_analysis(sk.drop("signature"), sig, smiles)
    link = sensitivity_link(index)
    cfg = prov["config"]

    rep = t["reproducibility"]
    n_rep = int((rep >= SIG_MIN_REPRODUCIBILITY).sum())
    s_keys = pl.read_parquet(DATA_PROCESSED / "sensitivity.parquet", columns=["inchi_key"]).unique()["inchi_key"].to_list()
    s_hits = [index._row(k) for k in s_keys]
    s_rep = sum(1 for h in s_hits if h is not None and index._ref[h[0]])
    gef = index.lookup("XGALLCVXEZPNRQ-UHFFFAOYSA-N")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(rep.drop_nulls().to_numpy(), bins=60, color="#4c72b0")
    ax[0].axvline(SIG_MIN_REPRODUCIBILITY, color="k", ls="--", lw=1)
    ax[0].set(xlabel="reproducibility (half-split r)", ylabel="compounds", title="Are responses reproducible?")
    if "same_r" in moa:
        bins = np.linspace(-0.6, 1, 60)
        ax[1].hist(moa["diff_r"], bins=bins, density=True, alpha=0.6, label="different MoA", color="#999999")
        ax[1].hist(moa["same_r"], bins=bins, density=True, alpha=0.7, label="same MoA", color="#c44e52")
        ax[1].set(xlabel="signature correlation r", ylabel="density",
                  title=f"Reproducible compounds (r ≥ {SIG_MIN_REPRODUCIBILITY})")
        ax[1].legend(frameon=False)
    fig_path = save_fig(fig, "l1000_signatures.png")

    L = [
        "# L1000 Level 4 signatures: profile and validation",
        "",
        "Generated by `uv run python -m oncora.eda.signatures` from the extracted consensus table. Do not edit by hand.",
        "",
        "**Role (D-09): evidence only.** Measured responses are attached to explanations for compounds LINCS "
        "profiled. They are never a model feature, because a new SMILES has no measured signature.",
        "",
        "## 1. The file and what was extracted",
        "",
        f"- `level4_beta_trt_cp_n1805898x12328.gctx`: one uncompressed, unchunked float32 HDF5 matrix, "
        f"{cfg['profiles']['total']:,} profiles × 12,328 genes (~89 GB), plate-normalized z-scores, every profile a "
        "compound treatment (`trt_cp`). SHA256 `" + str(cfg.get("gctx_sha256")) + "`.",
        "- 978 rows are the measured **landmark** genes; the other 11,350 are computationally inferred and are not used.",
        f"- Eligible profiles: QC pass, dose ≥ {cfg['min_dose_um']} µM, structure in the compound bridge: "
        f"**{cfg['profiles']['eligible']:,}**. One sequential pass (~90 s, ~6 GB RAM) reduced them to "
        f"**{t.height:,} compound consensus signatures** (`data/interim/l1000_consensus.parquet`, ~75 MB), "
        f"covering {t['connectivity'].n_unique():,} skeletons. The 89 GB file is not needed after this step.",
        f"- Consensus = {cfg['consensus']}. Profiles per compound: median {int(t['n_profiles'].median())}; "
        f"cell lines per compound: median {int(t['n_cell_lines'].median())}, max {int(t['n_cell_lines'].max())}.",
        "",
        "## 2. Coverage of Oncora's compounds",
        "",
        f"- Sensitivity (DSS3) compounds with a measured signature: **{sum(h is not None for h in s_hits)} / {len(s_keys)}**; "
        f"with a reproducible one: {s_rep}.",
        (f"- Demo compound gefitinib: {gef.n_profiles:,} profiles in {gef.n_cell_lines} cell lines, reproducibility "
         f"{_fmt(gef.reproducibility, 2)}." if gef else "- Demo compound gefitinib: no signature."),
        "- A new SMILES that LINCS never profiled has none: the evidence is simply absent for it.",
        "",
        "## 3. Are responses reproducible?",
        "",
        f"![signatures]({fig_path})",
        "",
        f"Reproducibility = correlation between consensus signatures built from two disjoint halves of a compound's "
        f"cell lines. Of {rep.drop_nulls().len():,} compounds profiled in ≥ 2 lines, the median is "
        f"{_fmt(rep.median(), 2)}, and **{n_rep:,} ({n_rep / rep.drop_nulls().len():.0%}) reach "
        f"r ≥ {SIG_MIN_REPRODUCIBILITY}**. Most compounds barely perturb transcription at the doses tested, so their "
        "consensus is mostly noise. Only reproducible signatures are offered as \"responds like\" references, and a "
        "query whose own response is irreproducible gets no neighbors.",
        "",
        "## 4. Do same-mechanism compounds respond alike?",
        "",
        "One compound per skeleton, restricted to compounds with a LINCS MoA annotation. AUC = probability that a "
        "same-MoA pair correlates more than a different-MoA pair (0.5 = no information). Top-5 hit = share of "
        "compounds whose 5 most similar signatures include a same-MoA compound; shuffled = the same with MoA labels "
        "permuted (20×).",
        "",
        "| reproducibility filter | compounds | same-MoA pairs | AUC | top-5 hit | top-5 shuffled |",
        "|---|---|---|---|---|---|",
    ]
    for r in moa["rows"]:
        label = "none (≥ 2 lines)" if r["threshold"] is None else f"r ≥ {r['threshold']}"
        L.append(f"| {label} | {r['compounds']:,} | {r['same_moa_pairs']:,} | {_fmt(r['auc'])} | "
                 f"{_fmt(r['top5_hit'])} | {_fmt(r['top5_shuffled'])} |")
    L += [
        "",
        f"Coherence rises steadily with reproducibility. The cutoff r ≥ {SIG_MIN_REPRODUCIBILITY} "
        "(`SIG_MIN_REPRODUCIBILITY`) was fixed before this analysis and is kept: it already lifts AUC and top-5 hits "
        "above the unfiltered set while keeping about twice as many reference compounds as r ≥ 0.5. Note that MoA "
        "annotations are incomplete (several neighbors below are known drugs with no LINCS MoA), so these numbers "
        "understate coherence.",
    ]
    if "auc_structure" in moa:
        L += [
            "",
            f"**Beyond structure (r ≥ {SIG_MIN_REPRODUCIBILITY} set):** ECFP4 Tanimoto alone separates same-MoA pairs "
            f"with AUC {_fmt(moa['auc_structure'])}. Restricted to structurally dissimilar pairs (Tanimoto < "
            f"{DISSIMILAR_TANIMOTO}; {moa['dissimilar_pairs']:,} pairs, {moa['dissimilar_same']:,} same-MoA), the signature "
            f"AUC is **{_fmt(moa['auc_signature_dissimilar'])}**: the measured response carries mechanism information "
            "that the chemical structure does not.",
        ]
    L += [
        "",
        "## 5. Do compounds that respond alike kill the same cell lines?",
        "",
        "DSS3 is ranked within each compound and each line's average rank is subtracted (removing generally fragile "
        f"lines). For every compound pair sharing ≥ {MIN_SHARED_LINES} lines, the similarity of these residual "
        "selectivity profiles is compared with their signature correlation. Mantel test: Spearman ρ over pairs; the "
        f"null permutes compound labels ({N_PERM}×).",
        "",
        "| compounds | pairs | Mantel ρ | null 95th pct | p |",
        "|---|---|---|---|---|",
    ]
    for name in ("all", "reproducible"):
        x = link[name]
        floor = 1 / (N_PERM + 1)
        p = f"< {floor:.3f} (no permutation reached it)" if x["p"] <= floor else f"{x['p']:.3f}"
        L.append(f"| {name} ({x['compounds']}) | {x['pairs']:,} | {_fmt(x['rho'])} | {_fmt(x['null_p95'])} | {p} |")
    L += [
        "",
        "## 6. Examples (what the evidence bundle shows)",
        "",
        "| compound | cell lines | reproducibility | most down-regulated | responds like (r) |",
        "|---|---|---|---|---|",
        *showcase(index, bridge),
        "",
        "## Use in Oncora",
        "",
        "- `annotations.measured_signature` (spec 07 Req 1.4): top up/down landmark genes, profile counts, "
        "reproducibility, and up to 5 \"responds like\" compounds with their LINCS names, MoAs and targets.",
        "- `derived.signature_target_overlap`: predicted targets that are annotated targets of those neighbors. The "
        "structure-based prediction and the measured cell response are independent, so agreement is worth showing, "
        "but it remains correlational.",
        "- Never a model feature: `assert_structure_only` rejects any signature column (FR-10).",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPORTS / "eda_signatures.md")
    args = ap.parse_args()
    if not CONSENSUS.exists() or not GENES.exists():
        raise SystemExit("Run `python -m oncora.data_prep.signatures` first (needs the optional L1000 GCTX).")
    args.out.write_text(build_report())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
