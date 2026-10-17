# Unseen-compound demo: gefitinib

> **These are computational research predictions, not validated clinical conclusions.**

Gefitinib (`XGALLCVXEZPNRQ`) was excluded from **both** models at table-build time, and
`scripts/demo_unseen.py` re-asserts that absence against each model's stored training index
before predicting. Its 540 ChEMBL pairs and 929 measured cell lines were set aside as ground
truth. Raw output: `XGALLCVXEZPNRQ-UHFFFAOYSA-N.json`.

The nearest training compound has Tanimoto similarity **0.919** — close analogs (erlotinib,
afatinib) remain in training, which is what makes this a realistic "new analog" test rather than
an unrealistically easy or hard one.

## Target prediction: matches known biology

| Rank | Predicted | Score | In held-out ChEMBL ground truth? |
|---|---|---|---|
| 1 | **EGFR** | 1.000 | ✅ pChEMBL 7.76 — gefitinib's clinical target |
| 2 | EPHB4 | 0.963 | not measured |
| 3 | EPHA2 | 0.934 | not measured |
| 4 | **ERBB4** | 0.845 | ✅ pChEMBL 6.59 |
| 5 | **RIPK2** | 0.844 | ✅ pChEMBL 7.31 |

Three of the top five are confirmed by activities the model never saw, and the top prediction is
the correct primary target. EPHB4 and EPHA2 are plausible neighbours: gefitinib's quinazoline
scaffold is shared across kinase inhibitors, and EPH-family kinases are common off-targets of
this chemotype. They are unconfirmed rather than wrong, since ChEMBL has no measurement for them.

## Sensitivity prediction: weak but real, and not mechanistic

Against 929 held-out cell lines with measured DSS3:

- Spearman ρ = **0.137** (p = 2.7 × 10⁻⁵) — significantly positive, modestly predictive.
- Median measured DSS3 is **0.206** in the top-20 predicted lines versus **0.146** across all
  lines: a 41% enrichment at the top of the ranking.

This is consistent with the held-out evaluation (`reports/eval/sensitivity_model.md`): the model
beats the line-average baseline at picking out a compound's most responsive lines, but not at
ordering the full list.

### The expression check contradicts the mechanism

The interpretation layer surfaces a problem rather than a confirmation:

- EGFR expression in the top-25 predicted lines sits at the **15th percentile** of all CCLE lines
  (mean log1p TPM 0.28, against 3.4 across all lines): those lines are nearly EGFR-null.
- Yet in the measured data, DSS3 for gefitinib **positively** correlates with EGFR expression
  (ρ = 0.219 across the same 929 lines).

So the model's top cell-line picks run *opposite* to the real mechanism on the one axis where
gefitinib's biology is best understood, even though its overall ranking correlation is positive.
The sensitivity model is not learning "EGFR-high lines respond to EGFR inhibitors"; it is
learning a weaker, less interpretable signal.

**This is the single most important caveat in the demo.** It is reported rather than hidden
because the expression check exists precisely to catch it, and because a user shown a ranked list
of cell lines deserves to know the ranking is not mechanistically grounded. A reviewer should
weigh the target model (strong, confirmed) separately from the sensitivity model (weak, not
mechanistic) — which is exactly why the two models are trained and validated independently and
their outputs are only ever combined as correlational evidence (D4, FR-15).

## What the evidence summary did

Generated with the **template** renderer (no `ANTHROPIC_API_KEY` set), proving the full pipeline
is judgeable offline (NFR-9). Every claim cites bundle paths; model predictions and database
annotations appear in separate labeled sections; the cross-evidence section states explicitly
that co-occurrence is correlational, not proof.

Cancer-type ranking was unavailable (no DepMap `Model.csv` installed), and the summary says so
rather than silently omitting the section.
