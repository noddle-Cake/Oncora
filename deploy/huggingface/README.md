---
title: Oncora
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Compound → protein targets → sensitive cancer cell lines
---

# Oncora

Paste a molecule as a SMILES string. Oncora predicts likely **protein targets**, **sensitive cancer
cell lines**, and explains the evidence behind them, citing each source. It is built for compounds
never seen in training. Try the gefitinib example: it was held out of all training data.

> **These are computational research predictions, not validated clinical conclusions.**

- Target model: 1,360 per-target LightGBM classifiers on ECFP4 fingerprints, validated on compounds
  disjoint from training.
- Sensitivity model: LightGBM on compound structure × cell-line expression (978 landmark genes).
- Evidence: ChEMBL measured activities, LINCS annotations, and measured L1000 expression responses,
  kept separate from model predictions.

## Data and licences

Derived from ChEMBL 37 (EMBL-EBI, CC BY-SA 3.0), LINCS L1000 / CLUE (Broad Institute), DepMap/CCLE
(Broad Institute), and PharmacoDB. Each source's own terms apply.
