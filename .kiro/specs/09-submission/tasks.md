# Implementation Plan — 09 Evaluation Hardening & Submission

- [x] 1. Demo compound holdout (do this before the final training runs in 05/06)
  - Chosen: **gefitinib** (connectivity `XGALLCVXEZPNRQ`): clean DSS3 scale, 1,110 cell lines of ground truth, EGFR as its top ChEMBL active (pChEMBL 7.76), annotated MoA "EGFR inhibitor"
  - Pick the demo drug; add its InChIKey to `config.DEMO_HOLDOUT`; confirm it is excluded from both modeling tables
  - _Requirements: 1.1, 1.2_

- [ ] 2. Unseen-compound demo script
  - Implement `scripts/demo_unseen.py` with the training-index assertion, full output capture, and predicted-vs-actual comparison
  - Run it; write the commentary in `reports/demo/`
  - Learn: end-to-end evaluation on a true holdout
  - _Requirements: 1.1, 1.3, 1.4, 1.5_

- [ ] 3. Reproducibility pass
- [x] 3.1 Write `scripts/reproduce.sh` and `scripts/compare_metrics.py`; commit the reference metrics
  - _Requirements: 2.1, 2.3_
- [ ] 3.2 Run on a fresh clone, offline, with no API key; fix any nondeterminism; document results in the README
  - Learn: deterministic pipelines
  - _Requirements: 2.2, 2.4_

- [ ] 4. Submission writeup
  - Write `reports/submission.md`: criterion → evidence map, headline metrics vs. baselines with CIs, limitations, disclaimer
  - _Requirements: 3.1, 3.2, 3.3, 3.4_
