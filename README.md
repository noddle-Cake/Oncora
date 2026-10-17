# Oncora

Speeding cancer drug discovery with explainable AI (eMerge Hackathon Challenge 6).

Given a compound as a SMILES string, Oncora predicts likely protein targets, sensitive cancer
cell lines, and cancer types worth investigating, and explains the evidence behind each.

> **These are computational research predictions, not validated clinical conclusions.**

## Setup

```bash
brew install uv libomp   # once; libomp is required by LightGBM on macOS
uv sync                  # creates .venv with pinned deps (Python 3.11)
uv run pytest
```

Raw data goes in `data/raw/` (not committed). Verify it with:

```bash
uv run python -m oncora.io.checksums
```

## Project docs

- `.kiro/steering/`: product, tech, structure, data sources, rigor rules
- `.kiro/specs/01-…09-*/`: requirements, design, and tasks per feature, in build order
