---
inclusion: fileMatch
fileMatchPattern: ["src/oncora/interpret/**", "src/oncora/serve/**", "app/**"]
---

# Oncora — LLM Grounding Contract

Claude renders explanations. It does not predict.

- **Input** is only the compact, already-computed `EvidenceBundle` for one query. Never send raw
  dataset rows, bulk numeric data, or training data (FR-16a).
- **Output** restates facts present in the bundle and cites each claim's source field. It adds
  no outside biology. It keeps model predictions and database annotations separate, and frames
  target + sensitivity co-occurrence as correlational (FR-15).
- **Fallback:** with no `ANTHROPIC_API_KEY`, or on any API error or timeout, render the
  deterministic template from the same bundle. Predictions, metrics, and rankings never depend
  on the LLM (NFR-9).
- **Cost:** one bounded call per explanation, made on demand. Model and `max_tokens` are set in
  `config.py`. No batch LLM processing. Total budget is about $100 (NFR-10).
- **Model:** default to the latest capable Claude model, configurable. The key comes from the
  environment only.
- The research disclaimer is attached to every rendered summary, whichever renderer produced it.
