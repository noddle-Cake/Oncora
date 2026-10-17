# Implementation Plan — 08 Serving & UI

- [x] 1. API schemas and shared disclaimer
  - Define the pydantic request/response models; put `DISCLAIMER` in `config.py`
  - _Requirements: 1.2, 1.5_

- [x] 2. FastAPI service
- [x] 2.1 Lifespan artifact loading and `/health`
  - Learn: typed inference APIs, load-once artifact serving
  - _Requirements: 1.1, 1.4_
- [x] 2.2 `/predict` orchestration with per-stage timings, invalid-SMILES 422 path, and A4-missing fallback
  - _Requirements: 1.2, 1.3, 1.6, 3.2_
- [x] 2.3 API contract tests with fixture artifacts
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3. Streamlit UI
  - Disclaimer banner, molecule render, three ranked views, styled prediction vs. annotation, near-seen notice, inline errors, example SMILES
  - Learn: rapid data-app UIs
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 4. Latency check (STRETCH)
  - Measure `/predict` on the M1 Max; optimize the hot path if it exceeds 5 s
  - _Requirements: 3.1_

- [x] 5. Batch endpoint (STRETCH)
  - `/predict/batch` with a size cap and the template renderer only
  - _Requirements: 4.1, 4.2_
