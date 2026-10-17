# Implementation Plan — 07 Biological Interpretation Layer

- [ ] 1. Evidence bundle
- [ ] 1.1 Define the pydantic `EvidenceBundle` schema with path resolution and canonical JSON
  - _Requirements: 1.1, 1.3, 1.5, 1.6_
- [ ] 1.2 Implement the exact-InChIKey annotation lookup (compoundinfo, ChEMBL actives), including the explicit "no annotation" case
  - Learn: separating prediction from annotation
  - _Requirements: 1.2, 1.4_
- [ ] 1.3 Implement the derived cross-evidence: target–annotation overlap and target-gene expression percentiles (STRETCH)
  - _Requirements: 2.1, 2.2, 2.3_
- [ ] 1.4 Determinism and path-resolution tests
  - _Requirements: 1.5, 1.6_

- [ ] 2. Template renderer
  - Per-claim citations, separate sections, disclaimer; offline test
  - Learn: template-based text generation, reproducibility
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Claude grounded renderer (STRETCH)
- [ ] 3.1 Implement `render_claude` with the structured claims schema, grounding system prompt, and timeout
  - _Requirements: 4.1, 4.2, 4.5_
- [ ] 3.2 Implement `validate_claims` (path and numeric checks) and the fallback logic in `explain`
  - Learn: grounded prompting, guardrails, cost control
  - _Requirements: 4.3, 4.4, 4.6_
- [ ] 3.3 Stubbed-client tests (missing key, exception, timeout, fabricated claim); usage logging
  - _Requirements: 4.3, 4.4, 4.5_
