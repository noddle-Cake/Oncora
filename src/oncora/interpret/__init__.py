"""Biological interpretation layer (spec 07): evidence bundle → evidence summary."""

import logging

from oncora.config import LLM_ENABLED
from oncora.interpret import render_template
from oncora.interpret.bundle import EvidenceBundle
from oncora.interpret.summary import EvidenceSummary

log = logging.getLogger(__name__)


def explain(bundle: EvidenceBundle, *, use_llm: bool = True, client=None) -> EvidenceSummary:
    """Claude prose when enabled and credentials exist; the deterministic template otherwise.

    Any LLM failure (missing key, API error, timeout, refusal, invalid output) falls back to the
    template. It is logged, never shown to the user as an error. Predictions, metrics and
    rankings never depend on this path (NFR-9).
    """
    if use_llm and LLM_ENABLED:
        from oncora.interpret import render_claude

        if client is not None or render_claude.llm_available():
            try:
                return render_claude.render(bundle, client=client)
            except render_claude.RenderError as exc:
                log.warning("LLM renderer unavailable, using template: %s", exc)
                summary = render_template.render(bundle)
                summary.fallback_reason = str(exc)
                return summary
    return render_template.render(bundle)
