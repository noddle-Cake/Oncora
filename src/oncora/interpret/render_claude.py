"""Claude grounded renderer: optional, falls back to the template (spec 07, Req 4; FR-16a, NFR-10).

Grounding contract:
- the only input is the canonical JSON of one query's evidence bundle (no dataset rows);
- the output is constrained by **structured outputs** to a list of claims, each citing bundle paths;
- every claim is validated: each cited path must exist, and every number in the text must equal
  a bundle value at 3 significant figures. Failing claims are dropped. If none survive, or on any
  error, timeout or refusal, the caller falls back to the template;
- one bounded call per explanation (low effort, capped max_tokens); token usage is appended to
  `reports/llm_usage.jsonl` to track spend against the ~$100 budget.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

from oncora.config import LLM_EFFORT, LLM_MAX_TOKENS, LLM_MODEL, LLM_TIMEOUT_S, LLM_USAGE_LOG
from oncora.interpret.bundle import EvidenceBundle, r3
from oncora.interpret.summary import Claim, EvidenceSummary

SYSTEM = """You turn a computational drug-discovery evidence bundle into a short, readable explanation.

Rules:
- Use only facts present in the bundle JSON. Add no outside biology, mechanisms, or drug knowledge.
- Every claim must cite the bundle paths it relies on, e.g. "predictions.targets[0].score".
- Put model outputs in section "prediction", database facts in section "annotation", and links between them in "cross".
- Never state or imply that one model's prediction proves or causes another's; co-occurrence is correlational.
- annotations.measured_signature is measured lab data, not a prediction. A similar measured response
  (its neighbors) suggests but does not prove a shared mechanism. If it is not reproducible, call it inconclusive.
- When you mention a number, copy it exactly as written in the bundle.
- Do not add a disclaimer; the system appends it.
- Write 4 to 8 claims, each one or two plain sentences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["prediction", "annotation", "cross"]},
                    "text": {"type": "string"},
                    "cites": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["section", "text", "cites"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_NUMBER = re.compile(r"(?<![A-Za-z0-9_\-])\d+(?:\.\d+)?(?:[eE]-?\d+)?")


class RenderError(RuntimeError):
    pass


def _bundle_numbers(node, out: set[float]) -> set[float]:
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        out.add(r3(float(node)))
    elif isinstance(node, dict):
        for v in node.values():
            _bundle_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _bundle_numbers(v, out)
    return out


def validate_claims(raw_claims: list[dict], bundle: EvidenceBundle) -> tuple[list[Claim], int]:
    """Keep claims whose cited paths exist and whose numbers all appear in the bundle."""
    allowed = _bundle_numbers(bundle.model_dump(mode="json"), set())
    kept, dropped = [], 0
    for c in raw_claims:
        try:
            claim = Claim(**c)
        except Exception:
            dropped += 1
            continue
        paths_ok = bool(claim.cites) and all(bundle.has(p) for p in claim.cites)
        numbers_ok = all(r3(float(n)) in allowed for n in _NUMBER.findall(claim.text))
        if paths_ok and numbers_ok:
            kept.append(claim)
        else:
            dropped += 1
    return kept, dropped


def _log_usage(response, seconds: float) -> None:
    LLM_USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    u = response.usage
    rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": response.model,
           "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "seconds": round(seconds, 2),
           "stop_reason": response.stop_reason}
    with open(LLM_USAGE_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def render(bundle: EvidenceBundle, client=None) -> EvidenceSummary:
    """Raise RenderError on any failure; the caller falls back to the template."""
    import anthropic

    client = client or anthropic.Anthropic()
    t0 = time.time()
    try:
        response = client.with_options(timeout=LLM_TIMEOUT_S, max_retries=1).beta.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": "Evidence bundle:\n" + bundle.to_canonical_json()}],
            output_config={"effort": LLM_EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIError as exc:
        raise RenderError(f"API error: {type(exc).__name__}") from exc
    _log_usage(response, time.time() - t0)
    if response.stop_reason == "refusal":
        raise RenderError("model declined the request")
    if response.stop_reason == "max_tokens":
        raise RenderError("response truncated at max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RenderError("no text block in response")
    try:
        raw = json.loads(text)["claims"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RenderError("response was not the expected JSON") from exc
    kept, dropped = validate_claims(raw, bundle)
    if not kept:
        raise RenderError(f"all {dropped} claims failed validation")
    return EvidenceSummary(renderer="claude", claims=kept, disclaimer=bundle.disclaimer, dropped_claims=dropped)


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
