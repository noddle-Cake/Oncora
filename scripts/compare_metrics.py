#!/usr/bin/env python
"""Compare freshly produced metrics against committed reference values (spec 09, Req 2.2).

    uv run python scripts/compare_metrics.py            # compare
    uv run python scripts/compare_metrics.py --update    # re-record the references

Exits non-zero if any headline metric drifts beyond the tolerance, so a reproducibility run
fails loudly instead of quietly producing different numbers.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from oncora.config import REPORTS_EVAL

REFERENCE = REPORTS_EVAL / "reference_metrics.json"
TOLERANCE = 1e-6

# report stem -> {result name/split -> metrics that must reproduce}
TRACKED = {
    "target_model": {"model/compound": ["pr_auc_macro", "pr_auc_weighted", "roc_auc_macro", "top1", "top10"]},
    "sensitivity_model": {
        "model/compound": ["spearman_per_cell_line", "spearman_per_compound", "pr_auc_within_top10"],
        "model/scaffold": ["spearman_per_cell_line", "spearman_per_compound", "pr_auc_within_top10"],
    },
}


def collect() -> dict:
    out: dict[str, dict] = {}
    for stem, wanted in TRACKED.items():
        path = REPORTS_EVAL / f"{stem}.json"
        if not path.exists():
            continue
        results = {f"{r['name']}/{r['split']}": r["metrics"] for r in json.loads(path.read_text())["results"]}
        for key, metrics in wanted.items():
            if key in results:
                out[f"{stem}:{key}"] = {m: results[key].get(m) for m in metrics}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    fresh = collect()
    if not fresh:
        sys.exit("No evaluation reports found; train the models first.")
    if args.update or not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(fresh, indent=1, sort_keys=True))
        print(f"Recorded {sum(len(v) for v in fresh.values())} reference metrics -> {REFERENCE}")
        return

    reference = json.loads(REFERENCE.read_text())
    drift, checked = [], 0
    for group, metrics in reference.items():
        for name, expected in metrics.items():
            got = fresh.get(group, {}).get(name)
            checked += 1
            if got is None or expected is None:
                if got != expected:
                    drift.append(f"{group}.{name}: reference {expected}, got {got}")
                continue
            if not math.isclose(got, expected, rel_tol=args.tolerance, abs_tol=args.tolerance):
                drift.append(f"{group}.{name}: reference {expected:.6f}, got {got:.6f} "
                             f"(Δ {abs(got - expected):.2e})")
    if drift:
        print(f"{len(drift)} of {checked} metrics drifted beyond {args.tolerance:g}:")
        print("\n".join(f"  {d}" for d in drift))
        sys.exit(1)
    print(f"All {checked} headline metrics reproduce within {args.tolerance:g}.")


if __name__ == "__main__":
    main()
