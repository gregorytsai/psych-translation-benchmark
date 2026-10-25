#!/usr/bin/env python3
"""Normalize only mechanically recoverable ellipsis quotes for import.

The raw judge bundle is never overwritten.  If a judge returns a quote such
as ``left...right`` and both anchors occur in order in the canonical target,
the import copy expands it to that exact target substring.  No score,
severity, source evidence, or critical flag is changed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def expand_ellipsis(quote: str, target: str) -> str | None:
    if "..." not in quote:
        return None
    left, right = quote.split("...", 1)
    start = target.find(left)
    if start < 0:
        return None
    right_start = target.find(right, start + len(left))
    if right_start < 0:
        return None
    return target[start : right_start + len(right)]


def normalize(bundle: dict) -> tuple[dict, list[dict[str, str]]]:
    output = copy.deepcopy(bundle)
    repairs: list[dict[str, str]] = []
    observed_template_kwargs = []
    for event in output.get("raw_events", []):
        request = event.get("request", {}) if isinstance(event, dict) else {}
        kwargs = request.get("chat_template_kwargs") if isinstance(request, dict) else None
        if isinstance(kwargs, dict):
            observed_template_kwargs.append(kwargs)
    if observed_template_kwargs and all(kwargs == observed_template_kwargs[0] for kwargs in observed_template_kwargs):
        output.setdefault("settings", {})["chat_template_kwargs"] = copy.deepcopy(observed_template_kwargs[0])
        output.setdefault("import_normalization", {})["effective_request_template_kwargs"] = copy.deepcopy(observed_template_kwargs[0])
    for row in output.get("records", []):
        if row.get("status") != "ok" or row.get("track") != "pointwise":
            continue
        target = row.get("candidate", {}).get("raw_translation_zh_tw", "")
        for error in row.get("pointwise", {}).get("errors", []):
            quote = error.get("target_exact_quote")
            if not isinstance(quote, str) or quote == "" or quote in target:
                continue
            replacement = expand_ellipsis(quote, target)
            if replacement is None:
                continue
            error["original_target_exact_quote"] = quote
            error["target_exact_quote"] = replacement
            error["exact_quote"] = replacement
            error["target_start"] = target.find(replacement)
            error["target_end"] = error["target_start"] + len(replacement)
            error["offset_status"] = "computed_exact_quote_posthoc_ellipsis_expansion"
            repairs.append({"item_id": str(row.get("item_id")), "candidate_id": str(row.get("candidate_id")), "original": quote, "replacement": replacement})
    output["import_normalization"] = {
        "status": "MECHANICAL_EXACT_QUOTE_ONLY",
        "raw_bundle_preserved": True,
        "score_fields_changed": False,
        "repair_count": len(repairs),
        "repairs": repairs,
    }
    if observed_template_kwargs and all(kwargs == observed_template_kwargs[0] for kwargs in observed_template_kwargs):
        output["import_normalization"]["effective_request_template_kwargs"] = copy.deepcopy(observed_template_kwargs[0])
    return output, repairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output)
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing {target}")
    bundle = json.loads(source.read_text(encoding="utf-8"))
    normalized, repairs = normalize(bundle)
    target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"input": str(source), "output": str(target), "repair_count": len(repairs), "repairs": repairs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
