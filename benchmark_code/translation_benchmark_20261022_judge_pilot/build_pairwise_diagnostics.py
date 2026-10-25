#!/usr/bin/env python3
"""Build owner-only AB/BA diagnostics from blind judge bundles.

This is deliberately a diagnostic table, not a pairwise ranking.  It maps
orientation-local A/B winners back to the canonical blind candidate IDs and
marks position-sensitive swaps.  Shared-context bundles remain explicitly
labelled because their AB/BA consistency is not an independent fresh-chat
position-bias estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_pilot import _pair_winner, load_bundle


def _judge_metadata(metadata: Mapping[str, Any]) -> tuple[str, str, Any, Any]:
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), Mapping) else {}
    judge_id = str(judge.get("judge_id") or "")
    family = str(judge.get("family") or "")
    fresh = nested.get("pairwise_fresh_chat_independence")
    limitation = nested.get("pairwise_limitation")
    return judge_id, family, fresh, limitation


def _diagnose_group(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ab = rows.get("AB")
    ba = rows.get("BA")
    if ab is None or ba is None:
        status = "incomplete"
    else:
        ab_winner = _pair_winner(ab)
        ba_winner = _pair_winner(ba)
        if ab_winner == ba_winner:
            status = "consistent"
        elif ab_winner != "TIE" and ba_winner != "TIE":
            status = "position_sensitive"
        else:
            status = "tie_disagreement"
    return {
        "candidate_order_ab": f"{ab.get('candidate_a')} vs {ab.get('candidate_b')}" if ab else "",
        "candidate_order_ba": f"{ba.get('candidate_a')} vs {ba.get('candidate_b')}" if ba else "",
        "winner_ab_canonical": _pair_winner(ab) if ab else "",
        "winner_ba_canonical": _pair_winner(ba) if ba else "",
        "status": status,
    }


def build(input_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    judge_context: dict[str, dict[str, Any]] = {}
    for path in input_paths:
        metadata, rows = load_bundle(path)
        judge_id, family, fresh, limitation = _judge_metadata(metadata)
        for row in rows:
            if row.get("_track") != "pairwise":
                continue
            judge_id_row = str(row.get("_judge_id") or judge_id)
            key = (str(row.get("_item_id")), str(row.get("pair_id")), judge_id_row)
            judge_context[judge_id_row] = {
                "family": str(row.get("_family") or family),
                "fresh_chat_independence": fresh,
                "limitation": limitation,
            }
            orientation = str(row.get("orientation") or "").upper()
            if orientation not in {"AB", "BA"} or row.get("_status") != "ok":
                continue
            grouped.setdefault(key, {})[orientation] = row

    for (item_id, pair_id, judge_id), orientations in sorted(grouped.items()):
        diagnosed = _diagnose_group(orientations)
        context = judge_context.get(judge_id, {})
        detail_rows.append({
            "judge_id": judge_id,
            "judge_family": context.get("family", ""),
            "item_id": item_id,
            "pair_id": pair_id,
            "fresh_chat_independence": context.get("fresh_chat_independence"),
            "context_status": "shared_context_contaminated" if context.get("fresh_chat_independence") is False else "fresh_context_claimed" if context.get("fresh_chat_independence") is True else "not_recorded",
            **diagnosed,
            "pairwise_limitation": context.get("limitation", ""),
        })

    summary: dict[str, Any] = {
        "schema_version": "judge-pilot-pairwise-diagnostics-v1",
        "status": "descriptive_only",
        "note": "AB/BA winners are mapped to canonical blind candidate IDs. Shared-context runs are not independent fresh-chat position-bias tests; disconnected pairwise graphs do not support a global model ranking.",
        "recommendation": "HOLD Mistral and Qwen pairwise conclusions until a larger fresh-swap calibration; do not select Mistral solely because it is 128B. GPT's 3/3 consistency is shared-context evidence, not an independent position-bias test.",
        "judges": {},
    }
    by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        by_judge[row["judge_id"]].append(row)
    for judge_id, rows in sorted(by_judge.items()):
        counts = {status: sum(row["status"] == status for row in rows) for status in ("consistent", "position_sensitive", "tie_disagreement", "incomplete")}
        complete = counts["consistent"] + counts["position_sensitive"] + counts["tie_disagreement"]
        context = judge_context.get(judge_id, {})
        summary["judges"][judge_id] = {
            "family": context.get("family", ""),
            "complete_swaps": complete,
            "consistent": counts["consistent"],
            "position_sensitive": counts["position_sensitive"],
            "tie_disagreement": counts["tie_disagreement"],
            "incomplete": counts["incomplete"],
            "swap_consistency": counts["consistent"] / complete if complete else None,
            "fresh_chat_independence": context.get("fresh_chat_independence"),
            "context_status": "shared_context_contaminated" if context.get("fresh_chat_independence") is False else "fresh_context_claimed" if context.get("fresh_chat_independence") is True else "not_recorded",
            "recommendation": "HOLD independent position-bias conclusion; larger fresh-swap calibration needed" if context.get("fresh_chat_independence") is False or counts["position_sensitive"] else "descriptive only; larger fresh-swap calibration needed",
        }
    return detail_rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows, summary = build(args.input)
    fields = [
        "judge_id", "judge_family", "item_id", "pair_id", "fresh_chat_independence",
        "context_status", "candidate_order_ab", "candidate_order_ba",
        "winner_ab_canonical", "winner_ba_canonical", "status", "pairwise_limitation",
    ]
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"detail_rows": len(rows), "judges": summary["judges"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
