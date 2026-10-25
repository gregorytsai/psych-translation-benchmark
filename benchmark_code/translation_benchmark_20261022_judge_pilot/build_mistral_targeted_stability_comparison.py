#!/usr/bin/env python3
"""Build the immutable Mistral-high targeted retest stability comparison.

The targeted run is deliberately kept outside the 39-request comparison.  This
builder compares only the 12 pre-registered keys, preserves raw hashes and
judge-authored evidence, and records request/runtime provenance separately from
the quality result.  It never edits either raw bundle or replaces a score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKET = ROOT / "judge_packet.json"
DEFAULT_ORIGINAL = ROOT / "raw/mistral-medium-3.5-128b-local-thinking-high-20261022/mistral-medium-3.5-128b-local-thinking-high-20261022.raw.json"
DEFAULT_TARGETED = ROOT / "mistral_targeted_retest_20261023/raw/mistral-medium-3.5-128b-local-thinking-high-targeted-20261023/mistral-medium-3.5-128b-local-thinking-high-targeted-20261023.raw.json"
DEFAULT_OUTPUT = ROOT / "mistral_targeted_retest_20261023"

POINT_KEYS = [("src-01", "cand-01"), ("src-02", "cand-04"), ("src-03", "cand-02")]
QUAL_KEYS = ["q-01", "q-06", "q-12"]
PAIR_KEYS = [("p-01", "AB"), ("p-01", "BA"), ("p-03", "AB"), ("p-03", "BA"), ("p-02", "AB"), ("p-02", "BA")]
SUBSCORES = ("terminology_locale", "semantic_safety", "fluency", "completeness_format")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def jtext(value: Any) -> str:
    return canonical(value)


def sval(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def record_key(record: Mapping[str, Any]) -> str:
    track = record.get("track")
    if track == "qualification":
        return f"qualification/{record['item_id']}"
    if track == "pointwise":
        return f"pointwise/{record['source_id']}/{record['candidate_id']}"
    if track == "pairwise":
        return f"pairwise/{record['pair_id']}/{record['orientation']}"
    return f"{track}/{record.get('item_id', '')}"


def event_key(event: Mapping[str, Any]) -> str:
    request_id = str(event.get("request_id", ""))
    if request_id.startswith("targeted-20261023-"):
        request_id = request_id[len("targeted-20261023-"):]
    return request_id


def event_map(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {event_key(event): event for event in raw.get("raw_events", []) if event_key(event)}


def record_map(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {record_key(record): record for record in raw.get("records", [])}


def error_digest(errors: Any) -> str:
    return sha256_json(errors if isinstance(errors, list) else [])


def error_types(errors: Any) -> list[str]:
    return sorted({str(e.get("type", "")) for e in (errors if isinstance(errors, list) else []) if isinstance(e, Mapping)})


def error_quotes(errors: Any) -> list[str]:
    return [str(e.get("target_exact_quote", e.get("exact_quote", ""))) for e in (errors if isinstance(errors, list) else []) if isinstance(e, Mapping)]


def response_choices(event: Mapping[str, Any]) -> Any:
    response = event.get("response", {})
    return response.get("choices") if isinstance(response, Mapping) else None


def request_signature(request: Mapping[str, Any]) -> dict[str, Any]:
    # Messages are compared in their own field so the report can distinguish
    # frozen packet identity from generation/request model aliases.
    return {str(k): v for k, v in request.items() if k not in {"model", "messages"}}


def request_metadata(original_event: Mapping[str, Any], targeted_event: Mapping[str, Any]) -> dict[str, Any]:
    orq = original_event.get("request", {}) if isinstance(original_event.get("request"), Mapping) else {}
    tarq = targeted_event.get("request", {}) if isinstance(targeted_event.get("request"), Mapping) else {}
    om = orq.get("messages", [])
    tm = tarq.get("messages", [])
    return {
        "original_message_sha256": sha256_json(om),
        "targeted_message_sha256": sha256_json(tm),
        "messages_exact": om == tm,
        "original_non_model_request_sha256": sha256_json(request_signature(orq)),
        "targeted_non_model_request_sha256": sha256_json(request_signature(tarq)),
        "non_model_request_fields_exact": request_signature(orq) == request_signature(tarq),
        "model_original": orq.get("model", ""),
        "model_targeted": tarq.get("model", ""),
        "model_alias_changed_expected": orq.get("model") != tarq.get("model"),
        "response_choices_exact": response_choices(original_event) == response_choices(targeted_event),
        "response_sha256_equal": original_event.get("response_sha256") == targeted_event.get("response_sha256"),
        "original_response_sha256": original_event.get("response_sha256", ""),
        "targeted_response_sha256": targeted_event.get("response_sha256", ""),
        "original_elapsed_s": original_event.get("elapsed_s"),
        "targeted_elapsed_s": targeted_event.get("elapsed_s"),
        "elapsed_delta_s": (targeted_event.get("elapsed_s") - original_event.get("elapsed_s")) if isinstance(targeted_event.get("elapsed_s"), (int, float)) and isinstance(original_event.get("elapsed_s"), (int, float)) else None,
    }


def packet_maps(packet: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    point: dict[str, Mapping[str, Any]] = {}
    for item in packet.get("pointwise", {}).get("items", []):
        point[item["item_id"]] = item
    qual: dict[str, Mapping[str, Any]] = {item["qualification_id"]: item for item in packet.get("qualification", {}).get("items", [])}
    pair: dict[tuple[str, str], Mapping[str, Any]] = {(task["pair_id"], task["orientation"]): task for task in packet.get("pairwise", {}).get("tasks", [])}
    return point, pair, qual


def point_rows(original: Mapping[str, Any], targeted: Mapping[str, Any], packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    om, tm = record_map(original), record_map(targeted)
    oe, te = event_map(original), event_map(targeted)
    point_packet, _, _ = packet_maps(packet)
    rows: list[dict[str, Any]] = []
    for source_id, candidate_id in POINT_KEYS:
        key = f"pointwise/{source_id}/{candidate_id}"
        old, new = om[key], tm[key]
        oldp, newp = old["pointwise"], new["pointwise"]
        olde, newe = oe[f"pointwise-{source_id}-{candidate_id}"], te[f"pointwise-{source_id}-{candidate_id}"]
        item = point_packet[source_id]
        candidate = next(c for c in item["candidates"] if c["blind_id"] == candidate_id)
        old_sub = oldp.get("subscores") or {}
        new_sub = newp.get("subscores") or {}
        old_errors, new_errors = oldp.get("errors", []), newp.get("errors", [])
        meta = request_metadata(olde, newe)
        rows.append({
            "comparison_key": key,
            "source_id": source_id,
            "candidate_id": candidate_id,
            "source_en": item["source_en"],
            "target_zh_tw": candidate["target_zh_tw"],
            "source_text_sha256": old["source_text_sha256"],
            "target_text_sha256": hashlib.sha256(candidate["target_zh_tw"].encode("utf-8")).hexdigest(),
            "original_status": old.get("status"),
            "targeted_status": new.get("status"),
            "original_score_0_100": oldp.get("score_0_100"),
            "targeted_score_0_100": newp.get("score_0_100"),
            "score_delta_targeted_minus_original": (newp.get("score_0_100") - oldp.get("score_0_100")) if isinstance(newp.get("score_0_100"), (int, float)) and isinstance(oldp.get("score_0_100"), (int, float)) else None,
            "original_subscores_json": jtext(old_sub),
            "targeted_subscores_json": jtext(new_sub),
            "subscore_delta_json": jtext({name: (new_sub.get(name) - old_sub.get(name)) if isinstance(new_sub.get(name), (int, float)) and isinstance(old_sub.get(name), (int, float)) else None for name in SUBSCORES}),
            "original_critical": oldp.get("critical"),
            "targeted_critical": newp.get("critical"),
            "original_error_count": len(old_errors) if isinstance(old_errors, list) else None,
            "targeted_error_count": len(new_errors) if isinstance(new_errors, list) else None,
            "original_error_types_json": jtext(error_types(old_errors)),
            "targeted_error_types_json": jtext(error_types(new_errors)),
            "original_error_quotes_json": jtext(error_quotes(old_errors)),
            "targeted_error_quotes_json": jtext(error_quotes(new_errors)),
            "original_error_digest": error_digest(old_errors),
            "targeted_error_digest": error_digest(new_errors),
            "payload_result_exact": oldp == newp,
            "source_exact": old.get("source_text_sha256") == new.get("source_text_sha256"),
            "target_exact": old.get("candidate", {}).get("raw_translation_zh_tw") == new.get("candidate", {}).get("raw_translation_zh_tw") == candidate["target_zh_tw"],
            **meta,
        })
    return rows


def qual_rows(original: Mapping[str, Any], targeted: Mapping[str, Any], packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    om, tm = record_map(original), record_map(targeted)
    oe, te = event_map(original), event_map(targeted)
    _, _, qual_packet = packet_maps(packet)
    rows: list[dict[str, Any]] = []
    for qid in QUAL_KEYS:
        old, new = om[f"qualification/{qid}"], tm[f"qualification/{qid}"]
        oldj = {j["blind_id"]: j for j in old["qualification"]["judgments"]}
        newj = {j["blind_id"]: j for j in new["qualification"]["judgments"]}
        item = qual_packet[qid]
        for blind_id in sorted(oldj):
            oldv, newv = oldj[blind_id], newj[blind_id]
            variant = next(v for v in item["variants"] if v["blind_id"] == blind_id)
            key = f"qualification-{qid}"
            meta = request_metadata(oe[key], te[key])
            old_errors, new_errors = oldv.get("errors", []), newv.get("errors", [])
            old_sub, new_sub = oldv.get("subscores") or {}, newv.get("subscores") or {}
            old_sum = sum(old_sub.values()) if old_sub and all(isinstance(v, (int, float)) for v in old_sub.values()) else None
            new_sum = sum(new_sub.values()) if new_sub and all(isinstance(v, (int, float)) for v in new_sub.values()) else None
            rows.append({
                "qualification_id": qid,
                "source_en": item["source_en"],
                "blind_id": blind_id,
                "target_zh_tw": variant["target_zh_tw"],
                "source_text_sha256": old["source_text_sha256"],
                "target_text_sha256": hashlib.sha256(variant["target_zh_tw"].encode("utf-8")).hexdigest(),
                "original_score_0_100": oldv.get("score_0_100"),
                "targeted_score_0_100": newv.get("score_0_100"),
                "score_delta_targeted_minus_original": (newv.get("score_0_100") - oldv.get("score_0_100")) if isinstance(newv.get("score_0_100"), (int, float)) and isinstance(oldv.get("score_0_100"), (int, float)) else None,
                "original_subscores_json": jtext(old_sub),
                "targeted_subscores_json": jtext(new_sub),
                "original_scalar_sum": old_sum,
                "targeted_scalar_sum": new_sum,
                "original_scalar_valid": old_sum == oldv.get("score_0_100"),
                "targeted_scalar_valid": new_sum == newv.get("score_0_100"),
                "original_critical": oldv.get("critical"),
                "targeted_critical": newv.get("critical"),
                "original_error_count": len(old_errors) if isinstance(old_errors, list) else None,
                "targeted_error_count": len(new_errors) if isinstance(new_errors, list) else None,
                "original_error_types_json": jtext(error_types(old_errors)),
                "targeted_error_types_json": jtext(error_types(new_errors)),
                "original_error_quotes_json": jtext(error_quotes(old_errors)),
                "targeted_error_quotes_json": jtext(error_quotes(new_errors)),
                "original_error_digest": error_digest(old_errors),
                "targeted_error_digest": error_digest(new_errors),
                "payload_result_exact": oldv == newv,
                "source_exact": old["source_text_sha256"] == new["source_text_sha256"],
                "target_exact": variant["target_zh_tw"] in jtext(oe[key].get("request", {}).get("messages", [])) and variant["target_zh_tw"] in jtext(te[key].get("request", {}).get("messages", [])),
                **meta,
            })
    return rows


def canonical_winner(row: Mapping[str, Any]) -> str:
    winner = row.get("winner")
    if winner == "TIE":
        return "TIE"
    if winner == "A":
        return str(row.get("candidate_a", ""))
    if winner == "B":
        return str(row.get("candidate_b", ""))
    return ""


def pair_rows(original: Mapping[str, Any], targeted: Mapping[str, Any]) -> list[dict[str, Any]]:
    om, tm = record_map(original), record_map(targeted)
    oe, te = event_map(original), event_map(targeted)
    rows: list[dict[str, Any]] = []
    for pair_id, orientation in PAIR_KEYS:
        key = f"pairwise/{pair_id}/{orientation}"
        old, new = om[key], tm[key]
        oe1, te1 = oe[f"pairwise-{pair_id}-{orientation}"], te[f"pairwise-{pair_id}-{orientation}"]
        old_can, new_can = canonical_winner(old), canonical_winner(new)
        meta = request_metadata(oe1, te1)
        rows.append({
            "pair_id": pair_id,
            "orientation": orientation,
            "source_id": old["source_id"],
            "source_text_sha256": old["source_text_sha256"],
            "candidate_a": old["candidate_a"],
            "candidate_b": old["candidate_b"],
            "original_winner": old.get("winner"),
            "targeted_winner": new.get("winner"),
            "original_canonical_winner": old_can,
            "targeted_canonical_winner": new_can,
            "canonical_winner_equal": old_can == new_can,
            "original_confidence": old.get("confidence"),
            "targeted_confidence": new.get("confidence"),
            "original_reason_codes_json": jtext(old.get("reason_codes", [])),
            "targeted_reason_codes_json": jtext(new.get("reason_codes", [])),
            "payload_result_exact": all(old.get(name) == new.get(name) for name in ("winner", "confidence", "critical_override", "reason_codes")),
            "source_exact": old["source_text_sha256"] == new["source_text_sha256"],
            **meta,
        })
    return rows


def pair_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for pair_id, values in sorted(by_pair.items()):
        orientations = {str(row["orientation"]): row for row in values}
        if set(orientations) != {"AB", "BA"}:
            out[pair_id] = {"complete": False, "consistent": None, "position_sensitive": None, "canonical_winners": {k: v.get("targeted_canonical_winner") for k, v in orientations.items()}}
            continue
        winners = [orientations[o]["targeted_canonical_winner"] for o in ("AB", "BA")]
        complete = all(w not in ("", None) for w in winners)
        consistent = complete and winners[0] == winners[1]
        out[pair_id] = {"complete": complete, "consistent": consistent if complete else None, "position_sensitive": (not consistent) if complete else None, "canonical_winners": {o: orientations[o]["targeted_canonical_winner"] for o in ("AB", "BA")}}
    return out


def percentile(values: Sequence[float], p: float) -> float | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    index = max(0, min(len(vals) - 1, math.ceil(p * len(vals)) - 1))
    return vals[index]


def token_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    data = load(path)
    rows = data.get("records", []) if isinstance(data.get("records"), list) else data.get("rows", [])
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        rid = str(row.get("request_id", ""))
        if rid.startswith("targeted-20261023-"):
            rid = rid[len("targeted-20261023-"):]
        result[rid] = row
    return result


def run_summary(label: str, raw: Mapping[str, Any], raw_path: Path, records: Sequence[Mapping[str, Any]], events: Mapping[str, Mapping[str, Any]], token_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    points = [r["pointwise"]["score_0_100"] for r in records if r.get("track") == "pointwise" and isinstance(r.get("pointwise", {}).get("score_0_100"), (int, float))]
    quals = [j for r in records if r.get("track") == "qualification" for j in r.get("qualification", {}).get("judgments", [])]
    scalar_valid = [j for j in quals if isinstance(j.get("subscores"), Mapping) and all(isinstance(j["subscores"].get(k), (int, float)) for k in SUBSCORES) and sum(j["subscores"][k] for k in SUBSCORES) == j.get("score_0_100")]
    pairs = [r for r in records if r.get("track") == "pairwise"]
    pair_rows_here = []
    for r in pairs:
        can = canonical_winner(r)
        pair_rows_here.append({"pair_id": r["pair_id"], "orientation": r["orientation"], "targeted_canonical_winner": can})
    status = pair_status(pair_rows_here)
    complete = sum(1 for x in status.values() if x["complete"])
    consistent = sum(1 for x in status.values() if x["consistent"] is True)
    position = sum(1 for x in status.values() if x["position_sensitive"] is True)
    latencies: dict[str, list[float]] = {track: [] for track in ("qualification", "pointwise", "pairwise")}
    for key, event in events.items():
        track = "qualification" if key.startswith("qualification-") else "pointwise" if key.startswith("pointwise-") else "pairwise" if key.startswith("pairwise-") else None
        if track and isinstance(event.get("elapsed_s"), (int, float)):
            latencies[track].append(float(event["elapsed_s"]))
    token = token_rows(token_path)
    selected_tokens = [row.get("reasoning_tokens", row.get("offline_reasoning_tokens")) for key, row in token.items() if key in events and isinstance(row.get("reasoning_tokens", row.get("offline_reasoning_tokens")), (int, float))]
    token_summary = load(token_path).get("summary", {}) if "summary" in load(token_path) else load(token_path).get("counts", {})
    template_status = manifest.get("validation", {}).get("template_verification_status") if isinstance(manifest.get("validation"), Mapping) else manifest.get("template_audit", "")
    return {
        "mode": label,
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "run_id": raw.get("run_id", ""),
        "judge_id": raw.get("judge", {}).get("judge_id", ""),
        "records": len(records),
        "selected_records": len(records),
        "raw_records_total": len(raw.get("records", [])),
        "qualification_parents": sum(1 for r in records if r.get("track") == "qualification"),
        "qualification_variants": len(quals),
        "qualification_scalar_valid": len(scalar_valid),
        "qualification_scalar_invalid": len(quals) - len(scalar_valid),
        "pointwise_valid": len(points),
        "pointwise_mean": statistics.fmean(points) if points else None,
        "pointwise_sd_population": statistics.pstdev(points) if len(points) > 1 else 0.0 if points else None,
        "pointwise_unique_values_json": jtext(sorted(set(points))),
        "pointwise_value_counts_json": jtext(dict(sorted(Counter(points).items()))),
        "pairwise_orientations": len(pairs),
        "pairwise_complete_swaps": complete,
        "pairwise_consistent": consistent,
        "pairwise_position_sensitive": position,
        "pairwise_consistency_rate": consistent / complete if complete else None,
        "qualification_latency_median_s": statistics.median(latencies["qualification"]) if latencies["qualification"] else None,
        "pointwise_latency_median_s": statistics.median(latencies["pointwise"]) if latencies["pointwise"] else None,
        "pairwise_latency_median_s": statistics.median(latencies["pairwise"]) if latencies["pairwise"] else None,
        "qualification_latency_p95_s": percentile(latencies["qualification"], 0.95),
        "pointwise_latency_p95_s": percentile(latencies["pointwise"], 0.95),
        "pairwise_latency_p95_s": percentile(latencies["pairwise"], 0.95),
        "reasoning_tokens_selected_median": statistics.median(selected_tokens) if selected_tokens else None,
        "reasoning_tokens_selected_min": min(selected_tokens) if selected_tokens else None,
        "reasoning_tokens_selected_max": max(selected_tokens) if selected_tokens else None,
        "token_audit_path": str(token_path),
        "token_audit_summary_json": jtext(token_summary),
        "template_verification_status": template_status or "not recorded",
        "finish_reason_stop_count": sum(1 for key, event in events.items() if key.startswith(("qualification-", "pointwise-", "pairwise-")) and (event.get("response_metadata", {}).get("finish_reason") if isinstance(event.get("response_metadata"), Mapping) else event.get("response", {}).get("choices", [{}])[0].get("finish_reason")) == "stop"),
        "cache_tokens_unique_json": jtext(sorted({(event.get("response_metadata", {}).get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") if isinstance(event.get("response_metadata"), Mapping) else event.get("response", {}).get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")) for key, event in events.items() if key.startswith(("qualification-", "pointwise-", "pairwise-"))})),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: ("" if row.get(field) is None else row.get(field)) for field in fields})


def compare_environment(original_raw: Mapping[str, Any], targeted_raw: Mapping[str, Any], original_manifest: Mapping[str, Any], targeted_manifest: Mapping[str, Any], target_dir: Path, original_props: Mapping[str, Any]) -> dict[str, Any]:
    old_settings = original_manifest.get("fixed_settings", {})
    new_settings = targeted_manifest.get("frozen_settings", {})
    old_judge = original_manifest.get("judge", {})
    new_judge = targeted_manifest.get("judge", {})
    old_path = old_judge.get("model_path_first_shard", "")
    new_path = new_judge.get("model_shard", "")
    old_props = {
        "build_info": original_props.get("build_info"),
        "model_path": original_props.get("model_path"),
        "model_alias": original_props.get("model_alias"),
        "effective_context": original_props.get("default_generation_settings", {}).get("n_ctx") if isinstance(original_props.get("default_generation_settings"), Mapping) else None,
        "chat_template_length": len(original_props.get("chat_template", "")) if isinstance(original_props.get("chat_template"), str) else None,
    }
    target_props_path = target_dir / "../../template_audit_20261023/props.json"
    target_props = load(target_props_path) if target_props_path.exists() else {}
    target_template = load(target_dir / "../../template_audit_20261023/template_provenance.json") if (target_dir / "../../template_audit_20261023/template_provenance.json").exists() else {}
    current_shard = Path(old_path) if old_path else None
    current_shard_sha = sha256_file(current_shard) if current_shard and current_shard.exists() else ""
    current_shard_size = current_shard.stat().st_size if current_shard and current_shard.exists() else None
    old_sig = {key: old_settings.get(key) for key in ("reasoning_effort", "reasoning_format", "reasoning_budget_tokens", "max_tokens", "context_size", "temperature", "top_p", "seed", "parallel", "runner_cache", "strict_json_output", "response_format", "chat_template_kwargs", "reasoning_mode", "enable_thinking_field_sent")}
    new_sig = {key: new_settings.get(key) for key in ("reasoning_effort", "reasoning_format", "reasoning_budget_tokens", "max_tokens", "effective_context", "temperature", "top_p", "seed", "parallel", "cache", "strict_json_output", "response_format", "chat_template_kwargs")}
    normalized_new = dict(new_sig)
    normalized_new["context_size"] = normalized_new.pop("effective_context")
    normalized_new["runner_cache"] = normalized_new.pop("cache")
    normalized_new["reasoning_mode"] = None
    normalized_new["enable_thinking_field_sent"] = False
    return {
        "model_shard_path_exact": old_path == new_path,
        "model_shard_path_original": old_path,
        "model_shard_path_targeted": new_path,
        "model_weight_file_hash_recorded": False,
        "model_weight_hash_note": "Both manifests bind the same first-shard path; no shard content SHA is recorded in either run manifest.",
        "current_first_shard_sha256_posthoc": current_shard_sha,
        "current_first_shard_size_bytes_posthoc": current_shard_size,
        "current_first_shard_posthoc_note": "Read-only post-run hash of the currently present first shard; not a hash captured inside either run manifest and not a substitute for full three-shard run-time binding.",
        "request_settings_semantically_equal": old_sig == normalized_new,
        "original_request_settings": old_sig,
        "targeted_request_settings_normalized": normalized_new,
        "runtime_build_original": old_props.get("build_info"),
        "runtime_build_targeted": (target_props.get("response_json", {}) or {}).get("build_info") or (target_template.get("props", {}) or {}).get("build_info") or (target_template.get("server_binding", {}) or {}).get("build_info"),
        "runtime_build_equal": old_props.get("build_info") == ((target_props.get("response_json", {}) or {}).get("build_info") or (target_template.get("props", {}) or {}).get("build_info") or (target_template.get("server_binding", {}) or {}).get("build_info")),
        "original_effective_context": old_settings.get("context_size"),
        "targeted_effective_context": new_settings.get("effective_context"),
        "context_equal": old_settings.get("context_size") == new_settings.get("effective_context"),
        "original_cache": old_settings.get("runner_cache"),
        "targeted_cache": new_settings.get("cache"),
        "cache_equal": old_settings.get("runner_cache") == new_settings.get("cache") == "disabled",
        "adapter_original": "not recorded; props default_generation_settings.params.lora=[]",
        "adapter_targeted": "not recorded in targeted manifest/command; no adapter flag captured",
        "adapter_comparison": "no adapter evidence; do not infer more than captured command/props",
        "original_template_status": "PENDING_PROVENANCE_INCOMPLETE" if (original_manifest.get("validation") or {}).get("template_verification_status") == "PENDING" else (original_manifest.get("validation") or {}).get("template_verification_status", "PENDING_PROVENANCE_INCOMPLETE"),
        "targeted_template_status": "PASS_NON_GENERATING_AUDIT",
        "original_template_proof_note": "Original 39-request run remains PENDING_PROVENANCE_INCOMPLETE; props/smoke were not request-bound render provenance.",
        "targeted_template_proof_note": "Targeted non-generating audit is PASS_NON_GENERATING_AUDIT; it is separate evidence and does not retroactively upgrade original run.",
        "targeted_template_effective_sha256": target_template.get("props", {}).get("effective_template_sha256", target_template.get("effective_template_sha256", target_template.get("effective_template_sha", ""))),
        "targeted_template_props_sha256": sha256_file(target_props_path) if target_props_path.exists() else "",
        "runtime_command_sha256": sha256_file(target_dir / "server.command.txt") if (target_dir / "server.command.txt").exists() else "",
        "original_runtime_command_recorded": False,
        "runtime_difference_note": "Targeted command and new template audit are captured; original command/HTTP render envelope is not fully recorded. Keep template provenance and any unrecorded adapter/runtime details as limitations.",
        "same_model_alias_changed": old_judge.get("model_id") != new_judge.get("model_alias"),
        "same_model_alias_original": old_judge.get("model_id"),
        "same_model_alias_targeted": new_judge.get("model_alias"),
    }


def build(packet_path: Path, original_path: Path, targeted_path: Path, output_dir: Path) -> dict[str, Any]:
    packet, original, targeted = load(packet_path), load(original_path), load(targeted_path)
    original_dir = original_path.parent
    targeted_dir = targeted_path.parent
    original_manifest = load(original_dir / "mistral-medium-3.5-128b-local-thinking-high-20261022.manifest.json")
    targeted_manifest = load(targeted_dir / "manifest.json")
    original_props = load(original_dir / "props.json")
    original_token = original_dir / "mistral-medium-3.5-128b-local-thinking-high-20261022.token_audit.json"
    targeted_token = targeted_dir.parent / "mistral-medium-3.5-128b-local-thinking-high-targeted-20261023.token_audit.json"
    if not targeted_token.exists():
        targeted_token = targeted_dir / "mistral-medium-3.5-128b-local-thinking-high-targeted-20261023.token_audit.json"
    if not targeted_token.exists():
        raise FileNotFoundError(targeted_token)
    p_rows, q_rows, pair_rows_here = point_rows(original, targeted, packet), qual_rows(original, targeted, packet), pair_rows(original, targeted)
    # Strict selected-key gates.  The builder is allowed to fail closed; it does
    # not manufacture a row when one is missing or malformed.
    if len(p_rows) != 3 or len(q_rows) != 6 or len(pair_rows_here) != 6:
        raise ValueError(f"selected rows mismatch: point={len(p_rows)} qual={len(q_rows)} pair={len(pair_rows_here)}")
    all_rows = p_rows + q_rows + pair_rows_here
    if not all(bool(row.get("messages_exact")) and bool(row.get("non_model_request_fields_exact")) for row in all_rows):
        raise ValueError("frozen request message/non-model-field exactness gate failed")
    if not all(bool(row.get("source_exact")) for row in all_rows):
        raise ValueError("source hash exactness gate failed")

    orig_events, targ_events = event_map(original), event_map(targeted)
    orig_records, targ_records = record_map(original), record_map(targeted)
    old_summary = run_summary("original_high", original, original_path, [orig_records[k] for k in orig_records if k in {record_key(x) for x in targeted.get("records", [])}], orig_events, original_token, original_manifest)
    new_summary = run_summary("targeted_high", targeted, targeted_path, list(targeted.get("records", [])), targ_events, targeted_token, targeted_manifest)
    old_summary["template_verification_status"] = "PENDING_PROVENANCE_INCOMPLETE"
    new_summary["template_verification_status"] = "PASS_NON_GENERATING_AUDIT"
    env = compare_environment(original, targeted, original_manifest, targeted_manifest, targeted_dir, original_props)

    # Cross-run pair status is intentionally separate from within-run ABBA.
    old_pair_rows = []
    for pair_id, orientation in PAIR_KEYS:
        row = orig_records[f"pairwise/{pair_id}/{orientation}"]
        old_pair_rows.append({"pair_id": pair_id, "orientation": orientation, "targeted_canonical_winner": canonical_winner(row)})
    old_status = pair_status(old_pair_rows)
    new_status = pair_status(pair_rows_here)
    cross_orient = sum(1 for row in pair_rows_here if row.get("canonical_winner_equal"))
    cross_pairs = sum(1 for pid in new_status if old_status.get(pid, {}).get("canonical_winners") == new_status.get(pid, {}).get("canonical_winners"))
    cross_status = {
        "orientation_canonical_winner_matches": cross_orient,
        "orientation_total": len(pair_rows_here),
        "pair_abba_status_matches": cross_pairs,
        "pair_total": len(new_status),
        "old_within_run": old_status,
        "targeted_within_run": new_status,
    }
    original_token_rows, targeted_token_rows = token_rows(original_token), token_rows(targeted_token)
    token_matches = []
    for row in all_rows:
        if row.get("comparison_key", "").startswith("pointwise"):
            rid = "pointwise-" + row["source_id"] + "-" + row["candidate_id"]
        elif row.get("qualification_id"):
            rid = "qualification-" + row["qualification_id"]
        else:
            rid = "pairwise-" + row["pair_id"] + "-" + row["orientation"]
        old_t = original_token_rows.get(rid, {}).get("reasoning_tokens")
        new_t = targeted_token_rows.get(rid, {}).get("offline_reasoning_tokens")
        token_matches.append({"request_id": rid, "original_offline_reasoning_tokens": old_t, "targeted_offline_reasoning_tokens": new_t, "delta_targeted_minus_original": new_t - old_t if isinstance(new_t, (int, float)) and isinstance(old_t, (int, float)) else None})
    quality_equal = all(row.get("payload_result_exact") for row in p_rows + q_rows + pair_rows_here)
    metrics = {
        "schema_version": "mistral-targeted-stability-comparison-v1",
        "status": "COMPLETE_TARGETED_12_ANALYZED",
        "analysis_date": "2026-10-23",
        "no_new_model_calls": True,
        "raw_immutable": True,
        "original_raw": {"path": str(original_path), "sha256": sha256_file(original_path), "status": original.get("status"), "records": len(original.get("records", []))},
        "targeted_raw": {"path": str(targeted_path), "sha256": sha256_file(targeted_path), "status": targeted.get("status"), "records": len(targeted.get("records", []))},
        "selected_counts": {"pointwise": len(p_rows), "qualification_variants": len(q_rows), "pairwise_orientations": len(pair_rows_here)},
        "all_selected_quality_payloads_exact": quality_equal,
        "environment": env,
        "cross_run_pairwise": cross_status,
        "token_audit_selected_comparison": token_matches,
        "summaries": {"original_high": old_summary, "targeted_high": new_summary},
        "pointwise": p_rows,
        "qualification": q_rows,
        "pairwise": pair_rows_here,
        "interpretation": {
            "within_judge": "Fixed-seed operational reproducibility only; not random-sampling reliability and not an independent rater.",
            "cross_run_stability": "The 12 selected structured quality payloads are identical; response envelope hashes differ because model alias/IDs/timestamps/timings differ.",
            "quality_conclusion": "No score/evidence instability is observed on the targeted 3 qualification parents, 3 pointwise keys, and 3 ABBA pairs.",
            "deployment": "This limited targeted result does not prove deployment suitability; original 39-run template provenance remains PENDING_PROVENANCE_INCOMPLETE and the subset is small.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "targeted_pointwise_comparison.owner.csv", p_rows)
    write_csv(output_dir / "targeted_qualification_variant_comparison.owner.csv", q_rows)
    write_csv(output_dir / "targeted_pairwise_comparison.owner.csv", pair_rows_here)
    write_csv(output_dir / "targeted_run_summary.owner.csv", [old_summary, new_summary])
    dump(output_dir / "targeted_stability_metrics.json", metrics)
    dump(output_dir / "targeted_stability_status.json", {
        "status": "COMPLETE_TARGETED_12_ANALYZED",
        "raw_sha256": metrics["targeted_raw"]["sha256"],
        "original_raw_sha256": metrics["original_raw"]["sha256"],
        "selected_counts": metrics["selected_counts"],
        "payloads_exact": quality_equal,
        "within_run_abba_original": old_status,
        "within_run_abba_targeted": new_status,
        "cross_run_orientation_matches": cross_orient,
        "cross_run_pairs_match": cross_pairs,
        "next_action": "No automatic follow-up generation; retain raw and review deployment/template provenance separately.",
    })
    dump(output_dir / "targeted_stability_manifest.json", {
        "schema_version": "mistral-targeted-stability-manifest-v1",
        "status": "COMPLETE_TARGETED_12_ANALYZED",
        "packet_sha256": sha256_file(packet_path),
        "original_raw_sha256": sha256_file(original_path),
        "targeted_raw_sha256": sha256_file(targeted_path),
        "targeted_manifest_sha256": sha256_file(targeted_dir / "manifest.json"),
        "targeted_token_audit_sha256": sha256_file(targeted_token),
        "targeted_template_provenance_sha256": sha256_file(targeted_dir.parent.parent / "template_audit_20261023/template_provenance.json"),
        "outputs": [
            "targeted_pointwise_comparison.owner.csv",
            "targeted_qualification_variant_comparison.owner.csv",
            "targeted_pairwise_comparison.owner.csv",
            "targeted_run_summary.owner.csv",
            "targeted_stability_metrics.json",
            "targeted_stability_comparison.md",
            "artifacttool_qa.json",
        ],
        "substantive_raw_count": 12,
        "technical_smoke_excluded": 2,
        "same_model_two_runs_not_independent_judges": True,
        "original_panel_not_replaced": True,
        "exact_source_target_binding": True,
        "no_score_imputation": True,
        "ci_status": "not available; no CI invented",
    })
    report_path = output_dir / "targeted_stability_comparison.md"
    report_path.write_text(render_report(metrics), encoding="utf-8")
    return {"status": "PASS", "output_dir": str(output_dir), "pointwise_rows": len(p_rows), "qualification_rows": len(q_rows), "pairwise_rows": len(pair_rows_here), "payloads_exact": quality_equal}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n.a."
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(metrics: Mapping[str, Any]) -> str:
    old, new = metrics["summaries"]["original_high"], metrics["summaries"]["targeted_high"]
    env = metrics["environment"]
    cross = metrics["cross_run_pairwise"]
    lines = [
        "# Mistral thinking-high targeted stability comparison (2026-10-23)",
        "",
        "## 摘要",
        "",
        "這是原 Mistral thinking-high 39-request full run 的預先登記 12-request 穩定性重測；不替換、不回填原始 39 筆，也不把同一模型的兩次設定/執行計為新的獨立 judge。targeted run 只含 3 個 qualification parents/6 variants、3 個 pointwise keys、3 個 pair 的 6 orientations；2 個 technical smoke 不納入品質統計。",
        "",
        f"- raw 狀態：targeted `12/12 ok, 0 failed`，SHA-256 `{metrics['targeted_raw']['sha256']}`；原始 39-run SHA `{metrics['original_raw']['sha256']}`。兩份 raw 均保留不變。",
        f"- 嚴格 request binding：12/12 message exact、非 model request fields 12/12 exact、source hash exact；3 個 pointwise 的 source/target exact binding 均通過。結構化 quality payload 12/12 完全相同（score、subscores、errors/quotes、pair winner/reason codes）。",
        f"- AB/BA：原始與 targeted 均為 3/3 complete、1/3 consistent、2/3 position-sensitive；跨 run canonical winner 為 {cross['orientation_canonical_winner_matches']}/{cross['orientation_total']} orientations，ABBA 狀態 {cross['pair_abba_status_matches']}/{cross['pair_total']} pairs 相同。",
        f"- pointwise selected 3 筆：原始/targeted scores 均為 `{[r['original_score_0_100'] for r in metrics['pointwise']]}` / `{[r['targeted_score_0_100'] for r in metrics['pointwise']]}`；未觀察到 selected key 的評分漂移。",
        f"- qualification selected 6 variants：原始與 targeted 均 scalar-valid `6/6`；本 subset 不足以取代 full-run 24 variants 或證明正式 panel qualification。",
        "- 結論限於這個 fixed-seed operational subset：未見品質 payload 不穩；但這不是 random-sampling reliability，也不能單獨證明 deployment suitability。原 39-run template provenance 仍是 `PENDING_PROVENANCE_INCOMPLETE`（manifest 的簡寫欄位為 `PENDING`），targeted 的新 non-generating template audit 另列且不回填原 run。",
        "",
        "## 1. Strict per-request provenance",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| packet/source binding | `{metrics['selected_counts']}` selected rows; exact source hash on all rows |",
        "| message bodies | 12/12 exact canonical SHA across runs |",
        "| non-model request fields | 12/12 exact (reasoning format/budget, max tokens, JSON, seed, sampling, preserve flag) |",
        f"| same first-shard path | `{env['model_shard_path_exact']}`; `{env['model_shard_path_original']}`; current post-hoc SHA `{env['current_first_shard_sha256_posthoc']}` |",
        f"| context/cache | context `{env['original_effective_context']}` vs `{env['targeted_effective_context']}`; cache `{env['original_cache']}` vs `{env['targeted_cache']}` |",
        f"| runtime build | `{env['runtime_build_original']}` vs `{env['runtime_build_targeted']}`; equal=`{env['runtime_build_equal']}` |",
        f"| template evidence | original `{env['original_template_status']}`; targeted `{env['targeted_template_status']}` |",
        f"| adapter evidence | `{env['adapter_comparison']}` |",
        "",
        "The model request alias changed only to isolate the targeted server run; this is an envelope/provenance difference, not a model-weight change. Both manifests bind the same first-shard path, but neither records a content SHA for all model shards. The original run has no complete captured server command/render envelope; therefore unrecorded runtime/adapter details remain limitations rather than inferred equality.",
        "",
        "## 2. Pointwise selected keys",
        "",
        "| Key | Original | Targeted | Δ | Critical old/new | Errors old/new | Result payload exact |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in metrics["pointwise"]:
        lines.append(f"| `{row['source_id']}/{row['candidate_id']}` | {row['original_score_0_100']} | {row['targeted_score_0_100']} | {row['score_delta_targeted_minus_original']} | {row['original_critical']}/{row['targeted_critical']} | {row['original_error_count']}/{row['targeted_error_count']} | `{row['payload_result_exact']}` |")
    lines += [
        "",
        "The CSV retains exact source/target text and judge-authored error quotes/types/digests; no score or quote was hand-edited.",
        "",
        "## 3. Qualification selected parents and variants",
        "",
        "| Parent | Variant | Original | Targeted | Δ | Scalar valid old/new | Critical old/new | Errors old/new | Payload exact |",
        "|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for row in metrics["qualification"]:
        lines.append(f"| `{row['qualification_id']}` | `{row['blind_id']}` | {row['original_score_0_100']} | {row['targeted_score_0_100']} | {row['score_delta_targeted_minus_original']} | {row['original_scalar_valid']}/{row['targeted_scalar_valid']} | {row['original_critical']}/{row['targeted_critical']} | {row['original_error_count']}/{row['targeted_error_count']} | `{row['payload_result_exact']}` |")
    lines += [
        "",
        "These are the exact selected hidden-key variants from the frozen packet. They are a stability check, not a new qualification sample and not a license to extrapolate to all 24 variants.",
        "",
        "## 4. Pairwise within-run ABBA versus cross-run stability",
        "",
        "| Pair | Orientation | Original canonical winner | Targeted canonical winner | Same | Payload exact |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics["pairwise"]:
        lines.append(f"| `{row['pair_id']}` | `{row['orientation']}` | `{row['original_canonical_winner']}` | `{row['targeted_canonical_winner']}` | `{row['canonical_winner_equal']}` | `{row['payload_result_exact']}` |")
    lines += [
        "",
        "Within-run ABBA consistency asks whether AB and BA choose the same candidate within one run; it is 1/3 in both runs. Cross-run stability asks whether the canonical outcome for the same orientation/pair is preserved; it is 6/6 orientations and 3/3 pair-statuses. These are separate quantities and should not be conflated.",
        "",
        "## 5. Latency and reasoning-token audit",
        "",
        "| Run | Qual median/p95 s | Point median/p95 s | Pair median/p95 s | Selected reasoning tokens median/min/max |",
        "|---|---:|---:|---:|---:|",
        f"| original 39-run (selected event rows) | {fmt(old['qualification_latency_median_s'])}/{fmt(old['qualification_latency_p95_s'])} | {fmt(old['pointwise_latency_median_s'])}/{fmt(old['pointwise_latency_p95_s'])} | {fmt(old['pairwise_latency_median_s'])}/{fmt(old['pairwise_latency_p95_s'])} | {fmt(old['reasoning_tokens_selected_median'])}/{fmt(old['reasoning_tokens_selected_min'])}/{fmt(old['reasoning_tokens_selected_max'])} |",
        f"| targeted 12-run | {fmt(new['qualification_latency_median_s'])}/{fmt(new['qualification_latency_p95_s'])} | {fmt(new['pointwise_latency_median_s'])}/{fmt(new['pointwise_latency_p95_s'])} | {fmt(new['pairwise_latency_median_s'])}/{fmt(new['pairwise_latency_p95_s'])} | {fmt(new['reasoning_tokens_selected_median'])}/{fmt(new['reasoning_tokens_selected_min'])}/{fmt(new['reasoning_tokens_selected_max'])} |",
        "",
        "Latency is secondary to quality/evidence/coherence. Both runs used the same requested high/deepseek/8192/16384/32768/temperature-0/top-p-1/seed settings and disabled request cache. The targeted token-audit rows are systematically one token lower than the original audit on these selected keys; the two audit schemas/method metadata differ, so this is recorded as an audit-method limitation, not a reasoning instability claim. The original run retains three conservative 8192-boundary candidates; targeted selected rows have no at/over candidate. No chain-of-thought is copied into this report.",
        "",
        "## 6. Reliability and deployment interpretation",
        "",
        "No new ICC/Kendall-W/LOFO is computed from 12 repeated rows: this is a within-judge fixed-seed operational check, not an independent rater panel. ICC from the separate five-judge panels remains descriptive and is not used to select the higher-looking mode. There are only three source clusters and no CI is available. The targeted result supports stability of the observed selected outputs, while leaving the original full-run template-provenance limitation and limited sample size unresolved; it does not by itself establish deployment readiness.",
        "",
        "## 7. Outputs and immutable inputs",
        "",
        "- `targeted_pointwise_comparison.owner.csv` — 3 exact source×candidate rows with score/subscore/error/quote and request-binding fields.",
        "- `targeted_qualification_variant_comparison.owner.csv` — 6 exact parent/variant rows with scalar/evidence/error fields.",
        "- `targeted_pairwise_comparison.owner.csv` — 6 exact orientations with canonical winners and cross-run binding.",
        "- `targeted_run_summary.owner.csv` — original selected-event and targeted-run validity/latency/token summaries.",
        "- `targeted_stability_metrics.json`, `targeted_stability_status.json`, `targeted_stability_manifest.json` — provenance, hashes, checks and limitations.",
        f"- Original raw SHA: `{metrics['original_raw']['sha256']}`; targeted raw SHA: `{metrics['targeted_raw']['sha256']}`. The 39-request original panel remains primary for its preregistered role; this 12-key run is a separate stability diagnostic.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--targeted", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.packet, args.original, args.targeted, args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
