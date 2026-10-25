#!/usr/bin/env python3
"""Build a strict, paired comparison of two Nemotron request settings.

The settings are two runs of one model, not two independent judges.  This
module deliberately keeps them in separate five-judge panels and never feeds
both Nemotron bundles to one aggregate/ICC calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aggregate_pilot import icc_two_way_random_absolute, kendall_w, read_json
from build_fivejudge_report import slug
from validate_import import validate_import


MODE_LABELS = {"off": "thinking-off", "on": "thinking-on"}
TRACKS = ("qualification", "pointwise", "pairwise")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n.a."
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n.a."
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    number = _parse_float(value)
    return "n.a." if number is None else f"{100.0 * number:.{digits}f}%"


def _median(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if _parse_float(value) is not None]
    return statistics.median(numbers) if numbers else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _event_track(request_id: Any) -> str | None:
    value = str(request_id or "")
    for track in TRACKS:
        if value.startswith(track + "-"):
            return track
    return None


def _choice_finish_reason(event: Mapping[str, Any]) -> str | None:
    response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    if choices and isinstance(choices[0], Mapping):
        value = choices[0].get("finish_reason")
        return str(value) if value is not None else None
    return None


def _usage_value(event: Mapping[str, Any], *keys: str) -> float | None:
    response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    for key in keys:
        value = _parse_float(usage.get(key))
        if value is not None:
            return value
    return None


def _timings_value(event: Mapping[str, Any], *keys: str) -> float | None:
    response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    for key in keys:
        value = _parse_float(timings.get(key))
        if value is not None:
            return value
    return None


def _quality_telemetry(raw: Mapping[str, Any]) -> dict[str, Any]:
    events = raw.get("raw_events") if isinstance(raw.get("raw_events"), list) else []
    result: dict[str, Any] = {}
    all_quality_events = [event for event in events if isinstance(event, Mapping) and _event_track(event.get("request_id"))]
    result["quality_event_count"] = len(all_quality_events)
    result["quality_event_status_counts"] = dict(sorted(Counter(str(event.get("status")) for event in all_quality_events).items()))
    result["quality_finish_reason_counts"] = dict(sorted(Counter(str(_choice_finish_reason(event) or "missing") for event in all_quality_events).items()))
    result["quality_failed_events"] = sum(1 for event in all_quality_events if event.get("status") != "ok")
    result["quality_truncated_events"] = sum(
        1
        for event in all_quality_events
        if str(_choice_finish_reason(event) or "").lower() in {"length", "max_tokens", "truncated"}
        or str(event.get("status") or "").lower() in {"truncated", "timeout"}
    )
    by_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in all_quality_events:
        track = _event_track(event.get("request_id"))
        if track:
            by_track[track].append(event)
    for track in TRACKS:
        events_for_track = by_track.get(track, [])
        elapsed = [value for value in (_parse_float(event.get("elapsed_s")) for event in events_for_track) if value is not None]
        completion = [value for value in (_usage_value(event, "completion_tokens", "output_tokens") for event in events_for_track) if value is not None]
        prompt = [value for value in (_usage_value(event, "prompt_tokens", "input_tokens") for event in events_for_track) if value is not None]
        total = [value for value in (_usage_value(event, "total_tokens") for event in events_for_track) if value is not None]
        reasoning = [
            value
            for value in (
                _usage_value(event, "reasoning_tokens", "thinking_tokens")
                for event in events_for_track
            )
            if value is not None
        ]
        generation = [value for value in (_timings_value(event, "predicted_per_second", "generation_tokens_per_second", "tokens_per_second") for event in events_for_track) if value is not None]
        result[track] = {
            "event_count": len(events_for_track),
            "ok_events": sum(1 for event in events_for_track if event.get("status") == "ok"),
            "failed_events": sum(1 for event in events_for_track if event.get("status") != "ok"),
            "truncated_events": sum(1 for event in events_for_track if str(_choice_finish_reason(event) or "").lower() in {"length", "max_tokens", "truncated"}),
            "latency_median_s": _median(elapsed),
            "latency_p95_s": _quantile(elapsed, 0.95),
            "completion_tokens_median": _median(completion),
            "prompt_tokens_median": _median(prompt),
            "total_tokens_median": _median(total),
            "reasoning_tokens_median": _median(reasoning),
            "generation_tokens_per_s_median": _median(generation),
            "elapsed_n": len(elapsed),
            "completion_n": len(completion),
            "reasoning_n": len(reasoning),
        }
    return result


def _request_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    events = raw.get("raw_events") if isinstance(raw.get("raw_events"), list) else []
    quality = next((event for event in events if isinstance(event, Mapping) and _event_track(event.get("request_id")) and event.get("status") == "ok"), None)
    request = quality.get("request") if isinstance(quality, Mapping) and isinstance(quality.get("request"), Mapping) else {}
    template = request.get("chat_template_kwargs") if isinstance(request.get("chat_template_kwargs"), Mapping) else {}
    top_settings = raw.get("settings") if isinstance(raw.get("settings"), Mapping) else {}
    max_tokens_by_track: dict[str, Any] = {}
    configured_caps = top_settings.get("max_tokens_by_track")
    if isinstance(configured_caps, Mapping):
        for track, value in configured_caps.items():
            parsed = _parse_int(value)
            if parsed is not None:
                max_tokens_by_track[str(track)] = parsed
    for event in events:
        if not isinstance(event, Mapping):
            continue
        request_id = str(event.get("request_id") or "")
        track = _event_track(request_id)
        if track is None and request_id.startswith("capability-"):
            track = "capability"
        if not track:
            continue
        event_request = event.get("request") if isinstance(event.get("request"), Mapping) else {}
        cap = _parse_int(event_request.get("max_tokens"))
        if cap is not None:
            max_tokens_by_track.setdefault(track, cap)
    return {
        "enable_thinking": template.get("enable_thinking", top_settings.get("enable_thinking")),
        "preserve_reasoning": template.get("preserve_reasoning", top_settings.get("preserve_reasoning")),
        "reasoning_format": request.get("reasoning_format", top_settings.get("reasoning_format")),
        "reasoning_budget_tokens": request.get("reasoning_budget_tokens", top_settings.get("reasoning_budget_tokens")),
        "thinking_budget_tokens": request.get("thinking_budget_tokens", top_settings.get("thinking_budget_tokens")),
        "response_format": request.get("response_format", top_settings.get("response_format")),
        "max_tokens": request.get("max_tokens", top_settings.get("max_tokens")),
        "max_tokens_by_track": max_tokens_by_track,
        "temperature": request.get("temperature", top_settings.get("temperature")),
        "top_p": request.get("top_p", top_settings.get("top_p")),
        "seed": request.get("seed", top_settings.get("seed")),
    }


def _raw_pointwise_response_errors(raw: Mapping[str, Any], raw_path: Path) -> list[dict[str, Any]]:
    """Preserve error objects from HTTP-successful rows later rejected by shape validation.

    The runner's failed record can intentionally omit the nested pointwise
    object after a contract failure, while the raw HTTP response still
    contains the judge's two error objects.  Keep those objects as a clearly
    labelled raw-response annotation.  This is evidence preservation only:
    no score, quote, or eligibility field is repaired or reinterpreted.
    """

    record_map: dict[str, Mapping[str, Any]] = {}
    for record in raw.get("records") if isinstance(raw.get("records"), list) else []:
        if not isinstance(record, Mapping) or record.get("track") != "pointwise":
            continue
        request_id = f"pointwise-{record.get('item_id')}-{record.get('candidate_id')}"
        record_map[request_id] = record
    result: list[dict[str, Any]] = []
    for event in raw.get("raw_events") if isinstance(raw.get("raw_events"), list) else []:
        if not isinstance(event, Mapping):
            continue
        request_id = str(event.get("request_id") or "")
        if not request_id.startswith("pointwise-"):
            continue
        response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
        choices = response.get("choices") if isinstance(response.get("choices"), list) else []
        message = choices[0].get("message") if choices and isinstance(choices[0], Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, Mapping) or not isinstance(payload.get("errors"), list):
            continue
        record = record_map.get(request_id, {})
        # Valid pointwise rows already flow through the panel's ordinary
        # error-annotation CSV.  Preserve raw response errors here only when
        # the runner/validator rejected the row or its scalar shape.  This
        # avoids duplicating valid judge annotations while retaining the
        # failed response that the normalized record cannot represent.
        if record.get("status") == "ok":
            pointwise = record.get("pointwise") if isinstance(record.get("pointwise"), Mapping) else {}
            score = pointwise.get("score_0_100")
            subscores = pointwise.get("subscores") if isinstance(pointwise.get("subscores"), Mapping) else {}
            try:
                shape_valid = score is not None and sum(float(subscores[key]) for key in ("terminology_locale", "semantic_safety", "fluency", "completeness_format")) == float(score)
            except (TypeError, ValueError, KeyError):
                shape_valid = False
            if shape_valid:
                continue
        response_score = payload.get("score_0_100")
        response_subscores = payload.get("subscores") if isinstance(payload.get("subscores"), Mapping) else {}
        try:
            response_subscore_sum = sum(float(response_subscores[key]) for key in ("terminology_locale", "semantic_safety", "fluency", "completeness_format"))
        except (TypeError, ValueError, KeyError):
            response_subscore_sum = None
        response_shape_issue = ""
        response_score_number = _parse_float(response_score)
        if response_score_number is not None and response_subscore_sum is not None and response_subscore_sum != response_score_number:
            response_shape_issue = f"raw response score {response_score_number:g} vs subscore sum {response_subscore_sum:g}"
        elif record.get("status") != "ok":
            response_shape_issue = str(record.get("reason") or record.get("error") or "raw response record was not eligible")
        for index, error in enumerate(payload["errors"]):
            if not isinstance(error, Mapping):
                continue
            result.append({
                "item_id": record.get("item_id", request_id.removeprefix("pointwise-").rsplit("-", 1)[0]),
                "source_id": record.get("source_id", ""),
                "candidate_id": record.get("candidate_id", ""),
                "model_alias": "",
                "judge_id": record.get("judge_id", ""),
                "judge_family": record.get("family", ""),
                "error_index": f"raw_response_error_{index}",
                "severity": error.get("severity", ""),
                "type": error.get("type", ""),
                "target_exact_quote": error.get("target_exact_quote", ""),
                "source_evidence": error.get("source_evidence", ""),
                "source_span": error.get("source_span", ""),
                "target_anchor": error.get("target_anchor", ""),
                "confidence": error.get("confidence", ""),
                "exact_quote_status": "raw_response_preserved_not_revalidated",
                "source_text_sha256": record.get("source_text_sha256", ""),
                "bundle": str(raw_path),
                "mapping_scope": "owner_only_postjudge",
                "annotation_kind": "raw_response_error",
                "raw_event_request_id": request_id,
                "raw_event_response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "raw_response_score": response_score,
                "raw_response_subscores": _json_text(response_subscores),
                "raw_response_subscore_sum": response_subscore_sum,
                "raw_response_score_shape_issue": response_shape_issue,
            })
    return result


def _raw_mode_summary(raw_path: Path, raw: Mapping[str, Any], mode: str, validation: Mapping[str, Any]) -> dict[str, Any]:
    judge = raw.get("judge") if isinstance(raw.get("judge"), Mapping) else {}
    judge_id = str(judge.get("judge_id") or raw.get("judge_id") or "")
    settings = _request_settings(raw)
    probe_path = raw_path.parent / "reasoning_budget_tokenize_probe.json"
    probe = read_json(probe_path) if probe_path.exists() else {}
    audit_path = raw_path.parent / "reasoning_budget_tokenize_audit.json"
    budget_audit = read_json(audit_path) if audit_path.exists() else {}
    budget_counts = budget_audit.get("counts") if isinstance(budget_audit.get("counts"), Mapping) else {}
    probe_samples = probe.get("samples") if isinstance(probe.get("samples"), list) else []
    probe_at_or_over = [sample for sample in probe_samples if isinstance(sample, Mapping) and sample.get("token_count_at_or_over_budget") is True]
    probe_below = [sample for sample in probe_samples if isinstance(sample, Mapping) and sample.get("token_count_at_or_over_budget") is False]
    raw_response_errors = _raw_pointwise_response_errors(raw, raw_path)
    contract_failures: list[dict[str, Any]] = []
    seen_contract_keys: set[tuple[str, str]] = set()
    for error in raw_response_errors:
        key = (str(error.get("item_id") or ""), str(error.get("candidate_id") or ""))
        if key in seen_contract_keys:
            continue
        seen_contract_keys.add(key)
        contract_failures.append({
            "item_id": key[0],
            "candidate_id": key[1],
            "raw_response_score": error.get("raw_response_score"),
            "raw_response_subscores": error.get("raw_response_subscores", ""),
            "raw_response_subscore_sum": error.get("raw_response_subscore_sum"),
            "raw_response_score_shape_issue": error.get("raw_response_score_shape_issue", ""),
            "raw_event_request_id": error.get("raw_event_request_id", ""),
            "raw_event_response_sha256": error.get("raw_event_response_sha256", ""),
        })
    return {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "raw_path": str(raw_path),
        "raw_bundle_sha256": sha256_file(raw_path),
        "judge_id": judge_id,
        "judge_family": str(judge.get("family") or raw.get("family") or ""),
        "raw_status": raw.get("status"),
        "raw_complete": raw.get("complete"),
        "raw_record_count": len(raw.get("records") or []) if isinstance(raw.get("records"), list) else None,
        "raw_event_count": len(raw.get("raw_events") or []) if isinstance(raw.get("raw_events"), list) else None,
        "validation_status": validation.get("status"),
        "validation_errors": validation.get("errors") or [],
        "validation_warnings": validation.get("warnings") or [],
        "track_eligibility": validation.get("track_eligibility") or {},
        "request_settings": settings,
        "reasoning_budget_probe_path": str(probe_path) if probe_path.exists() else "",
        "reasoning_budget_probe_sha256": sha256_file(probe_path) if probe_path.exists() else "",
        "reasoning_budget_tokenize_method": probe.get("interpretation", "") if isinstance(probe, Mapping) else "",
        "reasoning_budget_tokenize_budget": probe.get("budget") if isinstance(probe, Mapping) else None,
        "reasoning_budget_probe_confirmed_count": len(probe_at_or_over),
        "reasoning_budget_probe_below_count": len(probe_below),
        "reasoning_budget_probe_samples": probe_samples,
        "reasoning_budget_audit_path": str(audit_path) if audit_path.exists() else "",
        "reasoning_budget_audit_sha256": sha256_file(audit_path) if audit_path.exists() else "",
        "reasoning_budget_audit_counts": dict(budget_counts),
        "reasoning_budget_audit_non_generating": budget_audit.get("non_generating") if isinstance(budget_audit, Mapping) else None,
        "reasoning_budget_audit_interpretation": budget_audit.get("interpretation", "") if isinstance(budget_audit, Mapping) else "",
        "raw_pointwise_response_errors": raw_response_errors,
        "raw_pointwise_contract_failures": contract_failures,
        "telemetry": _quality_telemetry(raw),
    }


def _panel_paths(panel: Path) -> dict[str, Path]:
    return {
        "long": panel / "pointwise_results_long.fivejudges.owner.csv",
        "comparison": panel / "translation_comparison.fivejudges.owner.csv",
        "summary": panel / "judge_summary.fivejudges.owner.csv",
        "pairwise": panel / "pairwise_judge_diagnostics.fivejudges.owner.csv",
        "errors": panel / "error_annotations.fivejudges.owner.csv",
        "qualification": panel / "qualification_summary.fivejudges.owner.json",
        "aggregate": panel / "aggregate.fivejudges.json",
    }


def _find_judge_id(rows: Sequence[Mapping[str, Any]], preferred: str) -> str:
    if any(str(row.get("judge_id")) == preferred for row in rows):
        return preferred
    candidates = sorted({str(row.get("judge_id") or "") for row in rows if "nemotron" in str(row.get("judge_id") or "").lower()})
    if len(candidates) != 1:
        raise ValueError(f"cannot identify one Nemotron judge in panel rows: {candidates}")
    return candidates[0]


def _mode_pointwise(panel: Path, raw_summary: Mapping[str, Any]) -> dict[str, Any]:
    paths = _panel_paths(panel)
    long_rows = _read_csv(paths["long"])
    judge_id = _find_judge_id(long_rows, str(raw_summary["judge_id"]))
    rows = [row for row in long_rows if row.get("judge_id") == judge_id]
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("item_id") or row.get("source_id") or ""), str(row.get("candidate_id") or ""))
        if key in by_key:
            raise ValueError(f"duplicate pointwise key for {judge_id}: {key}")
        by_key[key] = row
    comparison_rows = _read_csv(paths["comparison"])
    exact = {(str(row.get("item_id") or row.get("source_id") or ""), str(row.get("candidate_id") or "")): row for row in comparison_rows}
    # The owner comparison table is the frozen 21-unit inventory.  Keep a
    # missing/failed raw row visible in the paired output rather than silently
    # shrinking the key set or imputing a score.  Normal five-judge builds
    # include the row in the long table, but this guard also handles a
    # deliberately partial exploratory panel.
    missing_pointwise_keys = sorted(set(exact) - set(by_key))
    for key, comparison_row in exact.items():
        if key not in by_key:
            item_id, candidate_id = key
            by_key[key] = {
                "item_id": item_id,
                "source_id": comparison_row.get("source_id", item_id),
                "candidate_id": candidate_id,
                "candidate_family": comparison_row.get("candidate_family", ""),
                "model_alias": comparison_row.get("model_alias", ""),
                "score_0_100": "",
                "score_eligible_for_mean": "False",
                "score_shape_status": "MISSING",
                "score_shape_issues": "missing pointwise row in panel long table",
                "exact_quote_status": "",
                "critical": "",
                "error_count": "",
                "score_eligibility_issues": "missing pointwise row in panel long table",
            }
    summary_rows = [row for row in _read_csv(paths["summary"]) if row.get("judge_id") == judge_id]
    if len(summary_rows) != 1:
        raise ValueError(f"expected one judge summary row for {judge_id}, got {len(summary_rows)}")
    qualification = read_json(paths["qualification"])
    per_qual = qualification.get("per_judge", {}).get(judge_id, {}) if isinstance(qualification.get("per_judge"), Mapping) else {}
    aggregate = read_json(paths["aggregate"])
    errors = [row for row in _read_csv(paths["errors"]) if row.get("judge_id") == judge_id]
    pairwise = [row for row in _read_csv(paths["pairwise"]) if row.get("judge_id") == judge_id]
    return {
        "panel": str(panel),
        "panel_aggregate_sha256": sha256_file(paths["aggregate"]),
        "judge_id": judge_id,
        "pointwise": by_key,
        "comparison": exact,
        "judge_summary": summary_rows[0],
        "qualification": per_qual,
        "qualification_summary": qualification,
        "aggregate": aggregate,
        "errors": errors,
        "pairwise": pairwise,
        "missing_pointwise_keys": missing_pointwise_keys,
    }


def _q_metric(qualification: Mapping[str, Any], name: str, nested: Sequence[str] = ()) -> tuple[Any, Any, Any]:
    value: Any = qualification.get(name)
    for key in nested:
        value = value.get(key) if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping):
        return None, None, None
    return value.get("estimate"), value.get("numerator"), value.get("denominator")


def _build_summary_row(mode: str, raw_summary: Mapping[str, Any], panel_data: Mapping[str, Any]) -> dict[str, Any]:
    summary = panel_data["judge_summary"]
    q = panel_data["qualification"] if isinstance(panel_data["qualification"], Mapping) else {}
    qualification_audit = q.get("qualification_score_audit") if isinstance(q.get("qualification_score_audit"), Mapping) else {}
    telemetry = raw_summary["telemetry"]
    row: dict[str, Any] = {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "judge_id": panel_data["judge_id"],
        "judge_family": summary.get("judge_family", raw_summary.get("judge_family", "")),
        "panel": panel_data["panel"],
        "panel_aggregate_sha256": panel_data["panel_aggregate_sha256"],
        "raw_bundle_sha256": raw_summary["raw_bundle_sha256"],
        "validation_status": raw_summary["validation_status"],
        "pointwise_rows": _parse_int(summary.get("pointwise_rows")),
        "pointwise_valid_rows": _parse_int(summary.get("pointwise_valid_rows")),
        "pointwise_mean": _parse_float(summary.get("pointwise_mean")),
        "pointwise_median": _parse_float(summary.get("pointwise_median")),
        "pointwise_score_sd_population": _parse_float(summary.get("pointwise_score_sd_population")),
        "pointwise_score_sd_sample": _parse_float(summary.get("pointwise_score_sd_sample")),
        "pointwise_score_unique_values": summary.get("pointwise_score_unique_values", ""),
        "pointwise_score_value_counts": summary.get("pointwise_score_value_counts", ""),
        "pointwise_top_tie_items": _parse_int(summary.get("pointwise_top_tie_items")),
        "pointwise_top_tie_slots": _parse_int(summary.get("pointwise_top_tie_slots")),
        "pointwise_strict_top_items": _parse_int(summary.get("pointwise_strict_top_items")),
        "pointwise_error_annotations": _parse_int(summary.get("pointwise_error_annotations")),
        "pointwise_critical_flags": _parse_int(summary.get("pointwise_critical_flags")),
        "pointwise_primary_status": "ELIGIBLE" if _parse_int(summary.get("pointwise_valid_rows")) == 21 else "NOT_ELIGIBLE_POINTWISE",
        "raw_pointwise_contract_failure_count": len(raw_summary.get("raw_pointwise_contract_failures", [])),
        "raw_pointwise_contract_failure_details": _json_text(raw_summary.get("raw_pointwise_contract_failures", [])),
        "qualification_variants": _parse_int(summary.get("qualification_variants")),
        "qualification_scalar_valid": _parse_int(summary.get("qualification_scalar_valid")),
        "qualification_scalar_invalid": _parse_int(summary.get("qualification_scalar_invalid")),
        "qualification_scalar_missing": _parse_int(summary.get("qualification_scalar_missing")),
        "qualification_scalar_abstain": _parse_int(summary.get("qualification_scalar_abstain")),
        "qualification_scalar_status": summary.get("qualification_scalar_status", ""),
        "qualification_evidence_critical_status": summary.get("qualification_evidence_critical_status", ""),
        "qualification_error_annotations": _parse_int(summary.get("qualification_error_annotations")),
        "qualification_critical_flags": _parse_int(summary.get("qualification_critical_flags")),
        "qualification_audit_status": qualification_audit.get("status", ""),
        "qualification_audit_valid": qualification_audit.get("valid"),
        "qualification_audit_invalid": qualification_audit.get("invalid"),
        "qualification_audit_missing": qualification_audit.get("missing"),
        "qualification_audit_abstain": qualification_audit.get("abstain"),
        "pairwise_rows": _parse_int(summary.get("pairwise_rows")),
        "pairwise_complete_swaps": _parse_int(summary.get("pairwise_complete_swaps")),
        "pairwise_consistent": _parse_int(summary.get("pairwise_consistent")),
        "pairwise_position_sensitive": _parse_int(summary.get("pairwise_position_sensitive")),
        "pairwise_tie_disagreement": _parse_int(summary.get("pairwise_tie_disagreement")),
        "pairwise_swap_consistency": _parse_float(summary.get("pairwise_swap_consistency")),
        "pairwise_context_status": summary.get("pairwise_fresh_chat_independence", "") or "not_recorded",
        "pairwise_known_operational_evidence": "fresh independent request per orientation",
        "judge_context_packaging": summary.get("judge_context_packaging", ""),
        "request_settings": _json_text(raw_summary["request_settings"]),
        "reasoning_budget_probe_sha256": raw_summary.get("reasoning_budget_probe_sha256", ""),
        "reasoning_budget_tokenize_method": raw_summary.get("reasoning_budget_tokenize_method", ""),
        "reasoning_budget_tokenize_budget": raw_summary.get("reasoning_budget_tokenize_budget", ""),
        "reasoning_budget_probe_confirmed_count": raw_summary.get("reasoning_budget_probe_confirmed_count", ""),
        "reasoning_budget_probe_below_count": raw_summary.get("reasoning_budget_probe_below_count", ""),
        "reasoning_budget_audit_sha256": raw_summary.get("reasoning_budget_audit_sha256", ""),
        "reasoning_budget_audit_counts": _json_text(raw_summary.get("reasoning_budget_audit_counts", {})),
        "reasoning_budget_audit_non_generating": raw_summary.get("reasoning_budget_audit_non_generating", ""),
        "reasoning_budget_audit_interpretation": raw_summary.get("reasoning_budget_audit_interpretation", ""),
        "raw_complete": raw_summary.get("raw_complete"),
        "raw_record_count": raw_summary.get("raw_record_count"),
        "raw_event_count": raw_summary.get("raw_event_count"),
        "quality_failed_events": raw_summary["telemetry"].get("quality_failed_events"),
        "quality_truncated_events": raw_summary["telemetry"].get("quality_truncated_events"),
        "quality_finish_reason_counts": _json_text(raw_summary["telemetry"].get("quality_finish_reason_counts", {})),
    }
    for name, nested in (
        ("mutation_flag_rate_proxy", ()),
        ("mutation_evidence_supported_rate", ()),
        ("error_type_match", ()),
        ("invariance_accuracy", ()),
        ("clean_false_positive_rate", ()),
        ("critical_severity_recall_union", ("critical_severity_recall", "union_top_level_or_error_severity")),
    ):
        lookup_name = name
        lookup_nested: Sequence[str] = nested
        if name == "critical_severity_recall_union":
            lookup_name, lookup_nested = nested[0], nested[1:]
        estimate, numerator, denominator = _q_metric(q, lookup_name, lookup_nested)
        row[f"qualification_{name}_estimate"] = estimate
        row[f"qualification_{name}_numerator"] = numerator
        row[f"qualification_{name}_denominator"] = denominator
    for track in TRACKS:
        metrics = telemetry.get(track, {})
        for key in ("event_count", "ok_events", "failed_events", "truncated_events", "latency_median_s", "latency_p95_s", "completion_tokens_median", "prompt_tokens_median", "total_tokens_median", "reasoning_tokens_median", "generation_tokens_per_s_median", "elapsed_n", "completion_n", "reasoning_n"):
            row[f"{track}_{key}"] = metrics.get(key)
    return row


def _paired_rows(off: Mapping[str, Any], on: Mapping[str, Any], raw_off: Mapping[str, Any], raw_on: Mapping[str, Any]) -> list[dict[str, Any]]:
    off_keys = set(off["pointwise"])
    on_keys = set(on["pointwise"])
    if off_keys != on_keys:
        missing_off = sorted(on_keys - off_keys)
        missing_on = sorted(off_keys - on_keys)
        raise ValueError(f"thinking mode pointwise key sets differ; missing_off={missing_off}, missing_on={missing_on}")
    if len(off_keys) != 21:
        raise ValueError(f"expected 21 matched pointwise units, got {len(off_keys)}")
    rows: list[dict[str, Any]] = []

    def raw_response_error_count(raw_summary: Mapping[str, Any], item_id: str, candidate_id: str) -> int:
        """Count preserved raw HTTP error annotations for one pointwise unit.

        This is deliberately separate from the validated record's ``error_count``.
        A failed/shape-invalid normalized record may not contain an ``errors``
        array at all even when the HTTP-200 JSON response did.  The paired table
        therefore exposes this count only as raw-response provenance and never
        treats it as a validated judge annotation.
        """

        raw_errors = raw_summary.get("raw_pointwise_response_errors", [])
        if not isinstance(raw_errors, Sequence):
            return 0
        return sum(
            1
            for error in raw_errors
            if isinstance(error, Mapping)
            and str(error.get("item_id") or "") == item_id
            and str(error.get("candidate_id") or "") == candidate_id
        )

    for key in sorted(off_keys):
        item_id, candidate_id = key
        off_row = off["pointwise"][key]
        on_row = on["pointwise"][key]
        off_comp = off["comparison"].get(key, {})
        on_comp = on["comparison"].get(key, {})
        if off_comp.get("source_text_sha256") != on_comp.get("source_text_sha256"):
            raise ValueError(f"source hash differs between modes for {key}")
        if off_comp.get("source_en") != on_comp.get("source_en") or off_comp.get("target_zh_tw") != on_comp.get("target_zh_tw"):
            raise ValueError(f"exact source/target text differs between modes for {key}")
        off_score = _parse_float(off_row.get("score_0_100"))
        on_score = _parse_float(on_row.get("score_0_100"))
        off_ok = _parse_bool(off_row.get("score_eligible_for_mean")) is True and off_score is not None
        on_ok = _parse_bool(on_row.get("score_eligible_for_mean")) is True and on_score is not None
        # ``error_count=0`` on a failed normalized row means that the runner
        # had no validated record-level error array; it does not establish a
        # clean response.  Keep validated counts blank for ineligible rows and
        # expose any preserved HTTP-response error count separately.
        off_error_count = off_row.get("error_count", "") if off_ok else ""
        on_error_count = on_row.get("error_count", "") if on_ok else ""
        rows.append({
            "source_id": off_comp.get("source_id") or off_row.get("source_id"),
            "item_id": item_id,
            "candidate_id": candidate_id,
            "candidate_family": off_comp.get("candidate_family") or off_row.get("candidate_family"),
            "model_alias": off_comp.get("model_alias") or off_row.get("model_alias"),
            "source_en": off_comp.get("source_en", ""),
            "target_zh_tw": off_comp.get("target_zh_tw", ""),
            "source_text_sha256": off_comp.get("source_text_sha256", ""),
            "off_judge_id": off["judge_id"],
            "on_judge_id": on["judge_id"],
            "off_raw_bundle_sha256": raw_off["raw_bundle_sha256"],
            "on_raw_bundle_sha256": raw_on["raw_bundle_sha256"],
            "off_panel_aggregate_sha256": off["panel_aggregate_sha256"],
            "on_panel_aggregate_sha256": on["panel_aggregate_sha256"],
            "off_score_0_100": off_row.get("score_0_100", ""),
            "on_score_0_100": on_row.get("score_0_100", ""),
            "score_delta_on_minus_off": (on_score - off_score) if off_ok and on_ok else "",
            "off_score_eligible": off_ok,
            "on_score_eligible": on_ok,
            "paired_score_status": "paired_eligible" if off_ok and on_ok else "ineligible_or_missing",
            "score_changed": (off_score != on_score) if off_ok and on_ok else "",
            "off_score_shape_status": off_row.get("score_shape_status", ""),
            "on_score_shape_status": on_row.get("score_shape_status", ""),
            "off_exact_quote_status": off_row.get("exact_quote_status", ""),
            "on_exact_quote_status": on_row.get("exact_quote_status", ""),
            "off_critical": off_row.get("critical", ""),
            "on_critical": on_row.get("critical", ""),
            "off_error_count": off_error_count,
            "on_error_count": on_error_count,
            "off_error_count_scope": "validated_record" if off_ok else "unknown_ineligible_missing_pointwise",
            "on_error_count_scope": "validated_record" if on_ok else "unknown_ineligible_missing_pointwise",
            "off_raw_response_error_count": raw_response_error_count(raw_off, item_id, candidate_id),
            "on_raw_response_error_count": raw_response_error_count(raw_on, item_id, candidate_id),
        })
    return rows


def _panel_sensitivity_rows(mode: str, panel_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = panel_data["aggregate"]
    pointwise = aggregate.get("pointwise") if isinstance(aggregate.get("pointwise"), Mapping) else {}
    mappings = {}
    for row in panel_data["comparison"].values():
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            mappings[candidate_id] = (row.get("candidate_family", ""), row.get("model_alias", ""))
    candidates = []
    for candidate_id, report in pointwise.items():
        if not isinstance(report, Mapping):
            continue
        mean = _parse_float(report.get("score_mean"))
        if mean is None:
            continue
        candidates.append((candidate_id, mean, report))
    candidates.sort(key=lambda value: (-value[1], value[0]))
    long_rows = _read_csv(Path(panel_data["panel"]) / "pointwise_results_long.fivejudges.owner.csv")
    expected_source_ids = sorted({
        str(row.get("source_id") or row.get("item_id") or "")
        for row in panel_data["comparison"].values()
        if str(row.get("source_id") or row.get("item_id") or "")
    })
    rows: list[dict[str, Any]] = []
    for rank, (candidate_id, mean, report) in enumerate(candidates, 1):
        family, alias = mappings.get(candidate_id, ("", ""))
        balanced = report.get("score_mean_family_balanced") if isinstance(report.get("score_mean_family_balanced"), Mapping) else {}
        candidate_keys = {
            key
            for key, point_row in panel_data["pointwise"].items()
            if key[1] == str(candidate_id)
            and _parse_bool(point_row.get("score_eligible_for_mean")) is True
            and _parse_float(point_row.get("score_0_100")) is not None
        }
        panel_valid_keys = {
            (str(row.get("item_id") or row.get("source_id") or ""), str(row.get("candidate_id") or ""))
            for row in long_rows
            if str(row.get("candidate_id") or "") == str(candidate_id)
            and _parse_bool(row.get("score_eligible_for_mean")) is True
            and _parse_float(row.get("score_0_100")) is not None
        }
        source_coverage = sorted({str(next((row.get("source_id") for row in long_rows if (str(row.get("item_id") or row.get("source_id") or ""), str(row.get("candidate_id") or "")) == key), key[0])) for key in panel_valid_keys})
        nemotron_source_coverage = sorted({str(panel_data["comparison"].get(key, {}).get("source_id") or key[0]) for key in candidate_keys})
        rows.append({
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "candidate_id": candidate_id,
            "candidate_family": family,
            "model_alias": alias,
            "rank": rank,
            "score_mean": mean,
            "score_median": _parse_float(report.get("score_median")),
            "score_iqr": _parse_float(report.get("score_iqr")),
            "n_valid_scores": report.get("n_valid_scores"),
            "n_distinct_judges": report.get("n_distinct_judges"),
            "n_valid_source_units": len(panel_valid_keys),
            "panel_source_units_expected": len(expected_source_ids),
            "source_coverage": ";".join(source_coverage),
            "nemotron_valid_source_units": len(candidate_keys),
            "nemotron_source_units_expected": len(expected_source_ids),
            "nemotron_source_coverage": ";".join(nemotron_source_coverage),
            "candidate_coverage_status": "complete_three_sources" if len(panel_valid_keys) == 3 and len(candidate_keys) == 3 else "partial_source_coverage",
            "family_balanced_mean": balanced.get("estimate"),
            "panel_aggregate_sha256": panel_data["panel_aggregate_sha256"],
            "sensitivity_scope": "exploratory separate five-judge panel; not a joint six-judge panel",
        })
    return rows


def _reliability_row(mode: str, panel_data: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = panel_data["aggregate"]
    reliability = aggregate.get("reliability_overall") if isinstance(aggregate.get("reliability_overall"), Mapping) else {}
    icc = reliability.get("icc_2_1_2_k") if isinstance(reliability.get("icc_2_1_2_k"), Mapping) else {}
    kendall = reliability.get("kendall_w") if isinstance(reliability.get("kendall_w"), Mapping) else {}
    family = aggregate.get("family_sensitivity") if isinstance(aggregate.get("family_sensitivity"), Mapping) else {}
    lofo = family.get("leave_one_family_out") if isinstance(family.get("leave_one_family_out"), Mapping) else {}
    return {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "panel_aggregate_sha256": panel_data["panel_aggregate_sha256"],
        "n_judges_actual": icc.get("n_judges"),
        "n_units_actual": icc.get("n_units"),
        "icc_model": icc.get("model"),
        "icc_2_1": icc.get("icc_2_1"),
        "icc_2_k": icc.get("icc_2_k"),
        "icc_ci_status": "not available; implementation does not estimate ICC CI",
        "kendall_w": kendall.get("estimate"),
        "kendall_w_n_rankings": kendall.get("n_rankings"),
        "lofo_status": lofo.get("status"),
        "lofo_baseline_ranking": _json_text(lofo.get("baseline_ranking")),
        "lofo_by_family": _json_text(lofo.get("leave_one_family_out")),
        "reliability_scope": "full separate five-judge panel",
        "primary_panel_status": "not_assessed",
        "excluded_keys": "",
        "scope_note": "Exploratory separate five-judge sensitivity; n_source_clusters=3; not a joint six-judge estimate.",
    }


def _matched_panel_metrics(panel_data: Mapping[str, Any], keys: set[tuple[str, str]]) -> dict[str, Any]:
    """Compute complete-unit reliability on an explicitly shared key set.

    This is an in-memory restriction of the immutable panel CSV.  It is used
    only to compare the on panel's 20 complete units with the same 20 units
    from the off panel; no raw/panel source is rewritten and no score is
    filled for the excluded unit.
    """

    long_path = Path(panel_data["panel"]) / "pointwise_results_long.fivejudges.owner.csv"
    rows = _read_csv(long_path)
    judge_ids = sorted({str(row.get("judge_id") or "") for row in rows if str(row.get("judge_id") or "")})
    values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("item_id") or row.get("source_id") or ""), str(row.get("candidate_id") or ""))
        if key not in keys:
            continue
        score = _parse_float(row.get("score_0_100"))
        eligible = _parse_bool(row.get("score_eligible_for_mean")) is True
        if score is None or not eligible:
            continue
        judge_id = str(row.get("judge_id") or "")
        values[key][judge_id] = score
        metadata[key] = row
    complete_keys = sorted(key for key, judge_scores in values.items() if len(judge_scores) == len(judge_ids) and len(judge_ids) >= 2)
    matrix = [[values[key][judge] for judge in judge_ids] for key in complete_keys]
    icc = icc_two_way_random_absolute(matrix)
    normalized: list[dict[str, Any]] = []
    for key in complete_keys:
        row_meta = metadata[key]
        item_id, candidate_id = key
        source_id = str(row_meta.get("source_id") or item_id)
        for judge_id in judge_ids:
            normalized.append({
                "_status": "ok",
                "_judge_id": judge_id,
                "_score": values[key][judge_id],
                "_source_id": source_id,
                "_item_id": item_id,
                "_candidate_id": candidate_id,
            })
    concordance = kendall_w(normalized)
    return {
        "icc": icc,
        "kendall_w": concordance,
        "n_complete_units": len(complete_keys),
        "n_judges": len(judge_ids),
        "excluded_keys": sorted(keys - set(complete_keys)),
    }


def _matched_reliability_row(
    mode: str,
    panel_data: Mapping[str, Any],
    keys: set[tuple[str, str]],
    label: str,
) -> dict[str, Any]:
    metrics = _matched_panel_metrics(panel_data, keys)
    icc = metrics["icc"] if isinstance(metrics.get("icc"), Mapping) else {}
    concordance = metrics["kendall_w"] if isinstance(metrics.get("kendall_w"), Mapping) else {}
    return {
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "panel_aggregate_sha256": panel_data["panel_aggregate_sha256"],
        "n_judges_actual": icc.get("n_judges", metrics.get("n_judges")),
        "n_units_actual": icc.get("n_units", metrics.get("n_complete_units")),
        "icc_model": icc.get("model"),
        "icc_2_1": icc.get("icc_2_1"),
        "icc_2_k": icc.get("icc_2_k"),
        "icc_ci_status": "not available; implementation does not estimate ICC CI",
        "kendall_w": concordance.get("estimate"),
        "kendall_w_n_rankings": concordance.get("n_rankings"),
        "lofo_status": "not_run_for_matched_subset",
        "lofo_baseline_ranking": "",
        "lofo_by_family": "",
        "reliability_scope": label,
        "primary_panel_status": "diagnostic_matched_subset",
        "scope_note": "Matched common eligible pointwise units only; same five judges in each mode; not a joint six-judge estimate; n_source_clusters=3.",
        "excluded_keys": _json_text(metrics.get("excluded_keys", [])),
    }


def _join_errors(mode: str, panel_data: Mapping[str, Any], raw_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in panel_data["errors"]:
        copied = dict(row)
        copied["mode"] = mode
        copied["mode_label"] = MODE_LABELS[mode]
        copied["raw_bundle_sha256"] = raw_summary["raw_bundle_sha256"]
        copied.setdefault("annotation_kind", "judge_annotation")
        rows.append(copied)
    for row in raw_summary.get("raw_pointwise_response_errors", []) if isinstance(raw_summary.get("raw_pointwise_response_errors"), list) else []:
        copied = dict(row)
        copied["mode"] = mode
        copied["mode_label"] = MODE_LABELS[mode]
        copied["raw_bundle_sha256"] = raw_summary["raw_bundle_sha256"]
        copied["judge_family"] = copied.get("judge_family") or raw_summary.get("judge_family", "")
        rows.append(copied)
    # Preserve row-local contract failures in the owner error table.  These
    # are derived audit annotations, not judge-authored error claims and never
    # alter the score, quote, or qualification evidence.
    raw_failure_by_key = {
        (str(detail.get("item_id") or ""), str(detail.get("candidate_id") or "")): detail
        for detail in raw_summary.get("raw_pointwise_contract_failures", [])
        if isinstance(detail, Mapping)
    }
    for (item_id, candidate_id), pointwise_row in sorted(panel_data["pointwise"].items()):
        eligible = _parse_bool(pointwise_row.get("score_eligible_for_mean")) is True
        shape_status = str(pointwise_row.get("score_shape_status") or "").upper()
        row_status = str(pointwise_row.get("status") or "").lower()
        if eligible and shape_status not in {"FAIL", "MISSING"} and row_status not in {"failed", "missing"}:
            continue
        raw_failure = raw_failure_by_key.get((item_id, candidate_id), {})
        copied = {
            "item_id": item_id,
            "source_id": pointwise_row.get("source_id", item_id),
            "candidate_id": candidate_id,
            "model_alias": pointwise_row.get("model_alias", ""),
            "judge_id": panel_data["judge_id"],
            "judge_family": pointwise_row.get("judge_family", raw_summary.get("judge_family", "")),
            "error_index": "raw_contract_failure",
            "severity": "",
            "type": "score_shape" if shape_status == "FAIL" else "missing_pointwise_row",
            "target_exact_quote": "",
            "source_evidence": raw_failure.get("raw_response_score_shape_issue") or pointwise_row.get("score_shape_issues") or pointwise_row.get("score_eligibility_issues") or f"row status is {row_status or 'missing'}",
            "source_span": "",
            "target_anchor": "",
            "confidence": "",
            "exact_quote_status": "",
            "source_text_sha256": pointwise_row.get("source_text_sha256", ""),
            "bundle": pointwise_row.get("bundle", ""),
            "mapping_scope": "owner_only_postjudge",
            "bundle_sha256": raw_summary["raw_bundle_sha256"],
            "annotation_kind": "raw_contract_failure",
            "raw_response_score": raw_failure.get("raw_response_score", ""),
            "raw_response_subscores": raw_failure.get("raw_response_subscores", ""),
            "raw_response_subscore_sum": raw_failure.get("raw_response_subscore_sum", ""),
            "raw_response_score_shape_issue": raw_failure.get("raw_response_score_shape_issue", "") or pointwise_row.get("score_eligibility_issues", ""),
            "raw_event_request_id": raw_failure.get("raw_event_request_id", ""),
            "raw_event_response_sha256": raw_failure.get("raw_event_response_sha256", ""),
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
        }
        rows.append(copied)
    return rows


def _join_pairwise(mode: str, panel_data: Mapping[str, Any], raw_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in panel_data["pairwise"]:
        copied = dict(row)
        copied["mode"] = mode
        copied["mode_label"] = MODE_LABELS[mode]
        copied["raw_bundle_sha256"] = raw_summary["raw_bundle_sha256"]
        copied["known_operational_evidence"] = "fresh independent request per orientation"
        rows.append(copied)
    return rows


def _mode_settings_text(summary: Mapping[str, Any]) -> str:
    settings = summary.get("request_settings", {})
    return _json_text(settings)


def _mode_track_caps_text(summary: Mapping[str, Any]) -> str:
    settings = summary.get("request_settings", {})
    caps = settings.get("max_tokens_by_track") if isinstance(settings, Mapping) else {}
    if not isinstance(caps, Mapping):
        return "n.a."
    order = ("capability", "qualification", "pointwise", "pairwise")
    parts = []
    for track in order:
        if track in caps:
            parts.append(f"{track}={caps[track]}")
    for track in sorted(str(key) for key in caps if str(key) not in order):
        parts.append(f"{track}={caps[track]}")
    return ", ".join(parts) if parts else "n.a."


def _panel_rank_map(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Mapping[str, Any]]:
    return {str(row["candidate_id"]): row for row in rows if row.get("mode") == mode}


def _report(
    path: Path,
    off_summary: Mapping[str, Any],
    on_summary: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    off_panel: Mapping[str, Any],
    on_panel: Mapping[str, Any],
    sensitivity: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    off_raw: Mapping[str, Any],
    on_raw: Mapping[str, Any],
    output_dir: Path,
) -> None:
    eligible = [row for row in paired if row.get("paired_score_status") == "paired_eligible"]
    off_valid = _parse_int(off_summary.get("pointwise_valid_rows")) or 0
    on_valid = _parse_int(on_summary.get("pointwise_valid_rows")) or 0
    on_primary_status = "ELIGIBLE" if on_valid == 21 else "NOT_ELIGIBLE_POINTWISE"
    deltas = [float(row["score_delta_on_minus_off"]) for row in eligible]
    changed = [row for row in eligible if row.get("score_changed") is True or str(row.get("score_changed")).lower() == "true"]
    off_counts = json.loads(off_summary.get("pointwise_score_value_counts") or "{}")
    on_counts = json.loads(on_summary.get("pointwise_score_value_counts") or "{}")
    off_unique = json.loads(off_summary.get("pointwise_score_unique_values") or "[]")
    on_unique = json.loads(on_summary.get("pointwise_score_unique_values") or "[]")
    on_contract_failures = on_raw.get("raw_pointwise_contract_failures", []) if isinstance(on_raw.get("raw_pointwise_contract_failures"), list) else []
    off_ref = _panel_rank_map(sensitivity, "off")
    on_ref = _panel_rank_map(sensitivity, "on")
    reliability_by_mode = {str(row["mode"]): row for row in reliability}
    on_budget_counts = on_raw.get("reasoning_budget_audit_counts") if isinstance(on_raw.get("reasoning_budget_audit_counts"), Mapping) else {}
    on_finish_reasons = on_raw.get("telemetry", {}).get("quality_finish_reason_counts", {}) if isinstance(on_raw.get("telemetry"), Mapping) else {}
    lines = [
        "# Nemotron thinking-off vs thinking-on comparison (2026-10-22)",
        "",
        "## 繁中摘要",
        "",
        "本報告把 Nemotron 同一模型的兩種 request 設定並列比較：thinking-off 與使用者指定重試的 thinking-on。兩者不是第六、七位獨立 judge，故各自與相同四位 judge 形成獨立的五 judge panel；絕不把兩份 Nemotron raw 合併進同一個 ICC、Kendall W 或 LOFO。",
        "",
        f"- thinking-off：{off_valid}/21 pointwise 可用，{off_counts.get('96', off_counts.get('96.0', 0))}/{off_valid or 21} 筆為 96，unique={off_unique}，SD(pop)={_fmt(off_summary.get('pointwise_score_sd_population'), 3)}；這是明顯壓縮/區辨有限，不代表品質或 judge 較好。",
        f"- thinking-on：{on_valid}/21 pointwise 可用，頻數={on_counts}，unique={on_unique}，SD(pop)={_fmt(on_summary.get('pointwise_score_sd_population'), 3)}；與 off 配對的 `{len(eligible)}/{len(paired)}` 個共同有效 units 中，{len(changed)} 筆分數改變，平均差(on−off)={_fmt(statistics.fmean(deltas) if deltas else None, 3)}。因 pointwise 未達 21/21，on 五 judge panel status=`{on_primary_status}`，僅作 diagnostic sensitivity。",
        f"- on raw contract audit：`{len(on_contract_failures)}` 筆 row-local failure；已保留原始 score/subscore mismatch（不補分、不改 quote），其 HTTP-successful response 的 judge error objects 另以 `raw_response_error` annotations 保存。",
        f"- qualification scalar：off `{off_summary.get('qualification_scalar_valid')}/{off_summary.get('qualification_variants')}` valid、`{off_summary.get('qualification_scalar_invalid')}` invalid；on `{on_summary.get('qualification_scalar_valid')}/{on_summary.get('qualification_variants')}` valid、`{on_summary.get('qualification_scalar_invalid')}` invalid。invalid 保留並排除 scalar summary，沒有手修 score/quote。",
        f"- AB/BA：off `{off_summary.get('pairwise_complete_swaps')}` complete、`{off_summary.get('pairwise_consistent')}` consistent；on `{on_summary.get('pairwise_complete_swaps')}` complete、`{on_summary.get('pairwise_consistent')}` consistent。兩者都是同一 source cluster 的 3 pairs/6 orientations，診斷樣本很小。",
        f"- pointwise latency median：off `{_fmt(off_summary.get('pointwise_latency_median_s'), 3)}s`，on `{_fmt(on_summary.get('pointwise_latency_median_s'), 3)}s`；token/failed/truncation 僅報 raw 有記錄的值。",
        "- 不能以分數較高、spread 較大或看似較一致來選 primary judge。thinking-on 是指定 retry 的暫定設定敏感度，off 仍保留為對照；本報告不聲稱正式 30 qualification / 40-item panel 已通過。",
        "- 本輪 joint 21/21 gate 仍為 FAIL；雖然 on 的 qualification/evidence、compression/discrimination 與 matched-20 ICC/W 出現探索性改善訊號，這只是 parent-reviewed tradeoff decision 的依據，不是 Nemo 正式合格或 correctness 證明。依使用者 conditional 授權，另開一次 Mistral thinking retry 作針對性探索；不會把它回填到本報告或自動選 judge。",
        "",
        "## Strict import and mode binding",
        "",
        f"- off validation: `{off_raw.get('validation_status')}`, raw SHA-256 `{off_raw.get('raw_bundle_sha256')}`",
        f"- on validation: `{on_raw.get('validation_status')}`, raw SHA-256 `{on_raw.get('raw_bundle_sha256')}`",
        f"- off request settings: `{_mode_settings_text(off_raw)}`",
        f"- on request settings: `{_mode_settings_text(on_raw)}`",
        f"- max-token caps by track (raw request evidence): off `{_mode_track_caps_text(off_raw)}`; on `{_mode_track_caps_text(on_raw)}`. The off JSON `max_tokens=1200` is only the first successful qualification request exemplar; it does not describe pointwise/AB/BA caps.",
        "- 本輪的 thinking-on/off 是 request chat-template setting 差異；`reasoning_format=deepseek` 只作 JSON/parser compatibility metadata，不表示把模型換成 DeepSeek。",
        "- paired interpretation limitation：off run 的 isolated context cap 是 8,192；on run 是 16,384 以容納 reasoning+JSON。這是重要的設定差異，不能把所有 on/off 差異歸因於 thinking alone。",
        "- raw bundle、packet/source hashes、exact source/target text 與 manifest binding 均保留；on 的單一 row-local pointwise shape failure 只允許作 partial diagnostic import，不能升格為完整 primary panel；其他 binding/structural validation failure 仍會使比較標為 notrun，而不是 fallback 或修分。",
        "- 預先寫定的探索性改善判讀規則：21/21 pointwise eligible、qualification 至少 off 的 17/24 scalar-valid、evidence support 至少 10/11、critical recall 至少 6/6、clean FP 0/11、AB/BA consistency 至少 2/3，並且壓縮/同分有實質改善或清楚記錄 tradeoff。這是本輪 decision rule，不是通用門檻；不滿足任一項即不宣稱改善已證實，也不因單一 spread 變大而擴大重試。",
        "- 判讀優先順序是 quality/evidence、paired discrimination 與 absolute-agreement ICC/W；latency 是次要。較慢不會單獨否決 on，但較高 ICC 也不等於 correctness，不會為了提高 ICC 刪 judge 或改 score；critical/evidence、score coherence 與合理同分需一起看。",
        "",
        "## Matched 21-unit pointwise comparison",
        "",
        f"Matched eligible units: `{len(eligible)}` / `{len(paired)}`; changed scores: `{len(changed)}`; unchanged: `{len(eligible) - len(changed)}`; mean delta on−off: `{_fmt(statistics.fmean(deltas) if deltas else None, 3)}`; median delta: `{_fmt(statistics.median(deltas) if deltas else None, 3)}`; population SD of delta: `{_fmt(statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None), 3)}`. The one ineligible on unit remains in the 21-row CSV with no imputed score.",
        "",
        "The paired CSV retains exact source/target text and source hash plus each mode's shape, quote, critical, error-count, eligibility, raw hash, and score. For an ineligible/missing pointwise row, validated `error_count` is intentionally blank with scope `unknown_ineligible_missing_pointwise`; any preserved HTTP-response errors appear separately as `raw_response_error_count` and in the owner annotation CSV. Missing or invalid values are not imputed, and a blank validated count must not be read as zero.",
        "",
        "## Score spread / discrimination reference",
        "",
        "Spread is descriptive only. Concentration can reflect a reasonable shared score, a compressed rubric response, or both; greater spread is not evidence of correctness. The preserved off-panel judge rows provide the same-scale reference below.",
        "",
        "| Mode / judge | n | Mean | SD(pop) | Unique values | Frequencies | Top ties (items/slots) |",
        "|---|---:|---:|---:|---|---|---:|",
        f"| Nemotron off | {off_summary.get('pointwise_valid_rows')} | {_fmt(off_summary.get('pointwise_mean'), 3)} | {_fmt(off_summary.get('pointwise_score_sd_population'), 3)} | `{off_unique}` | `{off_counts}` | {off_summary.get('pointwise_top_tie_items')} / {off_summary.get('pointwise_top_tie_slots')} |",
        f"| Nemotron on | {on_summary.get('pointwise_valid_rows')} | {_fmt(on_summary.get('pointwise_mean'), 3)} | {_fmt(on_summary.get('pointwise_score_sd_population'), 3)} | `{on_unique}` | `{on_counts}` | {on_summary.get('pointwise_top_tie_items')} / {on_summary.get('pointwise_top_tie_slots')} |",
    ]
    off_summary_path = Path(off_panel["panel"])/"judge_summary.fivejudges.owner.csv"
    if off_summary_path.exists():
        for row in _read_csv(off_summary_path):
            if "nemotron" in str(row.get("judge_id", "")).lower():
                continue
            try:
                unique = json.loads(row.get("pointwise_score_unique_values") or "[]")
                counts = json.loads(row.get("pointwise_score_value_counts") or "{}")
            except json.JSONDecodeError:
                unique, counts = [], {}
            lines.append(f"| Other off-panel `{row.get('judge_id')}` | {row.get('pointwise_valid_rows')} | {_fmt(_parse_float(row.get('pointwise_mean')), 3)} | {_fmt(_parse_float(row.get('pointwise_score_sd_population')), 3)} | `{unique}` | `{counts}` | {row.get('pointwise_top_tie_items')} / {row.get('pointwise_top_tie_slots')} |")
    lines += [
        "",
        "The reference illustrates the requested caveat: Mistral is also concentrated (unique values 3, SD around 4.24) while Gemini has a wider observed scale; neither spread nor concentration alone establishes correctness. Evidence/qualification/critical/false-positive and pairwise diagnostics remain necessary.",
        "",
        "## Qualification evidence diagnostics",
        "",
        "| Mode | Scalar valid / variants | Invalid | Mutation flag proxy | Evidence supported | Critical recall (union) | Clean false-positive | Error-type match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (off_summary, on_summary):
        def metric(name: str) -> str:
            value = row.get(f"qualification_{name}_estimate")
            numerator = row.get(f"qualification_{name}_numerator")
            denominator = row.get(f"qualification_{name}_denominator")
            return f"{_pct(value)} ({numerator}/{denominator})"
        lines.append(f"| {row.get('mode_label')} | {row.get('qualification_scalar_valid')}/{row.get('qualification_variants')} | {row.get('qualification_scalar_invalid')} | {metric('mutation_flag_rate_proxy')} | {metric('mutation_evidence_supported_rate')} | {metric('critical_severity_recall_union')} | {metric('clean_false_positive_rate')} | {metric('error_type_match')} |")
    lines += [
        "",
        "These are owner-side hidden-key diagnostics. A scalar mismatch is retained and excluded from scalar summaries; critical/evidence metrics are not inferred from the pointwise quality score.",
        "",
        "## AB/BA and position sensitivity",
        "",
        "The raw context flag is preserved. Both Nemotron runs have known operational evidence of fresh independent requests per orientation even if a raw boolean flag is absent; this is not treated as a larger sample. `3 complete; 1 consistent (33.3%)` means one of three complete swaps was consistent, not three of three.",
        "",
        "| Mode | Complete swaps | Consistent | Position-sensitive | Tie disagreement | Consistency | Raw context flag | Known operation |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        f"| thinking-off | {off_summary.get('pairwise_complete_swaps')} | {off_summary.get('pairwise_consistent')} | {off_summary.get('pairwise_position_sensitive')} | {off_summary.get('pairwise_tie_disagreement')} | {_pct(off_summary.get('pairwise_swap_consistency'))} | `{off_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        f"| thinking-on | {on_summary.get('pairwise_complete_swaps')} | {on_summary.get('pairwise_consistent')} | {on_summary.get('pairwise_position_sensitive')} | {on_summary.get('pairwise_tie_disagreement')} | {_pct(on_summary.get('pairwise_swap_consistency'))} | `{on_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        "",
        "## Latency, tokens, failed events, and truncation",
        "",
        "Only raw quality-request telemetry is summarized. Capability/template smoke events are not counted in the track rows; missing telemetry remains unavailable.",
        "",
        "| Mode | Track | Events (ok/failed/truncated) | Latency median / p95 (s) | Completion tokens median | Reasoning tokens median | Generation tok/s median |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for raw_summary in (off_raw, on_raw):
        telemetry = raw_summary["telemetry"]
        for track in TRACKS:
            metrics = telemetry.get(track, {})
            lines.append(f"| {raw_summary.get('mode_label')} | {track} | {metrics.get('event_count')} ({metrics.get('ok_events')}/{metrics.get('failed_events')}/{metrics.get('truncated_events')}) | {_fmt(metrics.get('latency_median_s'), 3)} / {_fmt(metrics.get('latency_p95_s'), 3)} | {_fmt(metrics.get('completion_tokens_median'), 1)} | {_fmt(metrics.get('reasoning_tokens_median'), 1)} | {_fmt(metrics.get('generation_tokens_per_s_median'), 2)} |")
    lines += [
        "",
        f"Quality-event totals: off failed `{off_raw['telemetry'].get('quality_failed_events')}`, truncated `{off_raw['telemetry'].get('quality_truncated_events')}`; on failed `{on_raw['telemetry'].get('quality_failed_events')}`, truncated `{on_raw['telemetry'].get('quality_truncated_events')}`. No failed/truncated values are recoded as zero.",
        "",
        "## Thinking budget / force-end accounting",
        "",
        f"The on request fixed `reasoning_budget_tokens={on_raw.get('request_settings', {}).get('reasoning_budget_tokens')}` and track caps `{_mode_track_caps_text(on_raw)}`. The full non-generating `/tokenize` audit covers `{on_budget_counts.get('events', 'n.a.')}` events (`tokenize_ok={on_budget_counts.get('tokenize_ok', 'n.a.')}`, failed={on_budget_counts.get('tokenize_failed', 'n.a.')}), with `{on_budget_counts.get('budget_hit_exact', 'n.a.')}` exact budget hits and `{on_budget_counts.get('budget_at_or_over', 'n.a.')}` at/over-budget candidates; audit SHA-256 `{on_raw.get('reasoning_budget_audit_sha256') or 'n.a.'}`. It counts retained `reasoning_content` only; `completion_tokens` includes reasoning plus final JSON and is not itself a budget-hit signal. The earlier two-sample probe is retained as `{on_raw.get('reasoning_budget_probe_confirmed_count', 0)} at/over and {on_raw.get('reasoning_budget_probe_below_count', 0)} below` (probe SHA-256 `{on_raw.get('reasoning_budget_probe_sha256') or 'n.a.'}`), but the 41-event audit is the fuller accounting.",
        "",
        f"The quality HTTP events separately report finish_reason counts `{on_finish_reasons}`; `stop` is an API completion status and does not prove natural reasoning completion or distinguish a budget force-end. The audit is non-generating and does not expose chain-of-thought. The comparison retains stop/truncation fields and does not relabel cap-hit responses or re-score them. The off run had thinking disabled, so no cross-mode reasoning-token split is imputed.",
        "",
        "## Separate five-judge panel sensitivity (exploratory)",
        "",
        "Each row below is computed separately with four unchanged judge bundles plus one Nemotron setting. It is not a six-judge result and not a selection rule.",
        "Candidate `valid N` and source coverage are shown because the failed on pointwise row makes one candidate available on only 14/15 judge observations and 2/3 source units. The displayed rank is therefore an exploratory available-case rank, not a claim of balanced 21-unit coverage; no missing score is treated as zero.",
        "",
        "| Candidate | Off rank / mean (valid N; sources) | On rank / mean (valid N; sources) | Rank delta (on−off) |",
        "|---|---|---|---:|",
    ]
    for candidate_id in sorted(set(off_ref) | set(on_ref)):
        off_row, on_row = off_ref.get(candidate_id, {}), on_ref.get(candidate_id, {})
        rank_delta = (int(on_row["rank"]) - int(off_row["rank"])) if off_row and on_row else None
        def coverage_text(row: Mapping[str, Any]) -> str:
            valid_scores = _parse_int(row.get("n_valid_scores"))
            distinct_judges = _parse_int(row.get("n_distinct_judges"))
            expected_sources = _parse_int(row.get("panel_source_units_expected")) or 3
            expected_observations = (distinct_judges * expected_sources) if distinct_judges is not None else None
            observed = f"{valid_scores}/{expected_observations}" if valid_scores is not None and expected_observations else "n.a."
            panel_observed = row.get("n_valid_source_units", "n.a.")
            nemotron_observed = row.get("nemotron_valid_source_units", "n.a.")
            return f"{observed} observed; panel {panel_observed}/{expected_sources}; Nemotron {nemotron_observed}/{_parse_int(row.get('nemotron_source_units_expected')) or expected_sources}"
        off_coverage = coverage_text(off_row)
        on_coverage = coverage_text(on_row)
        lines.append(f"| `{candidate_id}` | {off_row.get('rank', 'n.a.')} / {_fmt(_parse_float(off_row.get('score_mean')), 2)} ({off_coverage}) | {on_row.get('rank', 'n.a.')} / {_fmt(_parse_float(on_row.get('score_mean')), 2)} ({on_coverage}) | {_fmt(rank_delta)} |")
    lines += [
        "",
        "| Panel | ICC(2,1) | ICC(2,k) | actual k | units n | ICC CI | Kendall W | rankings n | LOFO |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for mode in ("off", "on"):
        row = reliability_by_mode.get(mode, {})
        lines.append(f"| {MODE_LABELS[mode]} ({row.get('primary_panel_status', 'n.a.')}) | {_fmt(_parse_float(row.get('icc_2_1')), 6)} | {_fmt(_parse_float(row.get('icc_2_k')), 6)} | {row.get('n_judges_actual', 'n.a.')} | {row.get('n_units_actual', 'n.a.')} | not available | {_fmt(_parse_float(row.get('kendall_w')), 6)} | {row.get('kendall_w_n_rankings', 'n.a.')} | `{row.get('lofo_status', 'n.a.')}` |")
    for mode in sorted(key for key in reliability_by_mode if key.startswith(("off_matched", "on_matched"))):
        row = reliability_by_mode[mode]
        lines.append(f"| {row.get('reliability_scope', mode)} | {_fmt(_parse_float(row.get('icc_2_1')), 6)} | {_fmt(_parse_float(row.get('icc_2_k')), 6)} | {row.get('n_judges_actual', 'n.a.')} | {row.get('n_units_actual', 'n.a.')} | not available | {_fmt(_parse_float(row.get('kendall_w')), 6)} | {row.get('kendall_w_n_rankings', 'n.a.')} | `{row.get('lofo_status', 'n.a.')}` |")
    lines += [
        "",
        "ICC CIs are not estimated by the existing implementation and are not invented here. Full on W has only 2 complete source-cluster rankings because the failed src-03/cand-02 row leaves that source's seven-candidate panel ranking incomplete. The matched common-20 rows restore 3 source-cluster rankings for both off and on, but src-03 then contains only the 6 candidates common to all five judges, not a full 7-candidate ranking. Thus matched W is a paired diagnostic comparison, not an assertion of full-source/global ranking validity. The matched rows remain diagnostic, not a repair of the missing on unit. LOFO is one-family/one-judge sensitivity with the explicit one-family-one-judge caveat; full JSON is in `nemotron_thinking_off_vs_on_panel_sensitivity.owner.csv` and each panel's aggregate.",
        "",
        "## Output and provenance",
        "",
        "- No `panel_off/` copy is made: the immutable existing `../fivejudge_20261022/` remains the off panel source.",
        "- `panel_on/` contains the newly built five-judge thinking-on report and owner CSVs; its raw input is the new thinking-on bundle only.",
        "- `nemotron_thinking_off_vs_on_paired.owner.csv` contains the matched 21-unit comparison with exact source/target text and source hash.",
        "- `nemotron_thinking_off_vs_on_error_annotations.owner.csv` and `nemotron_thinking_off_vs_on_pairwise.owner.csv` preserve both modes' annotations and orientation rows with mode/raw-hash bindings.",
        "- Any HTTP-successful but contract-invalid pointwise response keeps its raw judge error objects as `annotation_kind=raw_response_error`; the separate `raw_contract_failure` row records only the score-shape/eligibility failure. Neither annotation kind repairs or recodes the invalid score.",
        "- `nemotron_thinking_off_vs_on_judge_summary.owner.csv` contains the two mode rows; `nemotron_thinking_off_vs_on_panel_sensitivity.owner.csv` contains separate candidate ranking rows; `nemotron_thinking_off_vs_on_reliability.owner.csv` contains separate ICC/W/LOFO rows.",
        "- No raw score, quote, error, or invalid qualification scalar was edited. No LiteLLM or additional judge API was used by this comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(packet_path: Path, off_raw_path: Path, on_raw_path: Path, off_panel: Path, on_panel: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = read_json(packet_path)
    off_raw = read_json(off_raw_path)
    on_raw = read_json(on_raw_path)
    off_validation = validate_import(packet_path, off_raw_path)
    on_validation = validate_import(packet_path, on_raw_path)
    _write_json(output_dir / "validation.off.json", off_validation)
    _write_json(output_dir / "validation.on.json", on_validation)
    off_summary = _raw_mode_summary(off_raw_path, off_raw, "off", off_validation)
    on_summary = _raw_mode_summary(on_raw_path, on_raw, "on", on_validation)
    _write_json(output_dir / "raw_mode_summary.json", {"off": off_summary, "on": on_summary})
    for label, audit in (("off", off_validation), ("on", on_validation)):
        errors = audit.get("errors") if isinstance(audit.get("errors"), list) else []
        eligibility = audit.get("track_eligibility") if isinstance(audit.get("track_eligibility"), Mapping) else {}
        pointwise = eligibility.get("pointwise") if isinstance(eligibility.get("pointwise"), Mapping) else {}
        pairwise = eligibility.get("pairwise") if isinstance(eligibility.get("pairwise"), Mapping) else {}
        # The validator records a contract-invalid pointwise row as a partial
        # track.  In the completed Nemotron-on bundle that row is preserved in
        # the raw HTTP event (score=92, subscore sum=94), while the top-level
        # audit remains PASS and therefore has no top-level ``errors`` list.
        # Accept that explicitly row-local partial shape case here.  Do not
        # broaden this gate to arbitrary partial imports or repair/impute the
        # missing unit downstream.
        shape_only = bool(errors) and all(str(error).startswith("row ") and " pointwise score shape:" in str(error) for error in errors)
        partial_on_allowed = (
            label == "on"
            and audit.get("status") == "PASS"
            and (not errors or shape_only)
            and pointwise.get("status") == "partial"
            and int(pointwise.get("n_ok", 0) or 0) > 0
            and int(pointwise.get("n_rows", 0) or 0) >= int(pointwise.get("n_ok", 0) or 0)
            and pairwise.get("status") == "eligible"
        )
        strict_ok = audit.get("status") == "PASS" and not errors and pointwise.get("status") == "eligible" and pairwise.get("status") == "eligible"
        if not strict_ok and not partial_on_allowed:
            status = {
                "status": "NOTRUN_STRICT_IMPORT_GATE",
                "mode": label,
                "validation_status": audit.get("status"),
                "errors": errors,
                "pointwise": pointwise,
                "pairwise": pairwise,
                "partial_on_allowed": partial_on_allowed,
                "raw_bundle_sha256": sha256_file(on_raw_path if label == "on" else off_raw_path),
            }
            _write_json(output_dir / "comparison_status.json", status)
            raise ValueError(f"strict import gate failed for {label}: {status}")
    off_data = _mode_pointwise(off_panel, off_summary)
    on_data = _mode_pointwise(on_panel, on_summary)
    paired = _paired_rows(off_data, on_data, off_summary, on_summary)
    _write_csv(output_dir / "nemotron_thinking_off_vs_on_paired.owner.csv", paired, list(paired[0]))
    summary_rows = [_build_summary_row("off", off_summary, off_data), _build_summary_row("on", on_summary, on_data)]
    _write_csv(output_dir / "nemotron_thinking_off_vs_on_judge_summary.owner.csv", summary_rows, list(summary_rows[0]))
    errors = _join_errors("off", off_data, off_summary) + _join_errors("on", on_data, on_summary)
    if errors:
        union_fields: list[str] = []
        for row in errors:
            for field in row:
                if field not in {"mode", "mode_label"} and field not in union_fields:
                    union_fields.append(field)
        error_fields = ["mode", "mode_label"] + union_fields
        _write_csv(output_dir / "nemotron_thinking_off_vs_on_error_annotations.owner.csv", errors, error_fields)
    else:
        _write_csv(output_dir / "nemotron_thinking_off_vs_on_error_annotations.owner.csv", [], ["mode", "mode_label", "raw_bundle_sha256"])
    pairwise = _join_pairwise("off", off_data, off_summary) + _join_pairwise("on", on_data, on_summary)
    pair_fields = ["mode", "mode_label"] + [field for field in pairwise[0] if field not in {"mode", "mode_label"}] if pairwise else ["mode", "mode_label", "raw_bundle_sha256"]
    _write_csv(output_dir / "nemotron_thinking_off_vs_on_pairwise.owner.csv", pairwise, pair_fields)
    sensitivity = _panel_sensitivity_rows("off", off_data) + _panel_sensitivity_rows("on", on_data)
    _write_csv(output_dir / "nemotron_thinking_off_vs_on_panel_sensitivity.owner.csv", sensitivity, list(sensitivity[0]) if sensitivity else ["mode"])
    reliability = [_reliability_row("off", off_data), _reliability_row("on", on_data)]
    reliability[0]["primary_panel_status"] = "ELIGIBLE"
    reliability[1]["primary_panel_status"] = "NOT_ELIGIBLE_POINTWISE" if len([row for row in paired if row.get("on_score_eligible") is True]) < 21 else "ELIGIBLE"
    matched_keys = {
        (str(row.get("item_id") or ""), str(row.get("candidate_id") or ""))
        for row in paired
        if row.get("off_score_eligible") is True and row.get("on_score_eligible") is True
    }
    if matched_keys:
        matched_count = len(matched_keys)
        reliability.extend([
            _matched_reliability_row(f"off_matched{matched_count}", off_data, matched_keys, f"matched common eligible units (off baseline, n={matched_count})"),
            _matched_reliability_row(f"on_matched{matched_count}", on_data, matched_keys, f"matched common eligible units (on diagnostic, n={matched_count})"),
        ])
    _write_csv(output_dir / "nemotron_thinking_off_vs_on_reliability.owner.csv", reliability, list(reliability[0]))
    manifest = {
        "schema_version": "nemotron-thinking-comparison-v1",
        "packet_id": packet.get("packet_id"),
        "packet_sha256": sha256_file(packet_path),
        "off_raw": {"path": str(off_raw_path), "sha256": off_summary["raw_bundle_sha256"], "judge_id": off_summary["judge_id"]},
        "on_raw": {"path": str(on_raw_path), "sha256": on_summary["raw_bundle_sha256"], "judge_id": on_summary["judge_id"]},
        "off_panel": str(off_panel),
        "on_panel": str(on_panel),
        "paired_units": len(paired),
        "paired_eligible_units": len(matched_keys),
        "off_pointwise_primary_status": reliability[0]["primary_panel_status"],
        "on_pointwise_primary_status": reliability[1]["primary_panel_status"],
        "import_gate": {
            "off": "strict_eligible",
            "on": "partial_row_local_pointwise_shape_only" if reliability[1]["primary_panel_status"] != "ELIGIBLE" else "strict_eligible",
        },
        "same_model_two_settings_not_six_judges": True,
        "no_raw_mutation": True,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "nemotron_thinking_comparison_manifest.json", manifest)
    _report(output_dir / "nemotron_thinking_comparison.md", summary_rows[0], summary_rows[1], paired, off_data, on_data, sensitivity, reliability, off_summary, on_summary, output_dir)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "paired_rows": len(paired),
        "summary_rows": len(summary_rows),
        "error_rows": len(errors),
        "pairwise_rows": len(pairwise),
        "sensitivity_rows": len(sensitivity),
        "reliability_rows": len(reliability),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--off-raw", required=True, type=Path)
    parser.add_argument("--on-raw", required=True, type=Path)
    parser.add_argument("--off-panel", required=True, type=Path)
    parser.add_argument("--on-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_comparison(args.packet, args.off_raw, args.on_raw, args.off_panel, args.on_panel, args.output_dir)
    except Exception as exc:  # pragma: no cover - CLI failure is intentional evidence for strict gates
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

