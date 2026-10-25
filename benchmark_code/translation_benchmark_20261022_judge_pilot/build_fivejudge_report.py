#!/usr/bin/env python3
"""Build a versioned five-judge owner report from validated blind bundles.

This is an owner-side orchestration layer around the pilot's existing strict
validator, aggregate and CSV builders.  It never changes a raw bundle, score,
quote, packet, candidate mapping, or validation result.  All outputs are
written below a caller-supplied directory so the earlier three/four-judge
artifacts remain immutable.

The report intentionally keeps the pilot's distinctions:

* qualification is a hidden-key diagnostic track, not translation quality;
* invalid qualification scalar sums are retained and excluded from scalar
  summaries; and
* missing, failed, or unavailable timing/context evidence is reported as
  unavailable rather than imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aggregate_pilot import _pair_winner, aggregate, load_bundle, load_bundles, packet_index, read_json, write_json
from build_csv_results import _is_pointwise_shape_error, build as build_csv
from build_pairwise_diagnostics import build as build_pairwise
from qualification_summary import evaluate as evaluate_qualification
from validate_import import validate_import


SCRIPT_SCHEMA = "judge-pilot-fivejudge-report-v1"
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20261022
SCORE_KEYS = ("terminology_locale", "semantic_safety", "fluency", "completeness_format")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slug(value: Any) -> str:
    text = "".join(character if character.isalnum() else "_" for character in str(value or ""))
    text = "_".join(part for part in text.lower().split("_") if part)
    return text or "judge"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n.a."
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n.a."
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return "n.a."
    return f"{100.0 * float(value):.{digits}f}%"


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _metadata_judge(metadata: Mapping[str, Any]) -> dict[str, Any]:
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
    return {
        "judge_id": str(_first(judge, "judge_id", default=metadata.get("judge_id")) or ""),
        "family": str(_first(judge, "family", default=metadata.get("family")) or ""),
        "provider": _first(judge, "provider", default=metadata.get("provider")),
        "model": _first(judge, "model", "model_alias", default=metadata.get("model")),
        "ui_model_version": _first(judge, "ui_model_version", "model_label_from_ui", default=metadata.get("model_label_from_ui")),
    }


def _context_packaging(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    for source in (metadata, metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("judge_context_packaging", "context_packaging", "operational_context", "context"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if source.get("fresh_chat_per_source_or_orientation") is True:
            return "fresh_chat_per_source_or_orientation"
    family = str(_metadata_judge(metadata).get("family") or "").lower().replace("_", "-")
    provider = str(_metadata_judge(metadata).get("provider") or "").lower().replace("_", "-")
    if family in {"google-gemini", "google", "gemini"} or provider in {"gemini-ui", "gemini_ui", "google-gemini"}:
        return "fresh_chat_per_source_or_orientation"
    if family in {"gpt", "openai"}:
        return "fresh_full_packet_context"
    if family in {"qwen", "mistral", "nemotron"} or "nemotron" in family:
        return "fresh_request_per_candidate"
    return "not_recorded"


def _pairwise_context(metadata: Mapping[str, Any]) -> tuple[Any, Any]:
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), Mapping) else {}
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), Mapping) else {}
    fresh = _first(
        nested,
        "pairwise_fresh_chat_independence",
        default=_first(judge, "pairwise_fresh_chat_independence", default=_first(settings, "pairwise_fresh_chat_independence", default=metadata.get("pairwise_fresh_chat_independence"))),
    )
    limitation = _first(
        nested,
        "pairwise_limitation",
        default=_first(judge, "pairwise_limitation", default=_first(settings, "pairwise_limitation", default=metadata.get("pairwise_limitation"))),
    )
    return fresh, limitation


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]


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


def _score_distribution_values(
    scores: Sequence[float],
    top_by_item: Mapping[str, Sequence[tuple[str, float]]] | None = None,
) -> dict[str, Any]:
    finite_scores = _finite(scores)
    counts = Counter(f"{value:g}" for value in finite_scores)
    top_tie_items = top_tie_slots = strict_top_items = 0
    top_candidates: Counter[str] = Counter()
    if top_by_item:
        for values in top_by_item.values():
            if not values:
                continue
            maximum = max(score for _candidate, score in values)
            winners = [candidate for candidate, score in values if score == maximum]
            top_candidates.update(winners)
            if len(winners) > 1:
                top_tie_items += 1
                top_tie_slots += len(winners)
            else:
                strict_top_items += 1
    return {
        "n": len(finite_scores),
        "mean": statistics.fmean(finite_scores) if finite_scores else None,
        "sd_population": statistics.pstdev(finite_scores) if len(finite_scores) > 1 else 0.0 if finite_scores else None,
        "sd_sample": statistics.stdev(finite_scores) if len(finite_scores) > 1 else None,
        "unique_values": sorted(set(finite_scores)),
        "value_counts": dict(sorted(counts.items(), key=lambda pair: float(pair[0]))),
        "top_tie_items": top_tie_items,
        "top_tie_slots": top_tie_slots,
        "strict_top_items": strict_top_items,
        "top_candidate_counts": dict(sorted(top_candidates.items())),
    }


def _score_distribution_for_paths(paths: Sequence[Path]) -> dict[str, Any]:
    scores: list[float] = []
    for path in paths:
        _metadata, rows = load_bundle(path)
        scores.extend(
            float(row["_score"])
            for row in rows
            if row.get("_track") == "pointwise" and row.get("_status") == "ok" and isinstance(row.get("_score"), (int, float))
        )
    return _score_distribution_values(scores)


def _track_from_request_id(request_id: Any) -> str | None:
    value = str(request_id or "")
    for track in ("qualification", "pointwise", "pairwise"):
        if value.startswith(track):
            return track
    return None


def _timing_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract only recorded latency/throughput telemetry; never estimate gaps."""

    by_track: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen_timing_units: set[tuple[str, str]] = set()
    for row in rows:
        track = str(row.get("_track") or "")
        # Qualification rows are flattened to two variants per parent item;
        # latency is a parent request metric, so count each item once.
        if track == "qualification":
            unit = (track, str(row.get("_item_id") or ""))
            if unit in seen_timing_units:
                continue
            seen_timing_units.add(unit)
        timing = row.get("timings") if isinstance(row.get("timings"), Mapping) else {}
        elapsed = _first(timing, "api_elapsed_s", "elapsed_s", "latency_s")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(float(elapsed)):
            values: dict[str, float] = {"elapsed_s": float(elapsed)}
            for key in ("predicted_per_second", "generation_tokens_per_second", "tokens_per_second"):
                if isinstance(timing.get(key), (int, float)) and not isinstance(timing.get(key), bool) and math.isfinite(float(timing[key])):
                    values["generation_tok_s"] = float(timing[key])
                    break
            usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
            completion = _first(usage, "completion_tokens", "output_tokens")
            if isinstance(completion, (int, float)) and not isinstance(completion, bool) and math.isfinite(float(completion)):
                values["completion_tok_s"] = float(completion) / float(elapsed) if elapsed else float("nan")
            by_track[track].append(values)

    # Local runner records carry elapsed time, while the raw response events
    # carry generation speed and token counts.  Merge those fields by track
    # without double-counting an event as an additional quality observation.
    bundle = read_json(path)
    events = bundle.get("raw_events") if isinstance(bundle, Mapping) else None
    if isinstance(events, list):
        event_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        for event in events:
            if not isinstance(event, Mapping) or event.get("status") != "ok":
                continue
            track = _track_from_request_id(event.get("request_id"))
            if track is None:
                continue
            elapsed = event.get("elapsed_s")
            if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(float(elapsed)):
                continue
            response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
            timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
            values: dict[str, float] = {"elapsed_s": float(elapsed)}
            speed = timings.get("predicted_per_second")
            if isinstance(speed, (int, float)) and not isinstance(speed, bool) and math.isfinite(float(speed)):
                values["generation_tok_s"] = float(speed)
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            completion = _first(usage, "completion_tokens", "output_tokens")
            if isinstance(completion, (int, float)) and not isinstance(completion, bool) and math.isfinite(float(completion)):
                values["completion_tok_s"] = float(completion) / float(elapsed) if elapsed else float("nan")
            event_metrics[track].append(values)
        for track in ("qualification", "pointwise", "pairwise"):
            if not event_metrics.get(track):
                continue
            # Raw runner events include capability and template checks.  For
            # quality tracks, their request IDs are prefixed by the track and
            # the expected count is known from the imported quality rows.
            target_count = len(by_track.get(track, []))
            candidates = event_metrics[track]
            if target_count:
                candidates = candidates[-target_count:]
            if not by_track.get(track):
                by_track[track] = candidates
            else:
                # Preserve record elapsed values; enrich each positional
                # quality row with event-derived generation/throughput.
                for index, event_values in enumerate(candidates[:target_count]):
                    if index >= len(by_track[track]):
                        break
                    by_track[track][index].update({key: value for key, value in event_values.items() if key != "elapsed_s"})

    result: dict[str, Any] = {}
    for track in ("qualification", "pointwise", "pairwise"):
        values = by_track.get(track, [])
        elapsed = _finite(value.get("elapsed_s") for value in values)
        generation = _finite(value.get("generation_tok_s") for value in values)
        effective = _finite(value.get("completion_tok_s") for value in values)
        result[track] = {
            "n": len(values),
            "latency_median_s": statistics.median(elapsed) if elapsed else None,
            "latency_p95_s": _quantile(elapsed, 0.95),
            "generation_tok_s_median": statistics.median(generation) if generation else None,
            "generation_tok_s_p95": _quantile(generation, 0.95),
            "effective_completion_tok_s_median": statistics.median(effective) if effective else None,
            "effective_completion_tok_s_p95": _quantile(effective, 0.95),
            "status": "available" if elapsed else "unavailable",
            "note": None if elapsed else "No comparable per-request latency telemetry recorded for this judge bundle.",
        }
    return result


def _pointwise_judge_summary(rows: Sequence[Mapping[str, Any]], judge_id: str) -> dict[str, Any]:
    pointwise_rows = [row for row in rows if row.get("_track") == "pointwise" and row.get("_judge_id") == judge_id]
    scores = [float(row["_score"]) for row in pointwise_rows if row.get("_status") == "ok" and isinstance(row.get("_score"), (int, float))]
    top_by_item: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in pointwise_rows:
        if row.get("_status") == "ok" and isinstance(row.get("_score"), (int, float)):
            top_by_item[str(row.get("_item_id"))].append((str(row.get("_candidate_id")), float(row["_score"])))
    distribution = _score_distribution_values(scores, top_by_item)
    return {
        "n": len(scores),
        "mean": distribution["mean"],
        "median": statistics.median(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "distribution": distribution,
        "status_counts": dict(Counter(str(row.get("_status")) for row in pointwise_rows)),
        "error_annotations": sum(len(row.get("_errors", [])) for row in pointwise_rows if isinstance(row.get("_errors"), list)),
        "critical_flags": sum(1 for row in pointwise_rows if row.get("_critical") is True),
    }


def _qualification_shape(rows: Sequence[Mapping[str, Any]], judge_id: str) -> dict[str, Any]:
    values = [row for row in rows if row.get("_track") == "qualification" and row.get("_judge_id") == judge_id]
    statuses = Counter(str((row.get("_qualification_score_audit") or {}).get("status", "missing")) for row in values)
    return {"n": len(values), "valid": statuses.get("valid", 0), "invalid": statuses.get("invalid", 0), "missing": statuses.get("missing", 0), "abstain": statuses.get("abstain", 0), "status_counts": dict(sorted(statuses.items())), "scalar_status": "eligible" if statuses.get("invalid", 0) == 0 and statuses.get("missing", 0) == 0 and statuses.get("abstain", 0) == 0 and values else "partial"}


def _pairwise_counts(pairwise_rows: Sequence[Mapping[str, Any]], judge_id: str) -> dict[str, Any]:
    rows = [row for row in pairwise_rows if row.get("_judge_id") == judge_id and row.get("_status") == "ok"]
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        orientation = str(row.get("orientation") or "").upper()
        if orientation in {"AB", "BA"}:
            grouped[(str(row.get("_item_id")), str(row.get("pair_id")))][orientation] = row
    complete = consistent = position_sensitive = tie_disagreement = 0
    for orientations in grouped.values():
        if "AB" not in orientations or "BA" not in orientations:
            continue
        complete += 1
        ab, ba = orientations["AB"], orientations["BA"]
        first, second = _pair_winner(ab), _pair_winner(ba)
        if first == second:
            consistent += 1
        elif first != "TIE" and second != "TIE":
            position_sensitive += 1
        else:
            tie_disagreement += 1
    return {"rows": len(rows), "complete_swaps": complete, "consistent": consistent, "position_sensitive": position_sensitive, "tie_disagreement": tie_disagreement, "swap_consistency": consistent / complete if complete else None}


def _known_pairwise_operation(judge_id: str, summary: Mapping[str, Any] | None = None) -> str:
    """Report operational evidence separately from the raw context flag.

    The imported pairwise rows intentionally preserve ``not_recorded`` when
    the source bundle did not carry an explicit boolean.  The pilot's run
    evidence still records the known packaging, so the report can distinguish
    a missing raw flag from an assertion that the operation was unknown.
    """

    identity = str(judge_id or "").lower()
    family = str((summary or {}).get("judge_family") or "").lower()
    if "gemini" in identity or "gemini" in family:
        return "6 fresh UI chats (AB/BA operational evidence)"
    if "gpt" in identity or family in {"openai", "gpt"}:
        return "shared full-packet context"
    if any(token in identity or token in family for token in ("qwen", "mistral", "nemotron")):
        return "fresh independent request per orientation"
    return "not separately recorded"


def _judge_rows_and_summaries(
    input_paths: Sequence[Path],
    validations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in input_paths:
        metadata, rows = load_bundle(path)
        all_rows.extend(rows)
        judge = _metadata_judge(metadata)
        judge_id = judge["judge_id"] or (str(rows[0].get("_judge_id")) if rows else "")
        qualification = _qualification_shape(rows, judge_id)
        pointwise = _pointwise_judge_summary(rows, judge_id)
        pairwise = _pairwise_counts([row for row in rows if row.get("_track") == "pairwise"], judge_id)
        fresh, limitation = _pairwise_context(metadata)
        validation = validations[str(path)]
        track = validation.get("track_eligibility", {}) if isinstance(validation.get("track_eligibility"), Mapping) else {}
        qual_track = track.get("qualification", {}) if isinstance(track.get("qualification"), Mapping) else {}
        point_track = track.get("pointwise", {}) if isinstance(track.get("pointwise"), Mapping) else {}
        summaries.append({
            "judge_id": judge_id,
            "judge_family": judge.get("family", ""),
            "provider": judge.get("provider", ""),
            "model": judge.get("model", ""),
            "ui_model_version": judge.get("ui_model_version", ""),
            "bundle": str(path),
            "bundle_sha256": sha256_file(path),
            "packet_sha256": metadata.get("packet_sha256") or metadata.get("packet_hash") or "",
            "validation_status": validation.get("status"),
            "pointwise_track_status": point_track.get("status", ""),
            "pointwise_rows": validation.get("records", {}).get("tracks", {}).get("pointwise", pointwise["n"]),
            "pointwise_valid_rows": point_track.get("n_ok", pointwise["n"]),
            "pointwise_mean": pointwise["mean"],
            "pointwise_median": pointwise["median"],
            "pointwise_min": pointwise["min"],
            "pointwise_max": pointwise["max"],
            "pointwise_status_counts": json.dumps(pointwise["status_counts"], ensure_ascii=False, sort_keys=True),
            "pointwise_error_annotations": pointwise["error_annotations"],
            "pointwise_critical_flags": pointwise["critical_flags"],
            "pointwise_score_sd_population": pointwise["distribution"]["sd_population"],
            "pointwise_score_sd_sample": pointwise["distribution"]["sd_sample"],
            "pointwise_score_unique_values": json.dumps(pointwise["distribution"]["unique_values"], ensure_ascii=False),
            "pointwise_score_value_counts": json.dumps(pointwise["distribution"]["value_counts"], ensure_ascii=False, sort_keys=True),
            "pointwise_top_tie_items": pointwise["distribution"]["top_tie_items"],
            "pointwise_top_tie_slots": pointwise["distribution"]["top_tie_slots"],
            "pointwise_strict_top_items": pointwise["distribution"]["strict_top_items"],
            "pointwise_top_candidate_counts": json.dumps(pointwise["distribution"]["top_candidate_counts"], ensure_ascii=False, sort_keys=True),
            "pointwise_distribution": pointwise["distribution"],
            "qualification_variants": qualification["n"],
            "qualification_scalar_valid": qualification["valid"],
            "qualification_scalar_invalid": qualification["invalid"],
            "qualification_scalar_missing": qualification["missing"],
            "qualification_scalar_abstain": qualification["abstain"],
            "qualification_scalar_status": qual_track.get("scalar_score_status", qualification["scalar_status"]),
            "qualification_evidence_critical_status": qual_track.get("evidence_critical_status", ""),
            "qualification_scalar_status_counts": json.dumps(qualification["status_counts"], ensure_ascii=False, sort_keys=True),
            "qualification_error_annotations": sum(len(row.get("_errors", [])) for row in rows if row.get("_track") == "qualification" and isinstance(row.get("_errors"), list)),
            "qualification_critical_flags": sum(1 for row in rows if row.get("_track") == "qualification" and row.get("_critical") is True),
            "pairwise_rows": pairwise["rows"],
            "pairwise_complete_swaps": pairwise["complete_swaps"],
            "pairwise_consistent": pairwise["consistent"],
            "pairwise_position_sensitive": pairwise["position_sensitive"],
            "pairwise_tie_disagreement": pairwise["tie_disagreement"],
            "pairwise_swap_consistency": pairwise["swap_consistency"],
            "pairwise_fresh_chat_independence": fresh,
            "pairwise_limitation": limitation or "",
            "judge_context_packaging": _context_packaging(metadata, rows),
            "timing": _timing_summary(path, rows),
            "mapping_scope": "owner_only_postjudge",
        })
    return all_rows, {summary["judge_id"]: summary for summary in summaries}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_judge_summary_csv(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fields = [
        "judge_id", "judge_family", "provider", "model", "ui_model_version", "bundle", "bundle_sha256", "packet_sha256",
        "validation_status", "pointwise_track_status", "pointwise_rows", "pointwise_valid_rows", "pointwise_mean", "pointwise_median", "pointwise_min", "pointwise_max", "pointwise_status_counts", "pointwise_error_annotations", "pointwise_critical_flags", "pointwise_score_sd_population", "pointwise_score_sd_sample", "pointwise_score_unique_values", "pointwise_score_value_counts", "pointwise_top_tie_items", "pointwise_top_tie_slots", "pointwise_strict_top_items", "pointwise_top_candidate_counts",
        "qualification_variants", "qualification_scalar_valid", "qualification_scalar_invalid", "qualification_scalar_missing", "qualification_scalar_abstain", "qualification_scalar_status", "qualification_evidence_critical_status", "qualification_scalar_status_counts", "qualification_error_annotations", "qualification_critical_flags",
        "pairwise_rows", "pairwise_complete_swaps", "pairwise_consistent", "pairwise_position_sensitive", "pairwise_tie_disagreement", "pairwise_swap_consistency", "pairwise_fresh_chat_independence", "pairwise_limitation", "judge_context_packaging",
        "qualification_latency_median_s", "qualification_latency_p95_s", "pointwise_latency_median_s", "pointwise_latency_p95_s", "pairwise_latency_median_s", "pairwise_latency_p95_s",
        "qualification_generation_tok_s_median", "pointwise_generation_tok_s_median", "pairwise_generation_tok_s_median", "qualification_effective_completion_tok_s_median", "pointwise_effective_completion_tok_s_median", "pairwise_effective_completion_tok_s_median",
        "timing_status", "timing_note", "mapping_scope",
    ]
    rows: list[dict[str, Any]] = []
    for judge_id, summary in sorted(summaries.items()):
        timing = summary.get("timing", {})
        row = {key: summary.get(key, "") for key in fields}
        for track in ("qualification", "pointwise", "pairwise"):
            values = timing.get(track, {}) if isinstance(timing, Mapping) else {}
            row[f"{track}_latency_median_s"] = values.get("latency_median_s", "")
            row[f"{track}_latency_p95_s"] = values.get("latency_p95_s", "")
            row[f"{track}_generation_tok_s_median"] = values.get("generation_tok_s_median", "")
            row[f"{track}_effective_completion_tok_s_median"] = values.get("effective_completion_tok_s_median", "")
        statuses = [timing.get(track, {}).get("status") for track in ("qualification", "pointwise", "pairwise") if isinstance(timing.get(track), Mapping)]
        row["timing_status"] = "available" if any(status == "available" for status in statuses) else "unavailable"
        notes = [timing.get(track, {}).get("note") for track in ("qualification", "pointwise", "pairwise") if isinstance(timing.get(track), Mapping) and timing.get(track, {}).get("note")]
        row["timing_note"] = " | ".join(notes)
        rows.append(row)
    _write_csv(path, rows, fields)


def _augment_csv_hashes(path: Path, packet: Mapping[str, Any], bundle_hashes: Mapping[str, str], *, add_bundle_hash: bool = True) -> None:
    """Append audit hashes to a newly-created owner CSV without touching values."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    requested = ["source_text_sha256"] + (["bundle_sha256"] if add_bundle_hash else [])
    additions = [field for field in requested if field not in fields]
    if not additions:
        return
    for row in rows:
        if "source_text_sha256" in additions:
            item_id = str(row.get("item_id") or "")
            source_hash = str(packet_index(packet)["source_hashes"].get(item_id) or "")
            if not source_hash:
                for item in packet_index(packet)["items"].values():
                    if isinstance(item, Mapping) and str(item.get("item_id") or item.get("id") or item.get("source_id") or "") == item_id:
                        source = item.get("source_en") or item.get("source_text") or item.get("source")
                        if isinstance(source, str):
                            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
                        break
            row["source_text_sha256"] = source_hash
        if "bundle_sha256" in additions:
            bundle = row.get("bundle")
            row["bundle_sha256"] = bundle_hashes.get(str(bundle), "")
    _write_csv(path, rows, fields + additions)


def _candidate_table_rows(aggregate_result: Mapping[str, Any], mapping_path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    mapping_doc = read_json(mapping_path)
    mapping = mapping_doc.get("mapping", {}) if isinstance(mapping_doc, Mapping) else {}
    pointwise = aggregate_result.get("pointwise", {}) if isinstance(aggregate_result.get("pointwise"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for candidate_id, result in sorted(pointwise.items()):
        candidate = mapping.get(candidate_id, {}) if isinstance(mapping, Mapping) else {}
        rows.append({
            "candidate_id": candidate_id,
            "model_alias": candidate.get("model_alias", ""),
            "candidate_family": candidate.get("candidate_family", ""),
            "score_mean": result.get("score_mean"),
            "score_median": result.get("score_median"),
            "score_iqr": result.get("score_iqr"),
            "score_mean_family_balanced": (result.get("score_mean_family_balanced") or {}).get("estimate"),
            "n_valid_scores": result.get("n_valid_scores"),
            "n_distinct_judges": result.get("n_distinct_judges"),
            "deployment_status": result.get("deployment_status"),
        })
    return rows


def _judge_candidate_means(input_paths: Sequence[Path], mapping_path: Path) -> dict[str, dict[str, float | None]]:
    mapping_doc = read_json(mapping_path)
    mapping = mapping_doc.get("mapping", {}) if isinstance(mapping_doc, Mapping) else {}
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in input_paths:
        metadata, rows = load_bundle(path)
        judge = _metadata_judge(metadata)
        judge_id = judge["judge_id"] or (str(rows[0].get("_judge_id")) if rows else "")
        for row in rows:
            if row.get("_track") != "pointwise" or row.get("_status") != "ok" or row.get("_score") is None:
                continue
            candidate = str(row.get("_candidate_id"))
            values[candidate][judge_id].append(float(row["_score"]))
    result: dict[str, dict[str, float | None]] = {}
    for candidate, by_judge in values.items():
        result[candidate] = {judge: statistics.fmean(scores) if scores else None for judge, scores in by_judge.items()}
    return result


def _four_vs_five_rows(four_path: Path, five: Mapping[str, Any], mapping_path: Path) -> list[dict[str, Any]]:
    four = read_json(four_path)
    four_point = four.get("pointwise", {}) if isinstance(four, Mapping) else {}
    five_point = five.get("pointwise", {}) if isinstance(five, Mapping) else {}
    candidates = sorted(set(four_point) | set(five_point))
    def rank(point: Mapping[str, Any]) -> dict[str, int]:
        ordered = sorted(point, key=lambda candidate: (-float(point[candidate].get("score_mean") or float("-inf")), str(candidate)))
        return {candidate: index + 1 for index, candidate in enumerate(ordered)}
    four_rank, five_rank = rank(four_point), rank(five_point)
    mapping_doc = read_json(mapping_path)
    mapping = mapping_doc.get("mapping", {}) if isinstance(mapping_doc, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        alias = mapping.get(candidate, {}).get("model_alias", "") if isinstance(mapping, Mapping) else ""
        rows.append({
            "candidate_id": candidate,
            "model_alias": alias,
            "rank_4judge": four_rank.get(candidate),
            "rank_5judge": five_rank.get(candidate),
            "rank_delta_5_minus_4": (five_rank.get(candidate) - four_rank.get(candidate)) if candidate in four_rank and candidate in five_rank else None,
            "top1_4judge": bool(four_rank.get(candidate) == 1),
            "top1_5judge": bool(five_rank.get(candidate) == 1),
            "mean_4judge": (four_point.get(candidate) or {}).get("score_mean"),
            "mean_5judge": (five_point.get(candidate) or {}).get("score_mean"),
        })
    return rows


def _markdown_report(
    output_path: Path,
    packet_path: Path,
    instructions_path: Path,
    mapping_path: Path,
    input_paths: Sequence[Path],
    validations: Mapping[str, Mapping[str, Any]],
    aggregate_result: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    pairwise_summary: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    four_aggregate_path: Path | None,
    output_dir: Path,
) -> None:
    packet = read_json(packet_path)
    packet_hash = sha256_file(packet_path)
    instructions_hash = sha256_file(instructions_path)
    mapping_hash = sha256_file(mapping_path)
    scope = aggregate_result.get("sample_scope", {})
    judge_scope = aggregate_result.get("judge_scope", {})
    reliability = aggregate_result.get("reliability_overall", {})
    icc = reliability.get("icc_2_1_2_k", {}) if isinstance(reliability, Mapping) else {}
    kendall = reliability.get("kendall_w", {}) if isinstance(reliability, Mapping) else {}
    four_vs_five = _four_vs_five_rows(four_aggregate_path, aggregate_result, mapping_path) if four_aggregate_path and four_aggregate_path.exists() else []
    candidate_rows = _candidate_table_rows(aggregate_result, mapping_path, summaries)
    nemotron_judge_id = next((judge_id for judge_id in summaries if "nemotron" in judge_id.lower()), None)
    nemotron_summary = summaries.get(nemotron_judge_id, {}) if nemotron_judge_id else {}
    nemotron_path = next((path for path in input_paths if _metadata_judge(load_bundle(path)[0])["judge_id"] == nemotron_judge_id), None)
    nemotron_metadata = load_bundle(nemotron_path)[0] if nemotron_path else {}
    nemotron_settings = nemotron_metadata.get("settings") if isinstance(nemotron_metadata.get("settings"), Mapping) else {}
    nemotron_kwargs = nemotron_settings.get("chat_template_kwargs") if isinstance(nemotron_settings.get("chat_template_kwargs"), Mapping) else {}
    thinking_enabled = nemotron_kwargs.get("enable_thinking")
    if thinking_enabled is None:
        thinking_enabled = "thinking-on" in str(nemotron_judge_id or "").lower()
    nemotron_mode = "thinking-on" if thinking_enabled is True else "thinking-off"
    try:
        nemotron_unique = json.loads(nemotron_summary.get("pointwise_score_unique_values") or "[]")
        nemotron_counts = json.loads(nemotron_summary.get("pointwise_score_value_counts") or "{}")
    except json.JSONDecodeError:
        nemotron_unique, nemotron_counts = [], {}
    nemotron_point_valid = nemotron_summary.get("pointwise_valid_rows", 0)
    nemotron_point_rows = nemotron_summary.get("pointwise_rows", 0)
    nemotron_qual_valid = nemotron_summary.get("qualification_scalar_valid", 0)
    nemotron_qual_invalid = nemotron_summary.get("qualification_scalar_invalid", 0)
    rank_unchanged = bool(four_vs_five) and all(row.get("rank_delta_5_minus_4") == 0 for row in four_vs_five)

    lines: list[str] = []
    lines += [
        "# Five-judge translation-quality pilot report",
        "",
        "Date: 2026-10-22 (Asia/Taipei)",
        "",
        "Status: **PROVISIONAL_PILOT**. This owner-only report adds the Nemotron bundle to the existing four validated judge bundles. It is descriptive, non-clinical, and not a deployment approval or a translation gold standard.",
        "",
        "## 繁中摘要",
        "",
        f"Nemotron `{nemotron_mode}` 在 `{nemotron_point_valid}/{nemotron_point_rows}` 個 pointwise 單位可用，觀察 unique={nemotron_unique}、頻數={nemotron_counts}；分數區辨與缺失狀態都如實保留，量尺分散不是翻譯品質或 judge 優越性的證據。本輪明確 `enable_thinking={str(bool(thinking_enabled)).lower()}`、`preserve_reasoning={str(nemotron_kwargs.get('preserve_reasoning', False)).lower()}`；`reasoning_format=deepseek` 只是 parser compatibility。其 qualification 有 `{nemotron_qual_invalid}/{nemotron_summary.get('qualification_variants', 0)}` 筆 scalar sum invalid，保留原值並排除於 scalar summary。AB/BA 有 `{nemotron_summary.get('pairwise_complete_swaps', 0)}` 組完整 swap，其中 `{nemotron_summary.get('pairwise_consistent', 0)}` 組一致；pointwise latency median 為 `{fmt((nemotron_summary.get('timing') or {}).get('pointwise', {}).get('latency_median_s') if isinstance(nemotron_summary.get('timing'), Mapping) else None)} 秒。加入 Nemotron 後五 judge 排名相對四 judge {'未變' if rank_unchanged else '有變或無法比較'}；ICC(2,1)={fmt(icc.get('icc_2_1'), 3)}、ICC(2,{icc.get('n_judges', 'k')})={fmt(icc.get('icc_2_k'), 3)}、Kendall W={fmt(kendall.get('estimate'), 3)}，但僅 3 個 source clusters，且仍非正式 30/40-item panel。",
        "",
        "## Scope and binding",
        "",
        f"- Distinct judge IDs observed: `{judge_scope.get('n_distinct_judges', 0)}`; required for the configured panel-count gate: `{judge_scope.get('required_for_auto_pass', 5)}`; aggregate count flag: `{str(judge_scope.get('formal_panel', False)).lower()}`; **formal protocol status: NOT_FORMAL**. This remains the 21 source×candidate-unit / 3-source-cluster pilot with 12 qualification parents / 24 variants per judge, not the planned 30/40-item protocol. A count gate is not a qualification pass or deployment approval.",
        f"- Frozen packet: [`judge_packet.json`](../judge_packet.json), SHA-256 `{packet_hash}`; packet ID `{packet.get('packet_id')}`.",
        f"- Shared instructions SHA-256 `{instructions_hash}`; owner mapping SHA-256 `{mapping_hash}` (mapping is never judge-visible).",
        f"- Pointwise scope: `{scope.get('n_pointwise_records_raw', 0)}` raw rows / `{scope.get('n_pointwise_units_collapsed', 0)}` collapsed units over `{scope.get('n_source_clusters', 0)}` source clusters (`{', '.join(scope.get('source_ids', []))}`).",
        "- Qualification scope: 12 parent items and 24 anonymous variants per complete judge bundle. Qualification labels and expected mutations remain private; scalar validity, mutation evidence, and critical recall are separate diagnostics.",
        "- Pairwise scope: 3 fixed pairs × AB/BA = 6 orientations per judge, all from the frozen same-source pair bank. A disconnected graph does not support a seven-candidate global pairwise ranking.",
        "",
        f"Every input passed strict packet, source-hash, candidate, duplicate, orientation, and exact-target-quote binding unless explicitly shown below. The Nemotron `{nemotron_mode}` import is track-aware: pointwise `{nemotron_point_valid}/{nemotron_point_rows}` eligible (`{nemotron_summary.get('pointwise_track_status', '')}`), pairwise `{nemotron_summary.get('pairwise_rows', 0)}/{nemotron_summary.get('pairwise_rows', 0)}` rows imported, and qualification scalar `{nemotron_qual_invalid}/{nemotron_summary.get('qualification_variants', 0)}` invalid with `{nemotron_qual_valid}/{nemotron_summary.get('qualification_variants', 0)}` valid. Invalid/missing rows and qualification scalar sums are retained in raw imports and excluded only from eligible summaries; no score or quote was repaired by this report.",
        "",
        "## Judge validity, context, and latency",
        "",
        "Latency medians and p95 values are shown only when the raw bundle records comparable per-request telemetry. GPT/Codex and Gemini UI timing are unavailable or operationally different; those cells remain `n.a.`. Context packaging is recorded because GPT used one full-packet context, Gemini used UI conversations, and local/API judges used fresh request contexts. These are not matched-context quality comparisons. Nemotron used the exact frozen template with request-only `reasoning_format: \"deepseek\"` compatibility metadata plus `response_format: {\"type\":\"json_object\"}`; the initial JSON smoke incompatibility and the subsequent strict-JSON compatibility smoke are preserved in the adjacent raw run directory, not silently discarded.",
        "",
        "| Judge | Family / provider | Validation | Pointwise | Pointwise errors | Qualification scalar | Pairwise swaps | Context | Latency availability |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for judge_id, summary in sorted(summaries.items()):
        timing = summary.get("timing", {})
        point = timing.get("pointwise", {}) if isinstance(timing, Mapping) else {}
        latency = f"{fmt(point.get('latency_median_s'))} / {fmt(point.get('latency_p95_s'))} s" if point.get("status") == "available" else "n.a."
        pair = f"{summary.get('pairwise_complete_swaps', 0)} complete; {summary.get('pairwise_consistent', 0)} consistent ({pct(summary.get('pairwise_swap_consistency'))})"
        lines.append(f"| `{judge_id}` | `{summary.get('judge_family', '')}` / `{summary.get('provider', '')}` | `{summary.get('validation_status')}` | {summary.get('pointwise_valid_rows', 0)}/{summary.get('pointwise_rows', 0)}; mean {fmt(summary.get('pointwise_mean'), 2)} | {summary.get('pointwise_error_annotations', 0)} annotations; {summary.get('pointwise_critical_flags', 0)} critical flags | {summary.get('qualification_scalar_valid', 0)}/{summary.get('qualification_variants', 0)} valid; `{summary.get('qualification_scalar_status')}` | {pair} | `{summary.get('judge_context_packaging')}` | {latency} |")
    lines += [
        "",
        "Raw bundle SHA-256 and per-track latency details are in `judge_summary.fivejudges.owner.csv`; validation JSON files are adjacent in this directory. The raw source bundles remain under `../raw/` and were not overwritten.",
        "",
        "## Qualification diagnostics",
        "",
        "Qualification pass/fail is not inferred from judge output in the main aggregate. The private-key owner diagnostic below keeps mutation flag proxy, conservative evidence support, clean false-positive rate, critical-severity recall, and scalar shape validity distinct.",
        "",
        "| Judge | Scalar valid / checked | Mutation flag proxy | Evidence supported | Critical recall | Clean false-positive | Error-type match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    per_qual = qualification_result.get("per_judge", {}) if isinstance(qualification_result, Mapping) else {}
    for judge_id, summary in sorted(summaries.items()):
        report = per_qual.get(judge_id, {}) if isinstance(per_qual, Mapping) else {}
        def qmetric(path: Sequence[str]) -> Any:
            value: Any = report
            for key in path:
                value = value.get(key) if isinstance(value, Mapping) else None
            return value
        lines.append(f"| `{judge_id}` | {summary.get('qualification_scalar_valid', 0)}/{summary.get('qualification_variants', 0)} | {pct(qmetric(('mutation_flag_rate_proxy', 'estimate')))} ({qmetric(('mutation_flag_rate_proxy', 'numerator'))}/{qmetric(('mutation_flag_rate_proxy', 'denominator'))}) | {pct(qmetric(('mutation_evidence_supported_rate', 'estimate')))} ({qmetric(('mutation_evidence_supported_rate', 'numerator'))}/{qmetric(('mutation_evidence_supported_rate', 'denominator'))}) | {pct(qmetric(('critical_severity_recall', 'union_top_level_or_error_severity', 'estimate')))} ({qmetric(('critical_severity_recall', 'union_top_level_or_error_severity', 'numerator'))}/{qmetric(('critical_severity_recall', 'union_top_level_or_error_severity', 'denominator'))}) | {pct(qmetric(('clean_false_positive_rate', 'estimate')))} ({qmetric(('clean_false_positive_rate', 'numerator'))}/{qmetric(('clean_false_positive_rate', 'denominator'))}) | {pct(qmetric(('error_type_match', 'estimate')))} ({qmetric(('error_type_match', 'numerator'))}/{qmetric(('error_type_match', 'denominator'))}) |")
    lines += [
        "",
        "The private key remains read-only and is not included in judge-visible materials. A missing/failed/abstained qualification variant is not a zero; a scalar mismatch is not silently recomputed.",
        "",
        "## Five-judge candidate pointwise summary",
        "",
        "Scores are exploratory pointwise means over the 21 frozen source×candidate units. The family-balanced mean gives each operational family equal weight, then each judge equal weight inside its family. With one judge per family in this pilot it is a descriptive sensitivity, not a correction for lineage or dependence. Source-cluster bootstrap CIs in the aggregate are descriptive and unstable because `n_source_clusters=3`. Aggregate labels such as `AUTO_PASS_FOR_BENCHMARK` are the existing exploratory score rule; they do not certify the formal panel or the planned protocol.",
        "",
        "| Candidate | Family | 5-judge mean | Median | IQR | Family-balanced mean | Valid scores | Judges | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(candidate_rows, key=lambda value: (-(float(value["score_mean"]) if isinstance(value.get("score_mean"), (int, float)) else float("-inf")), str(value["candidate_id"]))):
        lines.append(f"| `{row['model_alias']}` (`{row['candidate_id']}`) | `{row['candidate_family']}` | {fmt(row.get('score_mean'), 2)} | {fmt(row.get('score_median'), 2)} | {fmt(row.get('score_iqr'), 2)} | {fmt(row.get('score_mean_family_balanced'), 2)} | {row.get('n_valid_scores', 0)} | {row.get('n_distinct_judges', 0)} | `{row.get('deployment_status')}` |")
    lines += ["", "Per-judge candidate means and exact source/target text are in `translation_comparison.fivejudges.owner.csv`; the machine-readable long form is `pointwise_results_long.fivejudges.owner.csv`.", "", "## Score distribution and discrimination check", "", f"Score spread is reported because agreement and candidate means can conceal a compressed judge scale. Standard deviation below is population SD over the observed eligible pointwise rows for that judge/pool; it is descriptive, not a measurement-error estimate. `Top-tie items/slots` counts source items whose highest score was shared, and the number of tied candidate slots. Nemotron `{nemotron_mode}` observed `{nemotron_counts}` over `{nemotron_point_valid}/{nemotron_point_rows}` eligible rows; this is a scale/discrimination diagnostic, not evidence that the judge or translations are intrinsically superior."]
    five_distribution = _score_distribution_for_paths(input_paths)
    four_paths: list[Path] = []
    for path in input_paths:
        metadata, _rows = load_bundle(path)
        judge = _metadata_judge(metadata)
        identity = f"{judge.get('judge_id', '')} {judge.get('family', '')}".lower()
        if "nemotron" not in identity:
            four_paths.append(path)
    four_distribution = _score_distribution_for_paths(four_paths)
    def counts_text(distribution: Mapping[str, Any]) -> str:
        values = distribution.get("value_counts", {}) if isinstance(distribution, Mapping) else {}
        return ", ".join(f"{key}×{value}" for key, value in values.items()) or "n.a."
    lines += [
        "",
        "| Pool / judge | n | Mean | SD(pop) | Unique values | Frequencies | Top-tie items / slots |",
        "|---|---:|---:|---:|---|---|---:|",
        f"| 4-judge pooled baseline | {four_distribution.get('n', 0)} | {fmt(four_distribution.get('mean'), 3)} | {fmt(four_distribution.get('sd_population'), 3)} | {', '.join(fmt(value, 0) for value in four_distribution.get('unique_values', [])) or 'n.a.'} | {counts_text(four_distribution)} | n.a. |",
    ]
    lines.append(f"| 5-judge pooled | {five_distribution.get('n', 0)} | {fmt(five_distribution.get('mean'), 3)} | {fmt(five_distribution.get('sd_population'), 3)} | {', '.join(fmt(value, 0) for value in five_distribution.get('unique_values', [])) or 'n.a.'} | {counts_text(five_distribution)} | n.a. |")
    for judge_id, summary in sorted(summaries.items()):
        distribution = summary.get("pointwise_distribution", {})
        lines.append(f"| `{judge_id}` | {summary.get('pointwise_rows', 0)} | {fmt(summary.get('pointwise_mean'), 3)} | {fmt(summary.get('pointwise_score_sd_population'), 3)} | {', '.join(fmt(value, 0) for value in distribution.get('unique_values', [])) or 'n.a.'} | {counts_text(distribution)} | {summary.get('pointwise_top_tie_items', 0)} / {summary.get('pointwise_top_tie_slots', 0)} |")
    lines += ["", "The pooled five-judge change is a sensitivity result; it inherits the additional Nemotron compression and does not imply that the added judge is independent of source difficulty or a correctness oracle.", ""]

    lines += [
        "## Reliability and ranking sensitivity",
        "",
        f"- ICC(2,1) = `{fmt(icc.get('icc_2_1'), 6)}`; ICC(2,k) = `{fmt(icc.get('icc_2_k'), 6)}` with actual observed `k={icc.get('n_judges', 'n.a.')}` and `{icc.get('n_units', 'n.a.')} complete units`. The aggregate implementation does not estimate an ICC confidence interval; CI is therefore **not available**, not filled from the earlier four-judge run.",
        f"- Kendall W = `{fmt(kendall.get('estimate'), 6)}` over `{kendall.get('n_rankings', 'n.a.')}` complete source/item rankings. This is concordance, not correctness or source-independent validity.",
        f"- Krippendorff nominal error-type alpha: `{fmt((reliability.get('krippendorff_alpha_nominal_error_type') or {}).get('estimate'), 6)}`; ordinal severity alpha: `{fmt((reliability.get('krippendorff_alpha_ordinal_severity') or {}).get('estimate'), 6)}`. Open-label error strings are not a validated MQM taxonomy.",
        f"- Gwet AC1 critical: `{fmt((reliability.get('gwet_ac1_critical') or {}).get('estimate'), 6)}`; AC2 severity: `{fmt((reliability.get('gwet_ac2_severity') or {}).get('estimate'), 6)}`. If all critical labels are negative, positive agreement is undefined and is not a safety claim.",
        f"- Span agreement: `{(reliability.get('span_agreement') or {}).get('status', 'n.a.')}`. Missing target coordinate spans remain missing and are not treated as non-overlap.",
        "",
        "### Four-judge versus five-judge ranking",
        "",
        "The following table compares the preserved four-judge aggregate with this new five-judge aggregate. A rank delta is `five minus four`; negative values move upward. This is a sensitivity comparison, not evidence that the added judge is more correct.",
        "",
        "| Candidate | 4-judge rank | 5-judge rank | Rank delta | 4-judge mean | 5-judge mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in four_vs_five:
        lines.append(f"| `{row['model_alias']}` (`{row['candidate_id']}`) | {fmt(row.get('rank_4judge'))} | {fmt(row.get('rank_5judge'))} | {fmt(row.get('rank_delta_5_minus_4'))} | {fmt(row.get('mean_4judge'), 2)} | {fmt(row.get('mean_5judge'), 2)} |")
    if not four_vs_five:
        lines.append("| n.a. | n.a. | n.a. | n.a. | n.a. | n.a. |")
    lines += ["", "LOFO details are retained in `aggregate.fivejudges.json` under `family_sensitivity.leave_one_family_out`; because this pilot has one judge per operational family and only three source clusters, LOFO is a sensitivity analysis, not a family-cluster robust CI or an identified crossed hierarchical model.", ""]

    lines += [
        "## Pairwise and position bias",
        "",
        "AB/BA results are diagnostic only. The graph uses three fixed pairs from one source cluster and remains disconnected, so no global seven-candidate Copeland ranking is reported.",
        "",
        "The `Context` column preserves the imported raw context flag, including `not_recorded`; that value means the boolean was absent from the raw row, not that the operational packaging is unknown. Known run evidence is shown separately: Gemini used 6 fresh UI chats for AB/BA, Qwen/Mistral/Nemotron used fresh independent requests per orientation, and GPT used one shared full-packet context.",
        "",
        "| Judge | Complete swaps | Consistent | Position-sensitive | Swap consistency | Raw context flag | Known operational evidence |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for judge_id, report in sorted((pairwise_summary.get("judges") or {}).items()):
        lines.append(f"| `{judge_id}` | {report.get('complete_swaps', 0)} | {report.get('consistent', 0)} | {report.get('position_sensitive', 0)} | {pct(report.get('swap_consistency'))} | `{report.get('context_status', 'not_recorded')}` | {_known_pairwise_operation(judge_id, summaries.get(judge_id))} |")
    lines += ["", "Full canonical AB/BA rows are in `pairwise_judge_diagnostics.fivejudges.owner.csv`; aggregate pairwise graph coverage and family-balanced diagnostics are in `aggregate.fivejudges.json`.", ""]

    lines += [
        "## Evidence files",
        "",
        f"- Aggregate: [`aggregate.fivejudges.json`](aggregate.fivejudges.json), SHA-256 `{sha256_file(output_dir / 'aggregate.fivejudges.json') if (output_dir / 'aggregate.fivejudges.json').exists() else 'written with this report'}`.",
        "- Validation audits: `validation.<judge-slug>.json` for each input bundle.",
        "- Owner CSVs: `translation_comparison.fivejudges.owner.csv`, `pointwise_results_long.fivejudges.owner.csv`, `pointwise_results_wide.fivejudges.owner.csv`, `error_annotations.fivejudges.owner.csv`, `judge_validity.fivejudges.owner.csv`, `judge_summary.fivejudges.owner.csv`, and `pairwise_judge_diagnostics.fivejudges.owner.csv`.",
        "- `fivejudge_manifest.json` binds each raw bundle hash, frozen packet/instructions hash, candidate mapping hash, source hash set, and output schema. Existing four-judge files are not overwritten.",
        "",
        "## Limitations",
        "",
        "- The panel is still a provisional five-rater pilot over three source clusters, not the formal 30/40-item protocol.",
        "- The judges have different context packing and latency provenance. GPT/Codex and Gemini UI timing cannot be compared to local API timings.",
        "- Pairwise swaps are all from one source cluster and have a disconnected candidate graph. Position-sensitive orientations are excluded from Copeland outcomes rather than imputed.",
        "- Qualification hidden-key metrics are owner diagnostics. They do not certify clinical safety, fact sensitivity, or deployment readiness.",
        "- LiteLLM calls remained paused. No external judge API was called by this aggregation step.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    packet_path: Path,
    instructions_path: Path,
    mapping_path: Path,
    answer_key_path: Path,
    input_paths: Sequence[Path],
    output_dir: Path,
    four_aggregate_path: Path | None,
    bootstrap_replicates: int,
    seed: int,
    allow_partial_pointwise: bool = False,
) -> dict[str, Any]:
    if len(input_paths) < 5:
        raise ValueError(f"expected five judge bundles, got {len(input_paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = read_json(packet_path)
    validations: dict[str, dict[str, Any]] = {}
    for path in input_paths:
        audit = validate_import(packet_path, path)
        validations[str(path)] = audit
        audit_path = output_dir / f"validation.{slug(_metadata_judge(load_bundle(path)[0])['judge_id'])}.json"
        write_json(audit_path, audit)
        # Pointwise/structural binding failures quarantine a bundle.  A
        # qualification scalar warning remains eligible as a partial track.
        errors = audit.get("errors") if isinstance(audit.get("errors"), list) else []
        # A malformed pointwise scalar is a row-local contract failure.  Keep
        # that row in the owner CSV, exclude it from means, and mark the
        # five-judge panel exploratory/partial when explicitly requested.  All
        # other validator errors remain a hard packet/binding quarantine.
        shape_only_errors = [error for error in errors if _is_pointwise_shape_error(error)]
        non_shape_errors = [error for error in errors if not _is_pointwise_shape_error(error)]
        allow_shape_partial = allow_partial_pointwise and bool(errors) and not non_shape_errors
        if non_shape_errors or (audit.get("status") != "PASS" and not allow_shape_partial):
            raise ValueError(f"refusing to aggregate quarantined bundle {path}: {(non_shape_errors or errors)[:4]}")
        eligibility = audit.get("track_eligibility", {}) if isinstance(audit.get("track_eligibility"), Mapping) else {}
        point = eligibility.get("pointwise", {}) if isinstance(eligibility.get("pointwise"), Mapping) else {}
        if point.get("status") != "eligible" and not (
            allow_partial_pointwise
            and not non_shape_errors
            and int(point.get("n_ok", 0) or 0) > 0
            and int(point.get("n_rows", 0) or 0) >= int(point.get("n_ok", 0) or 0)
        ):
            raise ValueError(f"refusing to aggregate pointwise-ineligible bundle {path}: {point}")

    aggregate_metadata, aggregate_rows = load_bundles(input_paths)
    aggregate_result = aggregate(aggregate_rows, aggregate_metadata, required_judges=5, bootstrap_replicates=bootstrap_replicates, seed=seed)
    aggregate_path = output_dir / "aggregate.fivejudges.json"
    write_json(aggregate_path, aggregate_result)

    csv_paths = {
        "long": output_dir / "pointwise_results_long.fivejudges.owner.csv",
        "wide": output_dir / "pointwise_results_wide.fivejudges.owner.csv",
        "comparison": output_dir / "translation_comparison.fivejudges.owner.csv",
        "validity_json": output_dir / "judge_validity.fivejudges.owner.json",
        "validity_csv": output_dir / "judge_validity.fivejudges.owner.csv",
        "errors": output_dir / "error_annotations.fivejudges.owner.csv",
    }
    csv_result = build_csv(packet_path, mapping_path, input_paths, csv_paths["long"], csv_paths["wide"], csv_paths["comparison"], csv_paths["validity_json"], csv_paths["validity_csv"], csv_paths["errors"])
    bundle_hashes = {str(path): sha256_file(path) for path in input_paths}
    for key in ("long", "errors"):
        _augment_csv_hashes(csv_paths[key], packet, bundle_hashes)
    # Translation comparison intentionally contains exact source/target text.
    # Add source hash as an audit column without touching any judge value.
    _augment_csv_hashes(csv_paths["comparison"], packet, bundle_hashes, add_bundle_hash=False)

    pair_rows, pair_summary = build_pairwise(input_paths)
    pair_csv = output_dir / "pairwise_judge_diagnostics.fivejudges.owner.csv"
    pair_json = output_dir / "pairwise_judge_diagnostics.fivejudges.owner.json"
    pair_fields = [
        "judge_id", "judge_family", "item_id", "pair_id", "fresh_chat_independence", "context_status", "candidate_order_ab", "candidate_order_ba", "winner_ab_canonical", "winner_ba_canonical", "status", "pairwise_limitation",
    ]
    _write_csv(pair_csv, pair_rows, pair_fields)
    write_json(pair_json, pair_summary)

    qualification_result = evaluate_qualification(input_paths, answer_key_path, packet_path)
    qualification_path = output_dir / "qualification_summary.fivejudges.owner.json"
    write_json(qualification_path, qualification_result)

    all_rows, summaries = _judge_rows_and_summaries(input_paths, validations)
    judge_summary_path = output_dir / "judge_summary.fivejudges.owner.csv"
    _write_judge_summary_csv(judge_summary_path, summaries)

    manifest = {
        "schema_version": SCRIPT_SCHEMA,
        "packet_id": packet.get("packet_id"),
        "packet_sha256": sha256_file(packet_path),
        "packet_canonical_json_sha256": canonical_json_sha256(packet),
        "instructions_sha256": sha256_file(instructions_path),
        "candidate_mapping_sha256": sha256_file(mapping_path),
        "answer_key_sha256": sha256_file(answer_key_path),
        "input_bundles": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "judge_id": _metadata_judge(load_bundle(path)[0])["judge_id"],
                "validation_path": str(output_dir / f"validation.{slug(_metadata_judge(load_bundle(path)[0])['judge_id'])}.json"),
                "validation_status": validations[str(path)].get("status"),
            }
            for path in input_paths
        ],
        "source_hashes": {item_id: source_hash for item_id, source_hash in sorted(packet_index(packet)["source_hashes"].items())},
        "candidate_ids": sorted(packet_index(packet)["candidate_ids"]),
        "output_dir": str(output_dir),
        "no_raw_mutation": True,
        "mapping_scope": "owner_only_postjudge",
    }
    write_json(output_dir / "fivejudge_manifest.json", manifest)

    report_path = output_dir / "selection_report.fivejudges.md"
    _markdown_report(report_path, packet_path, instructions_path, mapping_path, input_paths, validations, aggregate_result, summaries, pair_summary, qualification_result, four_aggregate_path, output_dir)
    return {
        "output_dir": str(output_dir),
        "aggregate": str(aggregate_path),
        "report": str(report_path),
        "csv": {key: str(value) for key, value in csv_paths.items()},
        "pairwise": str(pair_csv),
        "judge_summary": str(judge_summary_path),
        "qualification": str(qualification_path),
        "n_input_bundles": len(input_paths),
        "n_distinct_judges": aggregate_result.get("judge_scope", {}).get("n_distinct_judges"),
        "csv_result": csv_result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--instructions", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--four-aggregate", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-partial-pointwise", action="store_true", help="Keep strict PASS bundles with failed/missing pointwise rows as partial exploratory input; never repair or impute rows")
    args = parser.parse_args(argv)
    try:
        result = build_report(args.packet, args.instructions, args.mapping, args.answer_key, args.input, args.output_dir, args.four_aggregate, args.bootstrap_replicates, args.seed, args.allow_partial_pointwise)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
