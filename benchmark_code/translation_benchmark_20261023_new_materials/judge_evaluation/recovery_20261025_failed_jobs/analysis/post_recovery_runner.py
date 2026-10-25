#!/usr/bin/env python3
"""Isolated post-recovery aggregation for the 24 authorized non-valid jobs.

The normal analysis runner remains the baseline producer.  This script adds
only recovery-root attempts, preserves the baseline archive, and writes to
``post_recovery_reports`` after every target has a stable terminal recovery
outcome.  It never calls a model or mutates frozen inputs/raw baseline files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve()
RECOVERY_ROOT = HERE.parents[1]
JUDGE_ROOT = HERE.parents[2]
BUNDLE_ROOT = JUDGE_ROOT.parent
AGGREGATOR_PATH = JUDGE_ROOT / "analysis_runner" / "aggregate_new_panel.py"
BASELINE_ROOT = RECOVERY_ROOT / "baseline"
TARGET_REGISTRY = RECOVERY_ROOT / "analysis" / "recovery_target_registry.csv"
RECOVERY_MANIFEST = RECOVERY_ROOT / "recovery_manifest.json"
RECOVERY_POLICY = RECOVERY_ROOT / "recovery_policy.md"
OUTPUT_ROOT = RECOVERY_ROOT / "post_recovery_reports"
PROGRESS_PATH = RECOVERY_ROOT / "analysis" / "progress_post_recovery.md"

RECOVERY_TERMINAL_STATUSES = {
    "ok", "failed", "missing", "blocked", "unsupported", "cancelled",
    "invalid", "technical_invalid", "error",
}


def _numeric_value(value: Any) -> float | None:
    """Return a finite number for CSV/derived values, otherwise ``None``."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_value(left: Any, right: Any) -> bool:
    """Compare CSV strings and derived numeric/bool values without coercing text."""
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is not None and right_number is not None:
        # CSV writers intentionally round derived means/reliability values;
        # score/status/source binding fields remain exact integers/strings.
        return math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-9)
    if isinstance(left, bool) or isinstance(right, bool):
        return str(left).lower() == str(right).lower()
    left_text = "" if left is None else str(left)
    right_text = "" if right is None else str(right)
    if left_text.lower() in {"true", "false"} and right_text.lower() in {"true", "false"}:
        return left_text.lower() == right_text.lower()
    return left_text == right_text


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing required baseline CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _common_case_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("source_kind", "")), str(row.get("material_id", "")))
        for row in rows
        if str(row.get("common_case_eligible", "")).lower() == "true"
    }


def _common_source_page_count(rows: Sequence[Mapping[str, Any]], keys: set[tuple[str, str]]) -> int:
    clusters = {
        str(row.get("source_cluster") or row.get("source_id") or row.get("material_id"))
        for row in rows
        if (str(row.get("source_kind", "")), str(row.get("material_id", ""))) in keys
        and str(row.get("source_kind", "")) == "text"
    }
    return len(clusters)


def same_subset_invariance(aggregator: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that the original valid common subset was not changed by recovery.

    Recovery targets are deliberately outside the original 19-text/2-image
    intersection.  This gate compares immutable case/source/target bindings,
    every judge scalar/status field, model common means, and recomputed
    reliability statistics on exactly the archived common rows.  The expanded
    panel is reported separately and is never treated as an improvement of
    judge ability.
    """
    baseline_dir = BASELINE_ROOT / "reports"
    baseline_comparison = _csv_rows(baseline_dir / "translation_comparison.csv")
    baseline_reliability = _csv_rows(baseline_dir / "reliability_summary.csv")
    baseline_common_summary = _csv_rows(baseline_dir / "common_case_model_summary.csv")
    baseline_keys = _common_case_keys(baseline_comparison)
    new_rows = list(data.get("comparison_rows", []))
    new_keys = _common_case_keys(new_rows)
    missing_common_rows = sorted(baseline_keys - new_keys)

    baseline_by_key = {
        (str(row.get("source_kind", "")), str(row.get("material_id", "")), str(row.get("model_id", ""))): row
        for row in baseline_comparison
        if (str(row.get("source_kind", "")), str(row.get("material_id", ""))) in baseline_keys
    }
    new_by_key = {
        (str(row.get("source_kind", "")), str(row.get("material_id", "")), str(row.get("model_id", ""))): row
        for row in new_rows
        if (str(row.get("source_kind", "")), str(row.get("material_id", ""))) in baseline_keys
    }
    immutable_fields = [
        "material_id", "source_id", "source_kind", "source_en", "source_sha256",
        "image_alias", "image_sha256", "model_id", "model_name", "model_alias",
        "target_zh_tw", "image_ground_truth_en", "image_reference_text_kind",
        "image_ground_truth_status", "image_role", "image_layout_stratum",
        "translation_status", "translation_elapsed_seconds", "translation_untranslated_visible_english",
        "panel_required_k", "panel_n_valid", "panel_complete", "panel_mean", "panel_median",
    ]
    for slot in getattr(aggregator, "JUDGE_SLOTS", getattr(aggregator, "TEXT_PANEL_SLOTS", [])):
        immutable_fields.extend([
            f"{slot}_score", f"{slot}_status", f"{slot}_valid", f"{slot}_critical",
            f"{slot}_error_count", f"{slot}_evidence_warning_count",
            f"{slot}_selected_attempt", f"{slot}_attempt_kind",
            f"{slot}_infrastructure_recovery_used",
        ])
    # The current runner exposes fixed text/image slots rather than a single
    # JUDGE_SLOTS constant.  Keep all fixed-panel fields in the comparison.
    for slot in sorted(set(getattr(aggregator, "TEXT_PANEL_SLOTS", [])) | set(getattr(aggregator, "IMAGE_PANEL_SLOTS", []))):
        immutable_fields.extend([
            f"{slot}_score", f"{slot}_status", f"{slot}_valid", f"{slot}_critical",
            f"{slot}_error_count", f"{slot}_evidence_warning_count",
            f"{slot}_selected_attempt", f"{slot}_attempt_kind",
            f"{slot}_infrastructure_recovery_used",
        ])
    immutable_fields = list(dict.fromkeys(immutable_fields))
    binding_mismatches: list[dict[str, Any]] = []
    for key, old in sorted(baseline_by_key.items()):
        new = new_by_key.get(key)
        if new is None:
            binding_mismatches.append({"key": key, "field": "row_missing_after_recovery"})
            continue
        for field in immutable_fields:
            if field in old and field in new and not _same_value(old.get(field), new.get(field)):
                binding_mismatches.append({"key": key, "field": field, "baseline": old.get(field), "post_recovery": new.get(field)})

    # Recompute reliability from precisely the archived common rows.  The
    # restricted calculation uses a local gate so its numeric values are
    # comparable even if the expanded panel later reaches a full-track gate.
    restricted_rows = [
        dict(row) for row in new_rows
        if (str(row.get("source_kind", "")), str(row.get("material_id", ""))) in baseline_keys
    ]
    restricted_common: dict[str, dict[str, Any]] = {}
    for track in ("text", "image"):
        track_keys = {material_id for kind, material_id in baseline_keys if kind == track}
        restricted_common[track] = {
            "analysis_ready": bool(track_keys),
            "eligible_material_ids": sorted(track_keys),
            "eligible_item_count": len(track_keys),
            "source_page_count": _common_source_page_count(restricted_rows, {(track, item) for item in track_keys}),
            "excluded_case_ids": [],
            "excluded_case_reasons": {},
        }
    restricted_reliability = aggregator.build_reliability(
        restricted_rows,
        [dict(row) for row in data.get("point_rows", []) if (str(row.get("source_kind", "")), str(row.get("material_id", ""))) in baseline_keys],
        {"all_primary_terminal": True},
        restricted_common,
    )
    baseline_rel_by_track = {str(row.get("track")): row for row in baseline_reliability}
    post_rel_by_track = {str(row.get("track")): row for row in restricted_reliability}
    reliability_fields = [
        "common_case_items", "common_case_source_pages", "actual_k", "icc_2_1_absolute_agreement", "icc_2_k_absolute_agreement",
        "kendall_w", "kendall_w_system_level", "kendall_w_unit_level",
        "kendall_w_per_item_defined_count", "kendall_w_per_item_mean",
        "kendall_w_per_item_median",
    ]
    reliability_mismatches: list[dict[str, Any]] = []
    reliability_tracks: dict[str, dict[str, Any]] = {}
    for track in ("text", "image"):
        old = baseline_rel_by_track.get(track, {})
        new = post_rel_by_track.get(track, {})
        track_mismatches = []
        for field in reliability_fields:
            if not _same_value(old.get(field), new.get(field)):
                item = {"track": track, "field": field, "baseline": old.get(field), "recomputed": new.get(field)}
                reliability_mismatches.append(item)
                track_mismatches.append(item)
        reliability_tracks[track] = {
            "baseline": {field: old.get(field) for field in reliability_fields},
            "recomputed_same_subset": {field: new.get(field) for field in reliability_fields},
            "mismatches": track_mismatches,
        }

    # Check the same-denominator candidate means/medians as an additional
    # guard.  Do not compare expanded means because recovery changes n.
    baseline_model_by_key = {(str(row.get("track")), str(row.get("model_id"))): row for row in baseline_common_summary}
    restricted_model_summary = aggregator.build_common_case_model_summary(
        restricted_rows, restricted_common, data.get("source_rows", {})
    )
    post_model_by_key = {(str(row.get("track")), str(row.get("model_id"))): row for row in restricted_model_summary}
    model_fields = ["common_case_items", "required_judge_k", "valid_judge_cells", "expected_judge_cells", "common_case_panel_mean", "common_case_panel_median"]
    model_mismatches: list[dict[str, Any]] = []
    for key, old in sorted(baseline_model_by_key.items()):
        new = post_model_by_key.get(key, {})
        for field in model_fields:
            if not _same_value(old.get(field), new.get(field)):
                model_mismatches.append({"key": key, "field": field, "baseline": old.get(field), "recomputed": new.get(field)})

    return {
        "status": "PASS" if not missing_common_rows and not binding_mismatches and not reliability_mismatches and not model_mismatches else "FAIL",
        "baseline_common_case_counts": {
            "text": sum(kind == "text" for kind, _ in baseline_keys),
            "image": sum(kind == "image" for kind, _ in baseline_keys),
        },
        "post_recovery_common_case_counts": {
            "text": sum(kind == "text" for kind, _ in new_keys),
            "image": sum(kind == "image" for kind, _ in new_keys),
        },
        "baseline_common_keys_missing_after_recovery": [list(item) for item in missing_common_rows],
        "binding_mismatches": binding_mismatches,
        "reliability": reliability_tracks,
        "reliability_mismatches": reliability_mismatches,
        "model_mean_mismatches": model_mismatches,
    }


def expanded_panel_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return expanded/full-panel n/k and reliability fields separately."""
    status = data.get("status", {})
    reliability = {str(row.get("track")): row for row in data.get("reliability", [])}
    common_cases = data.get("common_cases", {})
    result: dict[str, Any] = {}
    for track in ("text", "image"):
        rel = reliability.get(track, {})
        common = common_cases.get(track, {})
        result[track] = {
            "analysis_status": rel.get("analysis_status", status.get("analysis_status")),
            "common_case_items": rel.get("common_case_items", common.get("eligible_item_count", 0)),
            "common_case_source_pages": rel.get("common_case_source_pages", common.get("source_page_count", 0)),
            "complete_balanced_units": rel.get("complete_balanced_units"),
            "expected_balanced_units": rel.get("expected_balanced_units"),
            "full_fixed_panel_gate": rel.get("full_fixed_panel_gate"),
            "actual_k": rel.get("actual_k"),
            "icc_2_1_absolute_agreement": rel.get("icc_2_1_absolute_agreement"),
            "icc_2_k_absolute_agreement": rel.get("icc_2_k_absolute_agreement"),
            "kendall_w_system_level": rel.get("kendall_w_system_level"),
            "kendall_w_unit_level": rel.get("kendall_w_unit_level"),
            "track_complete_units": status.get(f"{track}_complete_units"),
            "track_expected_units": status.get(f"{track}_expected_units"),
        }
    return result


def make_recovery_before_after_markdown(
    data: Mapping[str, Any],
    invariance: Mapping[str, Any],
    expanded: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> str:
    """Write a compact, explicit before/after evidence note."""
    lines = [
        "# Post-recovery before/after evidence",
        "",
        "This file is a derived recovery report. Original valid scores, frozen candidate outputs, raw files, and the baseline report are immutable. Recovery accepts at most the first strict-valid result for each of the 24 authorized non-valid target jobs; it never selects a highest score.",
        "",
        f"- Recovery terminal status: `{terminal.get('complete')}`; target count: `{terminal.get('target_count', 24)}`.",
        f"- Same-subset invariance: **{invariance.get('status')}**. The archived common subset remains the fixed `{invariance.get('baseline_common_case_counts', {}).get('text', 0)}` text / `{invariance.get('baseline_common_case_counts', {}).get('image', 0)}` image cases.",
        "- Any expanded-panel change is a change in included sample composition/coverage, not evidence that judge ability or inter-rater agreement improved. No score was imputed or regraded.",
        "",
        "## Same-subset reliability (original common cases only)",
        "",
        "The recomputation below uses exactly the archived common rows, not the expanded recovery intersection. `ICC CI` is not calculated; no model-mean bootstrap CI is presented here.",
        "",
        "|track|cases|source pages|k|baseline ICC(2,1)|post same-subset ICC(2,1)|baseline ICC(2,k)|post same-subset ICC(2,k)|baseline W|post same-subset W|status|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for track in ("text", "image"):
        item = invariance.get("reliability", {}).get(track, {})
        old = item.get("baseline", {})
        new = item.get("recomputed_same_subset", {})
        lines.append(
            f"|{track}|{invariance.get('baseline_common_case_counts', {}).get(track, 0)}|{_display(old.get('common_case_source_pages', expanded.get(track, {}).get('common_case_source_pages', '')))}|{_display(old.get('actual_k'))}|{_display(old.get('icc_2_1_absolute_agreement'))}|{_display(new.get('icc_2_1_absolute_agreement'))}|{_display(old.get('icc_2_k_absolute_agreement'))}|{_display(new.get('icc_2_k_absolute_agreement'))}|{_display(old.get('kendall_w_system_level'))}|{_display(new.get('kendall_w_system_level'))}|{invariance.get('status')}|"
        )
    lines.extend([
        "",
        "## Expanded/full-panel state",
        "",
        "These values use the post-recovery coverage and its own eligible intersection. They must be interpreted with their explicit n/k and gate; a changed value is not a before/after quality gain.",
        "",
        "|track|analysis status|common items|source pages|complete/expected units|fixed-panel gate|actual k|ICC(2,1)|ICC(2,k)|W system|W unit|",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for track in ("text", "image"):
        item = expanded.get(track, {})
        lines.append(
            f"|{track}|{_display(item.get('analysis_status'))}|{_display(item.get('common_case_items'))}|{_display(item.get('common_case_source_pages'))}|{_display(item.get('complete_balanced_units'))}/{_display(item.get('expected_balanced_units'))}|{_display(item.get('full_fixed_panel_gate'))}|{_display(item.get('actual_k'))}|{_display(item.get('icc_2_1_absolute_agreement'))}|{_display(item.get('icc_2_k_absolute_agreement'))}|{_display(item.get('kendall_w_system_level'))}|{_display(item.get('kendall_w_unit_level'))}|"
        )
    lines.extend([
        "",
        "The original common-subset numbers are the comparison baseline. Expanded/full-panel statistics are exploratory and remain subject to the frozen common-case and image-design limitations. Image scores are not OCR accuracy.",
        "",
    ])
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "" if value is None else str(value)


def load_aggregator():
    spec = importlib.util.spec_from_file_location("aggregate_new_panel", AGGREGATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import aggregator: {AGGREGATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_registry(path: Path = TARGET_REGISTRY) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"judge_slot", "track", "job_id", "material_id", "baseline_status", "baseline_valid"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("recovery target registry is missing required columns")
    keys = [(row["judge_slot"], row["track"], row["job_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("recovery target registry has duplicate keys")
    if any(row.get("baseline_valid", "").lower() == "true" for row in rows):
        raise RuntimeError("recovery registry contains an original valid target")
    return rows


def verify_baseline(aggregator: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    baseline = manifest.get("baseline") if isinstance(manifest.get("baseline"), Mapping) else {}
    files = baseline.get("files") if isinstance(baseline.get("files"), Mapping) else {}
    mismatches: list[str] = []
    for relative, expected in files.items():
        path = BASELINE_ROOT / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(relative))
    frozen = manifest.get("frozen_inputs") if isinstance(manifest.get("frozen_inputs"), Mapping) else {}
    actual_paths = {
        "packet_manifest_sha256": JUDGE_ROOT / "manifest.json",
        "source_translation_results_sha256": BUNDLE_ROOT / "results" / "translation_results.csv",
        "frozen_text_sha256": BUNDLE_ROOT / "materials" / "frozen_text_materials.jsonl",
        "image_selection_sha256": BUNDLE_ROOT / "materials" / "image_selection.json",
        "material_manifest_sha256": BUNDLE_ROOT / "materials" / "metadata_manifest.json",
    }
    frozen_mismatches: list[str] = []
    for key, path in actual_paths.items():
        if not path.is_file() or sha256_file(path) != frozen.get(key):
            frozen_mismatches.append(key)
    policy_expected = ((manifest.get("policy") or {}).get("sha256"))
    policy_ok = RECOVERY_POLICY.is_file() and sha256_file(RECOVERY_POLICY) == policy_expected
    registry_expected = ((manifest.get("target_registry") or {}).get("sha256"))
    registry_ok = TARGET_REGISTRY.is_file() and sha256_file(TARGET_REGISTRY) == registry_expected
    live_raw_mismatches: list[str] = []
    archived_analysis_manifest = BASELINE_ROOT / "reports" / "analysis_manifest.json"
    if archived_analysis_manifest.is_file():
        archived_snapshot = read_json(archived_analysis_manifest).get("raw_snapshot_files", [])
        for item in archived_snapshot:
            path = Path(str(item.get("path", "")))
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                live_raw_mismatches.append(str(path))
    else:
        live_raw_mismatches.append(str(archived_analysis_manifest))
    return {
        "baseline_files_ok": not mismatches,
        "baseline_mismatches": mismatches,
        "frozen_inputs_ok": not frozen_mismatches,
        "frozen_input_mismatches": frozen_mismatches,
        "policy_ok": policy_ok,
        "registry_ok": registry_ok,
        "live_raw_baseline_ok": not live_raw_mismatches,
        "live_raw_baseline_mismatches": live_raw_mismatches,
        "ok": not mismatches and not frozen_mismatches and policy_ok and registry_ok and not live_raw_mismatches,
    }


def is_under(path: str, roots: Sequence[Path]) -> bool:
    candidate = Path(path).resolve()
    for root in roots:
        try:
            candidate.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def split_recovery_attempts(
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    roots: Sequence[Path],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], int]:
    recovery: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    observed = 0
    for key, records in attempts.items():
        for record in records:
            if is_under(str(record.get("path", "")), roots):
                recovery[key].append(dict(record))
                observed += 1
    return recovery, observed


def _recovery_manifest_job(manifest: Mapping[str, Any], judge_slot: str, track: str, job_id: str) -> Mapping[str, Any] | None:
    """Find a job by its declared task binding, never by a filename."""
    for job in manifest.get("jobs", []) if isinstance(manifest.get("jobs"), list) else []:
        if not isinstance(job, Mapping):
            continue
        binding = job.get("task_binding") if isinstance(job.get("task_binding"), Mapping) else {}
        declared_job = binding.get("job_id") if track == "pointwise" else binding.get("pair_job_id")
        declared_slot = str((manifest.get("route") or {}).get("judge_slot") or judge_slot)
        if declared_slot == judge_slot and str(job.get("track") or binding.get("track") or "") == track and str(declared_job or "") == job_id:
            return job
    return None


def _strip_recovery_parent_id(value: Any, track: str) -> str | None:
    text = str(value or "")
    prefix = f"{track}-"
    return text[len(prefix):] if text.startswith(prefix) and len(text) > len(prefix) else None


def _hash_matches(path_value: Any, expected: Any) -> bool:
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    path = Path(str(path_value or ""))
    return path.is_file() and sha256_file(path) == expected


def _validate_gemini_recovery_group(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    registry_keys: set[tuple[str, str, str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one Gemini request/response pair and emit one logical attempt.

    The native UI capture creates two JSON artifacts.  Both are retained in
    the immutable snapshot, but only the request envelope is the canonical
    attempt after its manifest-declared response/hash/parent evidence passes.
    """
    issues: list[str] = []
    request_records = [
        record for record in records
        if isinstance(record.get("raw"), Mapping)
        and str(record["raw"].get("schema_version", "")) == "gemini-failed-job-recovery-request-v1"
    ]
    response_records = [
        record for record in records
        if isinstance(record.get("raw"), Mapping)
        and str(record["raw"].get("schema_version", "")) == "gemini-failed-job-recovery-response-envelope-v1"
    ]
    if not request_records:
        return None, ["gemini_request_envelope_count:0"]
    parsed_request_records = [record for record in request_records if record.get("parsed") is not None]
    if len(parsed_request_records) > 1:
        return None, [f"gemini_multiple_parsed_request_envelopes:{len(parsed_request_records)}"]
    # A prepared_not_sent envelope can remain beside the one successful
    # transport retry.  It is evidence of an interim state, not a second
    # scoring attempt, so prefer the sole parsed request and ignore the stale
    # unparsed envelope within this logical target group.
    request_record = dict(parsed_request_records[0] if parsed_request_records else request_records[-1])
    raw = dict(request_record.get("raw") or {})
    track = str(raw.get("track") or request_record.get("track") or "")
    judge_slot = str(raw.get("judge_slot") or request_record.get("judge_slot") or "")
    parent_request_id = str(raw.get("parent_request_id") or "")
    job_id = _strip_recovery_parent_id(parent_request_id, track)
    key = (judge_slot, track, str(job_id or ""))
    if track not in {"pointwise", "pairwise"} or not job_id or key not in registry_keys:
        issues.append(f"gemini_parent_not_registry:{parent_request_id}")
    manifest_path = root / "recovery_manifest.json"
    if not manifest_path.is_file():
        issues.append("gemini_manifest_missing")
        return None, issues
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None, ["gemini_manifest_unreadable"]
    job = _recovery_manifest_job(manifest, judge_slot, track, str(job_id or ""))
    if job is None:
        issues.append(f"gemini_manifest_job_missing:{key}")
    else:
        for field in ("parent_request_id", "task_sha256", "instruction_sha256", "public_schema_sha256", "user_authorized_failed_job_retry", "retry_number"):
            if field in job and not _same_value(raw.get(field), job.get(field)):
                issues.append(f"gemini_manifest_field_mismatch:{field}")
        manifest_binding = job.get("task_binding") if isinstance(job.get("task_binding"), Mapping) else {}
        if str(manifest_binding.get("job_id") if track == "pointwise" else manifest_binding.get("pair_job_id")) != str(job_id):
            issues.append("gemini_manifest_task_binding_mismatch")
        attempts = job.get("attempts") if isinstance(job.get("attempts"), list) else []
        declared_response_hashes = {str(item.get("response_envelope_sha256")) for item in attempts if isinstance(item, Mapping)}
        if raw.get("response_envelope_sha256") not in declared_response_hashes:
            issues.append("gemini_manifest_response_hash_missing")
    if raw.get("route_status") != "clean" or raw.get("route_contaminated"):
        issues.append("gemini_route_not_clean")
    if raw.get("user_authorized_failed_job_retry") is not True:
        issues.append("gemini_retry_authorization_missing")
    if raw.get("fresh_conversation_confirmed") is not True:
        issues.append("gemini_fresh_conversation_unconfirmed")
    expected_parent_attempt = (job or {}).get("parent_attempt") if job is not None else raw.get("parent_attempt")
    if raw.get("retry_number") != 1 or raw.get("parent_attempt") != expected_parent_attempt:
        issues.append("gemini_retry_number_or_parent_attempt_invalid")
    for field in ("task_sha256", "instruction_sha256", "prompt_sha256", "response_sha256", "full_ax_sha256", "response_envelope_sha256"):
        value = raw.get(field)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"gemini_hash_format:{field}")
    if not _hash_matches(raw.get("capture_path"), raw.get("full_ax_sha256")):
        issues.append("gemini_full_ax_hash_mismatch")
    if not _hash_matches(raw.get("prompt_path"), raw.get("prompt_sha256")):
        issues.append("gemini_prompt_hash_mismatch")
    parent_path = Path(str((job or {}).get("parent_raw_resolved_path") or (job or {}).get("parent_raw_path") or ""))
    parent_hash = (job or {}).get("parent_raw_sha256") if job else None
    if job and parent_hash:
        # Missing-parent jobs intentionally have no source hash.  When a hash
        # is declared, it remains mandatory and is checked against the exact
        # manifest-resolved artifact; this is not a blanket relaxation.
        if not _hash_matches(parent_path, parent_hash):
            issues.append("gemini_parent_raw_hash_mismatch")

    response_path = Path(str(raw.get("response_envelope_path") or ""))
    if not response_path.is_file() or not is_under(str(response_path), [root]):
        issues.append("gemini_response_envelope_path_invalid")
    elif not _hash_matches(response_path, raw.get("response_envelope_sha256")):
        issues.append("gemini_response_envelope_hash_mismatch")
    response_data: Mapping[str, Any] = {}
    if response_path.is_file() and not issues:
        try:
            loaded = read_json(response_path)
            response_data = loaded if isinstance(loaded, Mapping) else {}
        except (OSError, json.JSONDecodeError):
            issues.append("gemini_response_envelope_unreadable")
    if response_records:
        response_record_raw = next(
            (
                record.get("raw") for record in response_records
                if isinstance(record.get("raw"), Mapping)
                and str(record["raw"].get("request_id") or "") == str(raw.get("request_id") or "")
            ),
            response_records[0].get("raw"),
        )
        if isinstance(response_record_raw, Mapping) and response_data and dict(response_record_raw) != dict(response_data):
            issues.append("gemini_response_record_mismatch")
    if response_data:
        for field in ("parent_request_id", "task_id", "route_status", "user_authorized_failed_job_retry", "retry_number", "full_ax_sha256", "response_sha256"):
            expected = parent_request_id if field == "parent_request_id" else job_id if field == "task_id" else raw.get(field)
            if field in response_data and not _same_value(response_data.get(field), expected):
                issues.append(f"gemini_response_field_mismatch:{field}")
        response_parsed = response_data.get("parsed")
        if request_record.get("parsed") is not None and response_parsed != request_record.get("parsed"):
            issues.append("gemini_response_parsed_mismatch")
    response_raw_text = (raw.get("parse") or {}).get("response_raw") if isinstance(raw.get("parse"), Mapping) else None
    if isinstance(response_raw_text, str) and hashlib.sha256(response_raw_text.encode("utf-8")).hexdigest() != raw.get("response_sha256"):
        issues.append("gemini_response_sha256_mismatch")

    if issues:
        return None, issues
    request_record["request_id"] = str(job_id)
    request_record["raw_request_id"] = str(raw.get("request_id") or "")
    request_record["attempt"] = 1
    request_record["status"] = "ok" if request_record.get("parsed") is not None else str(raw.get("status") or "missing").lower()
    request_record["task_binding"] = dict((job or {}).get("task_binding") or {})
    request_record["task"] = dict((job or {}).get("public_task") or {})
    provenance = dict(request_record.get("provenance") or {})
    provenance.update({
        "recovery_metadata_valid": True,
        "recovery_manifest_verified": True,
        "parent_request_id": parent_request_id,
        "response_envelope_path": str(response_path),
        "response_envelope_sha256": raw.get("response_envelope_sha256"),
        "full_ax_sha256": raw.get("full_ax_sha256"),
        "task_sha256": raw.get("task_sha256"),
        "route_status": raw.get("route_status"),
        "parent_raw_hash_declared": bool(parent_hash),
        "parent_raw_artifact_missing_allowed": bool(job and not parent_hash and str(job.get("parent_status")) == "missing"),
    })
    request_record["provenance"] = provenance
    # The captured request envelope uses ``captured_native_ui`` as an
    # artifact status, while the parsed response itself is a successful
    # attempt.  The normal importer expects the in-memory raw execution
    # status to be one of its success values.  Preserve the capture status in
    # a separate provenance field; do not edit the append-only JSON file.
    if request_record.get("parsed") is not None:
        raw["capture_status"] = raw.get("status")
        raw["status"] = "ok"
        raw_provenance = dict(raw.get("provenance") or {})
        raw_provenance.update(provenance)
        raw["provenance"] = raw_provenance
    request_record["raw"] = raw
    return request_record, []


def validate_recovery_attempts(
    recovery_attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    roots: Sequence[Path],
    registry: Sequence[Mapping[str, str]],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], list[str]]:
    """Normalize append-only recovery artifacts into one verified attempt/job."""
    registry_keys = {(row["judge_slot"], row["track"], row["job_id"]) for row in registry}
    normalized: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    issues: list[str] = []
    gemini_groups: dict[tuple[Path, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for original_key, records in recovery_attempts.items():
        for record in records:
            path = Path(str(record.get("path", "")))
            raw_artifact = record.get("raw") or {}
            if (raw_artifact.get("schema_version") == "user-authorized-failed-job-retry-v1"
                    and path.name == "batch_manifest.json" and record.get("parsed") is None):
                # Execution manifests remain in the hashed file inventory;
                # they are not model responses or scoring attempts.
                continue
            root = next((candidate for candidate in roots if is_under(str(path), [candidate])), None)
            if root is not None and root.name == "gemini":
                raw = record.get("raw") if isinstance(record.get("raw"), Mapping) else {}
                track = str(raw.get("track") or record.get("track") or "")
                parent = str(raw.get("parent_request_id") or "")
                job_id = _strip_recovery_parent_id(parent, track) or str(record.get("request_id") or "")
                gemini_groups[(root, str(record.get("judge_slot") or "judge-05"), track, job_id)].append(record)
            else:
                key = original_key
                if key in registry_keys:
                    normalized[key].append(dict(record))
                else:
                    # Keep non-target artifacts out of the merged attempt map.
                    issues.append(f"non_target_recovery_record:{key}")
    for (root, judge_slot, track, job_id), records in gemini_groups.items():
        canonical, group_issues = _validate_gemini_recovery_group(records, root, registry_keys)
        if canonical is not None:
            key = (judge_slot, track, job_id)
            normalized[key].append(canonical)
        issues.extend(f"{judge_slot}/{track}/{job_id}:{issue}" for issue in group_issues)
    for key in normalized:
        normalized[key].sort(key=lambda record: (int(record.get("attempt", 0) or 0), str(record.get("path", ""))))
    return normalized, issues


def collect_baseline_snapshot(aggregator: Any) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Read only files in the archived baseline raw snapshot.

    Owners may add new diagnostic files under the live raw tree while recovery
    is running.  Those files must never become baseline attempts merely because
    the ordinary collector can see them.
    """
    archived_manifest = read_json(BASELINE_ROOT / "reports" / "analysis_manifest.json")
    allowed = {
        str(item.get("path")): str(item.get("sha256"))
        for item in archived_manifest.get("raw_snapshot_files", [])
        if item.get("path") and item.get("sha256")
    }
    attempts, diagnostics, files, checkpoints = aggregator.collect_raw_snapshot()
    filtered_attempts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, records in attempts.items():
        for record in records:
            if allowed.get(str(record.get("path"))) == str(record.get("raw_hash")):
                filtered_attempts[key].append(dict(record))
    filtered_files = [item for item in files if allowed.get(str(item.get("path"))) == str(item.get("sha256"))]
    return filtered_attempts, diagnostics, filtered_files, checkpoints


def merge_target_attempts(
    baseline_attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    recovery_attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    registry: Sequence[Mapping[str, str]],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Merge only the exact non-valid registry keys.

    Existing valid records are not in the registry and therefore cannot be
    replaced.  Recovery records are marked in-memory only; raw files remain
    untouched.  A recovery attempt must be numbered after the original and
    may contain at most one parsed scoring payload; an extra payload is an
    explicit policy violation rather than a quality-selected retry.
    """
    target_keys = {(row["judge_slot"], row["track"], row["job_id"]) for row in registry}
    merged: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, records in baseline_attempts.items():
        merged[key].extend(dict(record) for record in records)
    ignored_non_target = 0
    metadata_invalid = 0
    score_payload_overflow = 0
    accepted_records = 0
    for key, records in recovery_attempts.items():
        if key not in target_keys:
            ignored_non_target += len(records)
            continue
        parsed_count = 0
        for record in records:
            attempt = record.get("attempt")
            if not isinstance(attempt, int) or attempt < 1:
                metadata_invalid += 1
                continue
            if record.get("parsed") is not None:
                parsed_count += 1
            if parsed_count > 1:
                score_payload_overflow += 1
                continue
            marked = dict(record)
            marked["recovery_lane"] = True
            marked["recovery_round"] = 1
            merged[key].append(marked)
            accepted_records += 1
    for key in merged:
        merged[key].sort(key=lambda record: (int(record.get("attempt", 0) or 0), str(record.get("path", ""))))
    return merged, {
        "target_count": len(target_keys),
        "accepted_recovery_records": accepted_records,
        "ignored_non_target_records": ignored_non_target,
        "metadata_invalid_records": metadata_invalid,
        "score_payload_overflow_records": score_payload_overflow,
        "target_keys_observed": len(target_keys & set(recovery_attempts)),
    }


def recovery_terminal_statuses(
    recovery_attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    registry: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    pending: list[str] = []
    terminal: dict[str, str] = {}
    for row in registry:
        key = (row["judge_slot"], row["track"], row["job_id"])
        records = recovery_attempts.get(key, [])
        statuses = [str(record.get("status") or "").lower() for record in records]
        valid_statuses = [
            status for record, status in zip(records, statuses)
            if status in RECOVERY_TERMINAL_STATUSES
            # A transient ``ok`` envelope without a parsed result is not a
            # terminal recovery outcome; it must not close the retry gate.
            and not (status == "ok" and record.get("parsed") is None)
        ]
        if not valid_statuses:
            pending.append("/".join(key))
        else:
            terminal["/".join(key)] = valid_statuses[-1]
    return {"complete": not pending, "pending_keys": pending, "terminal_statuses": terminal}


def build_data(
    aggregator: Any,
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    snapshot_files: Sequence[Mapping[str, Any]],
    snapshot_diagnostics: Sequence[Mapping[str, Any]],
    checkpoint_terminal_statuses: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    source_rows, _ = aggregator.load_source_rows()
    materials, _, material_manifest = aggregator.load_materials()
    point_jobs, pair_jobs, private_context = aggregator.load_jobs()
    point_rows = aggregator.build_pointwise_rows(point_jobs, attempts, materials)
    comparison_rows = aggregator.build_comparison_rows(point_rows, source_rows, materials)
    pair_rows = aggregator.build_pairwise_rows(pair_jobs, attempts)
    canonical_rows = aggregator.build_pairwise_canonical(pair_rows)
    terminal_coverage = aggregator.terminal_attempt_coverage(point_jobs, pair_jobs, attempts, checkpoint_terminal_statuses)
    common_cases = {
        track: aggregator.common_case_intersection(comparison_rows, track, terminal_coverage)
        for track in ("text", "image")
    }
    aggregator.annotate_common_case(comparison_rows, common_cases)
    judge_summary = aggregator.summarize_judges(point_rows, pair_rows, canonical_rows, snapshot_files)
    model_summary = aggregator.build_model_summary(comparison_rows, point_rows, source_rows)
    common_case_model_summary = aggregator.build_common_case_model_summary(comparison_rows, common_cases, source_rows)
    reliability = aggregator.build_reliability(comparison_rows, point_rows, terminal_coverage, common_cases)
    bootstrap = aggregator.build_bootstrap(comparison_rows, materials, terminal_coverage, common_cases)
    lofo = aggregator.build_lofo(comparison_rows, terminal_coverage, common_cases)
    errors = aggregator.build_error_annotations(point_rows, attempts, point_jobs, materials)
    invalid = aggregator.build_invalid_annotations(point_rows, attempts, point_jobs, pair_rows, pair_jobs)
    generation = aggregator.build_translation_generation_summary(source_rows)
    settings = aggregator.build_settings_provenance(snapshot_files, attempts)
    status = aggregator.report_status(point_rows, comparison_rows, pair_rows, terminal_coverage, common_cases)
    image_visibility = aggregator.independent_image_flag_summary(materials, private_context.get("job_map", {}))
    return {
        "source_rows": source_rows, "materials": materials, "material_manifest": material_manifest,
        "point_jobs": point_jobs, "pair_jobs": pair_jobs, "private_context": private_context,
        "attempts": attempts, "snapshot_diagnostics": list(snapshot_diagnostics), "snapshot_files": list(snapshot_files),
        "checkpoint_terminal_statuses": dict(checkpoint_terminal_statuses), "common_cases": common_cases,
        "point_rows": point_rows, "comparison_rows": comparison_rows, "pair_rows": pair_rows,
        "canonical_rows": canonical_rows, "judge_summary": judge_summary, "model_summary": model_summary,
        "common_case_model_summary": common_case_model_summary, "reliability": reliability,
        "bootstrap": bootstrap, "lofo": lofo, "errors": errors, "invalid": invalid,
        "generation": generation, "settings": settings, "terminal_coverage": terminal_coverage,
        "status": status, "image_visibility": image_visibility,
    }


def status_snapshot() -> dict[str, Any]:
    aggregator = load_aggregator()
    manifest = read_json(RECOVERY_MANIFEST)
    registry = read_registry()
    baseline_attempts, baseline_diag, baseline_files, _ = collect_baseline_snapshot(aggregator)
    roots = [path for path in (RECOVERY_ROOT / "gemini", RECOVERY_ROOT / "qwen") if path.is_dir()]
    all_attempts, recovery_diag, recovery_files, _ = aggregator.collect_raw_snapshot(extra_roots=roots)
    recovery_attempts_raw, observed = split_recovery_attempts(all_attempts, roots)
    recovery_attempts, recovery_validation_issues = validate_recovery_attempts(recovery_attempts_raw, roots, registry)
    terminal = recovery_terminal_statuses(recovery_attempts, registry)
    return {
        "status": "READY_FOR_REFRESH" if terminal["complete"] else "PENDING_RECOVERY_ATTEMPTS",
        "baseline_verification": verify_baseline(aggregator, manifest),
        "target_count": len(registry),
        "recovery_roots": [str(root) for root in roots],
        "recovery_records_observed": observed,
        "verified_recovery_records": sum(len(records) for records in recovery_attempts.values()),
        "recovery_files_observed": len([item for item in recovery_files if is_under(item.get("path", ""), roots)]),
        "recovery_diagnostics": [*baseline_diag, *recovery_diag],
        "recovery_validation_issues": recovery_validation_issues,
        "terminal": terminal,
        "baseline_snapshot_files": len(baseline_files),
    }


def refresh() -> dict[str, Any]:
    aggregator = load_aggregator()
    manifest = read_json(RECOVERY_MANIFEST)
    registry = read_registry()
    verification = verify_baseline(aggregator, manifest)
    if not verification["ok"]:
        raise RuntimeError(f"baseline verification failed: {verification}")
    baseline_attempts, baseline_diag, baseline_files, _ = collect_baseline_snapshot(aggregator)
    roots = [path for path in (RECOVERY_ROOT / "gemini", RECOVERY_ROOT / "qwen") if path.is_dir()]
    all_attempts, recovery_diag, recovery_files, checkpoint_terminal_statuses = aggregator.collect_raw_snapshot(extra_roots=roots)
    recovery_attempts_raw, observed = split_recovery_attempts(all_attempts, roots)
    recovery_attempts, recovery_validation_issues = validate_recovery_attempts(recovery_attempts_raw, roots, registry)
    terminal = recovery_terminal_statuses(recovery_attempts, registry)
    if not terminal["complete"]:
        return {
            "status": "PENDING_RECOVERY_ATTEMPTS",
            "terminal": terminal,
            "recovery_records_observed": observed,
            "verified_recovery_records": sum(len(records) for records in recovery_attempts.values()),
            "recovery_validation_issues": recovery_validation_issues,
        }
    if recovery_validation_issues:
        raise RuntimeError(f"recovery metadata validation issues: {recovery_validation_issues}")
    merged, merge_stats = merge_target_attempts(baseline_attempts, recovery_attempts, registry)
    if merge_stats["metadata_invalid_records"] or merge_stats["score_payload_overflow_records"]:
        raise RuntimeError(f"recovery policy violation: {merge_stats}")
    data = build_data(aggregator, merged, recovery_files, [*baseline_diag, *recovery_diag], checkpoint_terminal_statuses)
    invariance = same_subset_invariance(aggregator, data)
    if invariance["status"] != "PASS":
        raise RuntimeError(f"same-subset invariance failed; refusing recovery report write: {json.dumps(invariance, ensure_ascii=False, sort_keys=True)}")
    expanded = expanded_panel_summary(data)
    # The report writer is reused only with an explicit isolated destination.
    original_report_root = aggregator.REPORT_ROOT
    original_progress = aggregator.ANALYSIS_PROGRESS
    try:
        aggregator.REPORT_ROOT = OUTPUT_ROOT
        aggregator.ANALYSIS_PROGRESS = PROGRESS_PATH
        aggregator.write_reports(data)
    finally:
        aggregator.REPORT_ROOT = original_report_root
        aggregator.ANALYSIS_PROGRESS = original_progress
    report_manifest_path = OUTPUT_ROOT / "analysis_manifest.json"
    report_manifest = read_json(report_manifest_path)
    report_manifest["recovery"] = {
        "recovery_id": manifest.get("recovery_id"),
        "target_count": len(registry),
        "observed_recovery_records": observed,
        "verified_recovery_records": sum(len(records) for records in recovery_attempts.values()),
        "recovery_file_hashes": {
            item["path"]: item["sha256"]
            for item in recovery_files
            if is_under(item.get("path", ""), roots)
        },
        "baseline_manifest_sha256": sha256_file(BASELINE_ROOT / "reports" / "analysis_manifest.json"),
        "target_registry_sha256": sha256_file(TARGET_REGISTRY),
        "policy_sha256": sha256_file(RECOVERY_POLICY),
        "merge_stats": merge_stats,
        "terminal": terminal,
        "same_subset_invariance": invariance,
        "expanded_full_panel": expanded,
        "baseline_snapshot_diagnostics": baseline_diag,
        "recovery_snapshot_diagnostics": recovery_diag,
        "recovery_validation_issues": recovery_validation_issues,
    }
    recovery_before_after = make_recovery_before_after_markdown(data, invariance, expanded, terminal)
    before_after_path = OUTPUT_ROOT / "recovery_before_after.md"
    aggregator.atomic_write(before_after_path, recovery_before_after)
    report_manifest.setdefault("reports", []).append("recovery_before_after.md")
    report_manifest.setdefault("report_sha256", {})["recovery_before_after.md"] = sha256_file(before_after_path)
    aggregator.write_json(report_manifest_path, report_manifest)
    return {
        "status": "RECOVERY_REPORTS_WRITTEN",
        "report_dir": str(OUTPUT_ROOT),
        "merge_stats": merge_stats,
        "terminal": terminal,
        "recovery_records_observed": observed,
    }


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="read-only recovery status")
    parser.add_argument("--refresh", action="store_true", help="write isolated reports only when all targets are terminal")
    args = parser.parse_args(argv)
    if args.refresh:
        result = refresh()
    else:
        result = status_snapshot()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
