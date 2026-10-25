#!/usr/bin/env python3
"""Validate a blind judge-result import against the canonical pilot packet.

Validation is intentionally stricter than aggregation.  It checks packet
identity, blind item/candidate IDs, source hashes, explicit ``ok``/``failed``/
``missing`` statuses, duplicate logical observations, and AB/BA orientation.
It never maps a blind candidate back to a model name and never treats a model
or version self-reported inside a judge answer as authentication.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_pilot import _qualification_score_audit, load_bundle, packet_hashes, packet_index, read_json, sha256_file, sha256_text, write_json


HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
ALLOWED_STATUS = {"ok", "failed", "missing"}
FORBIDDEN_METADATA_KEYS = {"api_key", "authorization", "bearer_token", "candidate_mapping", "model_mapping", "expected_answer", "old_score", "prior_score"}


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _warning(warnings: list[str], message: str) -> None:
    warnings.append(message)


SCORE_LIMITS = {
    "terminology_locale": 30,
    "semantic_safety": 30,
    "fluency": 20,
    "completeness_format": 20,
}


def _score_shape_issues(row: Mapping[str, Any]) -> list[str]:
    """Validate scalar score/subscore shape for pointwise and qualification rows.

    Qualification rows are flattened before this function is called, so the
    same check covers every anonymous variant rather than only the parent
    ``qualification`` record.  The original score is never repaired here.
    """

    nested = row.get("pointwise") if isinstance(row.get("pointwise"), Mapping) else {}
    abstain = row.get("abstain")
    score = row.get("score_0_100", nested.get("score_0_100"))
    subscores = row.get("subscores", nested.get("subscores"))
    critical = row.get("critical", nested.get("critical"))
    errors_value = row.get("errors", nested.get("errors"))
    issues: list[str] = []

    if abstain is True:
        if score is not None or subscores is not None:
            issues.append("abstain judgment must use null score and subscores")
        return issues

    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or not 0 <= float(score) <= 100:
        issues.append("score_0_100 missing or outside 0..100")
    if not isinstance(subscores, Mapping):
        issues.append("subscores missing")
    else:
        values: list[float] = []
        for key, maximum in SCORE_LIMITS.items():
            value = subscores.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= maximum:
                issues.append(f"subscore {key} missing/outside 0..{maximum}")
            else:
                values.append(float(value))
        if len(values) == len(SCORE_LIMITS) and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) and abs(sum(values) - float(score)) > 1e-6:
            issues.append(f"subscores sum {sum(values):g} does not equal score_0_100 {float(score):g}")
    if not isinstance(critical, bool):
        issues.append("critical must be boolean")
    if not isinstance(errors_value, list):
        issues.append("errors must be an array")
    return issues


def _row_key(row: Mapping[str, Any], include_run: bool = True) -> tuple[Any, ...]:
    return (
        row.get("_track"),
        row.get("_item_id"),
        row.get("_candidate_id"),
        row.get("pair_id"),
        row.get("orientation"),
        row.get("_judge_id"),
        row.get("_run_id") if include_run else None,
    )


def _check_metadata(metadata: Mapping[str, Any], errors: list[str], warnings: list[str]) -> None:
    for key in FORBIDDEN_METADATA_KEYS:
        if key in metadata:
            _error(errors, f"top-level metadata contains forbidden secret/ground-truth key {key!r}")
    judge = metadata.get("judge")
    if isinstance(judge, Mapping):
        for key in FORBIDDEN_METADATA_KEYS:
            if key in judge:
                _error(errors, f"judge metadata contains forbidden key {key!r}")
        if not judge.get("judge_id"):
            _error(errors, "top-level judge metadata must include judge_id")
        if not judge.get("family"):
            _warning(warnings, "judge family missing; dependency sensitivity cannot be summarized")
        if not judge.get("ui_model_version") and judge.get("provider") == "gemini_ui":
            _warning(warnings, "Gemini UI-displayed model version is not filled; this is metadata only and cannot be verified from the prompt")
    elif not metadata.get("judge_id"):
        _warning(warnings, "top-level judge_id missing; rows must carry judge_id")
    if metadata.get("model") or metadata.get("model_name"):
        _warning(warnings, "judge/model metadata is present in the import envelope; keep it out of blind prompt content and treat it as external audit metadata")
    nested_provider = judge.get("provider") if isinstance(judge, Mapping) else None
    external_ui_envelope = "task_id" in metadata or "model_label_from_ui" in metadata or nested_provider == "gemini_ui"
    if external_ui_envelope:
        for required_key in ("task_id", "model_label_from_ui"):
            value = metadata.get(required_key)
            if not isinstance(value, str) or not value.strip():
                _error(errors, f"external UI import must include non-empty top-level {required_key}")


def _pair_orientation_checks(rows: Sequence[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    pair_groups: dict[tuple[str, str, str], dict[str, set[tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["_track"] != "pairwise" or row["_status"] != "ok":
            continue
        orientation = str(row.get("orientation", "")).upper()
        if orientation not in {"AB", "BA"}:
            _error(errors, f"pairwise row {row.get('_item_id')}/{row.get('pair_id')} has invalid orientation {orientation!r}")
            continue
        candidate_a = row.get("candidate_a")
        candidate_b = row.get("candidate_b")
        if not isinstance(candidate_a, str) or not isinstance(candidate_b, str) or candidate_a == candidate_b:
            _error(errors, f"pairwise row {row.get('_item_id')}/{row.get('pair_id')} must have two distinct candidate IDs")
            continue
        pair_key = (str(row.get("_item_id")), str(row.get("pair_id")), str(row.get("_judge_id")))
        pair_groups[pair_key][orientation].add((candidate_a, candidate_b))
        winner = row.get("winner")
        if winner not in {"A", "B", "TIE", "tie", "Tie"} and winner not in {candidate_a, candidate_b}:
            _error(errors, f"pairwise row {row.get('_item_id')}/{row.get('pair_id')} has unknown winner {winner!r}")
    for (item_id, pair_id, judge_id), orientations in pair_groups.items():
        if "AB" in orientations and "BA" in orientations:
            ab = next(iter(orientations["AB"]))
            ba = next(iter(orientations["BA"]))
            if ba != (ab[1], ab[0]):
                _error(errors, f"pairwise AB/BA candidates are not canonical swaps for {item_id}/{pair_id}/{judge_id}: AB={ab}, BA={ba}")
        elif "AB" not in orientations and "BA" not in orientations:
            _warning(warnings, f"pairwise pair {item_id}/{pair_id}/{judge_id} has no AB or BA row")


def _track_eligibility(
    rows: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
    score_shape_checked: Mapping[str, int],
    score_shape_invalid: Mapping[str, int],
) -> dict[str, Any]:
    """Return explicit track-level import/use status.

    ``status=PASS`` is deliberately an envelope/binding result.  It must not
    be interpreted as every track being scalar-valid.  Qualification scalar
    mismatches remain warnings so that independent error/evidence/critical
    fields can be inspected, but they make the qualification scalar path
    ``partial`` and the candidate ineligible for a formal scalar
    qualification run until all variants are valid.  Any binding/structural
    error quarantines the whole bundle, including otherwise valid tracks.
    """

    by_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[str(row.get("_track"))].append(row)

    result: dict[str, Any] = {}
    for track in ("pointwise", "qualification", "pairwise"):
        track_rows = by_track.get(track, [])
        non_ok = Counter(str(row.get("_status")) for row in track_rows if row.get("_status") != "ok")
        checked = int(score_shape_checked.get(track, 0))
        invalid = int(score_shape_invalid.get(track, 0))
        if not track_rows:
            status = "unavailable"
        elif errors:
            status = "quarantined"
        elif invalid or non_ok:
            status = "partial"
        else:
            status = "eligible"
        result[track] = {
            "status": status,
            "n_rows": len(track_rows),
            "n_ok": sum(1 for row in track_rows if row.get("_status") == "ok"),
            "n_failed": non_ok.get("failed", 0),
            "n_missing": non_ok.get("missing", 0),
            "score_shape_checked": checked,
            "score_shape_invalid": invalid,
        }

    qualification = result["qualification"]
    qualification["scalar_score_status"] = qualification["status"]
    qualification["candidate_eligible_for_formal_scalar_qualification"] = qualification["status"] == "eligible"
    if qualification["status"] == "quarantined":
        qualification["evidence_critical_status"] = "quarantined"
    elif not qualification["n_rows"]:
        qualification["evidence_critical_status"] = "unavailable"
    elif qualification["n_failed"] or qualification["n_missing"]:
        qualification["evidence_critical_status"] = "partial"
    else:
        # Scalar sums may be invalid while raw error/evidence/critical fields
        # remain independently assessable.  This is not a qualification pass.
        qualification["evidence_critical_status"] = "eligible_as_diagnostic_only"
    qualification["formal_result_status"] = "not_evaluated_without_private_key"
    result["note"] = (
        "Top-level PASS means packet/hash/source/candidate/orientation import "
        "passed. Inspect each track here: qualification scalar partiality does "
        "not invalidate pointwise rows, but it is not formal qualification."
    )
    return result


def validate_import(packet_path: str | Path, input_path: str | Path) -> dict[str, Any]:
    packet = read_json(packet_path)
    packet_info = packet_index(packet)
    expected_hashes = packet_hashes(packet_path)
    raw_packet_hash = sha256_file(packet_path)
    metadata, rows = load_bundle(input_path)
    errors: list[str] = []
    warnings: list[str] = []
    _check_metadata(metadata, errors, warnings)

    supplied_hash = metadata.get("packet_sha256") or metadata.get("packet_hash")
    if not isinstance(supplied_hash, str) or not HEX64.fullmatch(supplied_hash):
        _error(errors, "import must include packet_sha256 (64 lowercase/hex characters)")
    elif supplied_hash.lower() != raw_packet_hash:
        _error(errors, "packet_sha256 does not match the frozen canonical packet file hash")
    expected_packet_id = packet.get("packet_id") if isinstance(packet, Mapping) else None
    supplied_packet_id = metadata.get("packet_id")
    if expected_packet_id and supplied_packet_id != expected_packet_id:
        _error(errors, f"packet_id mismatch: expected {expected_packet_id!r}, got {supplied_packet_id!r}")

    declared_judge = metadata.get("judge_id")
    if declared_judge is None and isinstance(metadata.get("judge"), Mapping):
        declared_judge = metadata["judge"].get("judge_id")
    seen_physical: Counter[tuple[Any, ...]] = Counter()
    seen_logical: Counter[tuple[Any, ...]] = Counter()
    observed_source_hashes: dict[str, str] = {}
    observed_source_ids: dict[str, str] = {}
    valid_rows = 0
    failed_rows = 0
    missing_rows = 0
    unknown_item_ids: set[str] = set()
    unknown_candidate_ids: set[str] = set()
    unknown_pair_ids: set[str] = set()
    score_shape_checked = Counter()
    score_shape_valid = Counter()
    score_shape_invalid = Counter()
    score_shape_errors: list[dict[str, Any]] = []
    qualification_score_audit_counts: Counter[str] = Counter()
    qualification_score_audit_examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        status = row.get("_status")
        if status not in ALLOWED_STATUS:
            _error(errors, f"row {index}: status must be one of {sorted(ALLOWED_STATUS)}, got {status!r}")
        if row.get("_item_id") is None:
            _error(errors, f"row {index}: missing item_id")
        elif packet_info["item_ids"] and row["_item_id"] not in packet_info["item_ids"]:
            unknown_item_ids.add(str(row["_item_id"]))
        if row.get("_judge_id") is None:
            _error(errors, f"row {index}: missing judge_id")
        elif declared_judge is not None and str(row["_judge_id"]) != str(declared_judge):
            _error(errors, f"row {index}: row judge_id {row['_judge_id']!r} disagrees with envelope judge_id {declared_judge!r}")
        if row.get("_source_id") is None:
            _warning(warnings, f"row {index}: source_id inferred from item_id; bootstrap scope may be wrong")
        if row.get("_source_hash") is None:
            _error(errors, f"row {index}: source_text_sha256/source_hash is required; raw source text is not sufficient for audit")
        else:
            item_id = str(row.get("_item_id"))
            item_object = row.get("item") if isinstance(row.get("item"), Mapping) else {}
            raw_source = row.get("source_en") or row.get("source_text") or row.get("source") or item_object.get("source_en") or item_object.get("source_text") or item_object.get("source")
            if isinstance(raw_source, str) and raw_source:
                raw_source_hash = sha256_text(raw_source)
                if raw_source_hash != row["_source_hash"]:
                    _error(errors, f"row {index}: supplied source_text_sha256 does not match supplied source text; source binding is quarantined")
            old_hash = observed_source_hashes.get(item_id)
            if old_hash and old_hash != row["_source_hash"]:
                _error(errors, f"row {index}: same item_id has inconsistent source_text_sha256 ({old_hash} vs {row['_source_hash']})")
            observed_source_hashes[item_id] = row["_source_hash"]
            if item_id in packet_info["source_hashes"] and packet_info["source_hashes"][item_id] not in {"__CONFLICT__", row["_source_hash"]}:
                _error(errors, f"row {index}: source_text_sha256 does not match canonical packet for item {item_id}")
        if row.get("_source_id") is not None:
            observed_source_ids[str(row.get("_item_id"))] = str(row.get("_source_id"))
        if row.get("_candidate_id") is not None:
            if packet_info["candidate_ids"] and row["_candidate_id"] not in packet_info["candidate_ids"]:
                unknown_candidate_ids.add(str(row["_candidate_id"]))
            candidate_object = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
            supplied_target = row.get("target_zh_tw") or row.get("target") or row.get("translation") or candidate_object.get("raw_translation_zh_tw") or candidate_object.get("target_zh_tw") or candidate_object.get("target") or candidate_object.get("translation")
            expected_target = packet_info["candidate_targets"].get((str(row.get("_item_id")), str(row.get("_candidate_id"))))
            if isinstance(supplied_target, str) and expected_target is not None and supplied_target != expected_target:
                _error(errors, f"row {index}: supplied target text does not match canonical blind packet for {row.get('_item_id')}/{row.get('_candidate_id')}; do not override packet content")
            target_for_quote_check = supplied_target if isinstance(supplied_target, str) else expected_target
            if isinstance(target_for_quote_check, str):
                for error_number, error in enumerate(row.get("_errors", []), 1):
                    if not isinstance(error, Mapping):
                        continue
                    quote = error.get("target_exact_quote")
                    if isinstance(quote, str) and quote and quote not in target_for_quote_check:
                        _error(errors, f"row {index} error {error_number}: target_exact_quote is not present in the canonical target; quote validation failed")
        if row["_track"] == "pairwise":
            pair_id = row.get("pair_id")
            if packet_info["pair_ids"] and pair_id not in packet_info["pair_ids"]:
                unknown_pair_ids.add(str(pair_id))
        if status == "ok":
            valid_rows += 1
            if row["_track"] == "pointwise":
                score_shape_checked["pointwise"] += 1
                shape_issues = _score_shape_issues(row)
                if shape_issues:
                    score_shape_invalid["pointwise"] += 1
                    for issue in shape_issues:
                        detail = {"row": index, "track": "pointwise", "item_id": row.get("_item_id"), "blind_id": row.get("_candidate_id"), "issue": issue}
                        score_shape_errors.append(detail)
                        _error(errors, f"row {index} pointwise score shape: {issue}")
                else:
                    score_shape_valid["pointwise"] += 1
            elif row["_track"] == "qualification":
                # Qualification scalar validity is intentionally a warning:
                # malformed score sums do not invalidate the same row's
                # critical/error/evidence assessment.
                score_shape_checked["qualification"] += 1
                audit = row.get("_qualification_score_audit")
                if not isinstance(audit, Mapping):
                    audit = _qualification_score_audit(row)
                audit_status = str(audit.get("status", "missing"))
                qualification_score_audit_counts[audit_status] += 1
                if audit_status == "valid":
                    score_shape_valid["qualification"] += 1
                else:
                    score_shape_invalid["qualification"] += 1
                    detail = {"row": index, "track": "qualification", "item_id": row.get("_item_id"), "blind_id": row.get("_candidate_id"), "status": audit_status, "score_0_100": audit.get("score_0_100"), "subscores": audit.get("subscores"), "subscores_sum": audit.get("subscores_sum"), "reason": audit.get("reason")}
                    score_shape_errors.append(detail)
                    qualification_score_audit_examples.append(detail)
                    _warning(warnings, f"row {index} qualification scalar score audit {audit_status}: {audit.get('reason')}; raw score retained, scalar excluded, evidence/critical fields remain assessable")
            if row["_track"] == "pairwise" and row.get("winner") is None:
                # The normalized row allows a nested pairwise object, but a
                # missing winner is still not a valid success.
                nested = row.get("pairwise") if isinstance(row.get("pairwise"), Mapping) else {}
                if nested.get("winner") is None:
                    _error(errors, f"row {index}: ok pairwise row requires winner A/B/TIE or canonical candidate ID")
        elif status == "failed":
            failed_rows += 1
            if not row.get("reason") and not row.get("error") and not row.get("failure_reason"):
                _warning(warnings, f"row {index}: failed status has no reason; preserve provider/parser error text")
        else:
            missing_rows += 1
            if not row.get("reason") and not row.get("missing_reason"):
                _warning(warnings, f"row {index}: missing status has no reason")
        physical = _row_key(row, True)
        logical = _row_key(row, False)
        seen_physical[physical] += 1
        seen_logical[logical] += 1

    for key, count in seen_physical.items():
        if count > 1:
            _error(errors, f"duplicate same-run record key appears {count} times: {key}")
    for key, count in seen_logical.items():
        if count > 1:
            _warning(warnings, f"repeat runs share one logical judge unit ({count} rows): {key}; aggregation will collapse runs and critical votes will count once")
    if unknown_item_ids:
        _error(errors, f"unknown packet item IDs: {sorted(unknown_item_ids)}")
    if unknown_candidate_ids:
        _error(errors, f"unknown packet candidate IDs: {sorted(unknown_candidate_ids)}")
    if unknown_pair_ids:
        _error(errors, f"unknown packet pair IDs: {sorted(unknown_pair_ids)}")
    for item_id, packet_hash in packet_info["source_hashes"].items():
        if packet_hash == "__CONFLICT__":
            _error(errors, f"canonical packet itself contains inconsistent source hashes for item {item_id}")
    _pair_orientation_checks(rows, errors, warnings)

    judge_ids = sorted({str(row.get("_judge_id")) for row in rows if row.get("_judge_id") is not None})
    track_eligibility = _track_eligibility(rows, errors, score_shape_checked, score_shape_invalid)
    return {
        "schema_version": "judge-pilot-import-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "packet": {"path": str(packet_path), "raw_sha256": raw_packet_hash, "accepted_hashes": sorted(expected_hashes), "supplied_hash": supplied_hash, "hash_match": isinstance(supplied_hash, str) and supplied_hash.lower() == raw_packet_hash, "n_packet_items": len(packet_info["item_ids"]), "n_packet_candidates": len(packet_info["candidate_ids"]), "n_packet_pairs": len(packet_info["pair_ids"])},
        "records": {"total": len(rows), "valid_ok": valid_rows, "failed": failed_rows, "missing": missing_rows, "n_judges": len(judge_ids), "judge_ids": judge_ids, "tracks": dict(Counter(row["_track"] for row in rows)), "qualification_score_audit": {"n_variants": sum(qualification_score_audit_counts.values()), "valid": qualification_score_audit_counts.get("valid", 0), "invalid": qualification_score_audit_counts.get("invalid", 0), "missing": qualification_score_audit_counts.get("missing", 0), "abstain": qualification_score_audit_counts.get("abstain", 0), "status_counts": dict(sorted(qualification_score_audit_counts.items())), "invalid_examples": qualification_score_audit_examples, "note": "Per-variant score/subscore sums are audited after flattening qualification.judgments. Invalid/missing scalar scores are not silently repaired or counted; critical/error/evidence fields remain separate."}},
        "score_shape": {
            "status": "PASS" if not score_shape_errors else ("PASS_WITH_WARNINGS" if not any(detail.get("track") == "pointwise" for detail in score_shape_errors) else "FAIL"),
            "checked_rows": sum(score_shape_checked.values()),
            "valid_rows": sum(score_shape_valid.values()),
            "invalid_rows": sum(score_shape_invalid.values()),
            "by_track": {
                track: {"checked": score_shape_checked[track], "valid": score_shape_valid[track], "invalid": score_shape_invalid[track]}
                for track in ("pointwise", "qualification")
                if score_shape_checked[track]
            },
            "errors": [detail for detail in score_shape_errors if detail.get("track") == "pointwise"],
            "warnings": [detail for detail in score_shape_errors if detail.get("track") == "qualification"],
            "note": "Qualification variants are flattened and audited independently. Invalid scalar sums are retained and flagged; no score is repaired by validation.",
        },
        "track_eligibility": track_eligibility,
        "source_scope": {"n_source_ids": len(set(observed_source_ids.values())), "source_ids": sorted(set(observed_source_ids.values())), "n_item_source_hashes": len(observed_source_hashes), "same_item_hash_consistent": not any("same item_id has inconsistent" in error for error in errors)},
        "duplicates": {"same_run_duplicates": sum(max(0, count - 1) for count in seen_physical.values()), "logical_repeat_rows": sum(max(0, count - 1) for count in seen_logical.values()), "note": "logical repeats are allowed only when run_id differs; they are not additional judges"},
        "errors": errors,
        "warnings": warnings,
        "notes": ["A source_text_sha256 is an audit binding; do not copy source text or expected answers into judge output.", "The UI-displayed model version is user-filled metadata and is not verified by a model self-report in the prompt.", "A separate AB and BA import may be incomplete by itself; aggregation reports incomplete swaps rather than inventing ties."],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, help="canonical blind judge_packet.json")
    parser.add_argument("--input", required=True, help="judge result JSON bundle")
    parser.add_argument("--output", required=True, help="validation audit JSON")
    args = parser.parse_args(argv)
    try:
        result = validate_import(args.packet, args.input)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
