#!/usr/bin/env python3
"""Build owner-only pointwise comparison CSVs and validity tables.

The input bundles remain blind.  Candidate mapping is joined only in the
owner-only outputs after judging.  The long/wide score tables remain compact
and omit source/target text; ``translation_comparison`` is the separate
owner-only table that joins the exact frozen source and candidate translation
for human result review.  Qualification rows are intentionally excluded from
the score comparison: their coverage and scalar-shape audit is reported
separately.

Every input bundle is validated before the first output is written.  Packet,
source, target, quote, duplicate, and other binding/structural failures
quarantine the whole bundle.  A pointwise scalar-shape failure is retained as
an owner-only diagnostic row, but is never eligible for a score mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_pilot import load_bundle, packet_index, read_json
from validate_import import _score_shape_issues, validate_import


SCORE_KEYS = ("terminology_locale", "semantic_safety", "fluency", "completeness_format")
SCORE_MEAN_POLICY = "validated_binding_and_row_shape_pass"
ERROR_ANNOTATION_FIELDS = (
    "item_id", "source_id", "candidate_id", "model_alias", "judge_id", "judge_family", "error_index",
    "severity", "type", "target_exact_quote", "source_evidence", "source_span", "target_anchor", "confidence",
    "exact_quote_status", "source_text_sha256", "bundle", "mapping_scope",
)


def _value(row: Mapping[str, Any], nested: Mapping[str, Any], key: str) -> Any:
    return row[key] if key in row else nested.get(key)


def _slug(value: Any) -> str:
    if "gpt-6.1-sol" in str(value):
        return "gpt_6_1_sol"
    if "qwen3.8-flash-next" in str(value):
        return "qwen3_8_flash_next"
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "judge")).strip("_").lower()
    return text or "judge"


def _judge_slug_map(judge_ids: Sequence[Any]) -> dict[str, str]:
    """Return stable, collision-free column slugs for judge IDs."""

    result: dict[str, str] = {}
    used: set[str] = set()
    for value in sorted({str(judge_id) for judge_id in judge_ids}):
        base = _slug(value)
        slug = base
        if slug in used:
            suffix = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "judge"
            slug = f"{base}_{suffix}"
            counter = 2
            while slug in used:
                slug = f"{base}_{suffix}_{counter}"
                counter += 1
        result[value] = slug
        used.add(slug)
    return result


def _judge_context(
    judge_family: Any,
    judge_metadata: Mapping[str, Any] | None = None,
    row: Mapping[str, Any] | None = None,
) -> str:
    """Describe how a judge was packaged operationally.

    Prefer an explicit context supplied by the import envelope or row.  The
    Gemini UI adapter records one fresh chat per source/orientation, so retain
    that operational distinction even when the family/provider labels are the
    only available metadata.
    """

    for source in (row, judge_metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("judge_context_packaging", "context_packaging", "operational_context", "context"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if source.get("fresh_chat_per_source_or_orientation") is True:
            return "fresh_chat_per_source_or_orientation"

    family = str(judge_family or "").lower().replace("_", "-")
    provider = str((judge_metadata or {}).get("provider") or "").lower().replace("_", "-")
    if family in {"google-gemini", "google", "gemini"} or provider in {"gemini-ui", "google-gemini", "gemini"}:
        return "fresh_chat_per_source_or_orientation"
    if family in {"gpt", "openai"}:
        return "fresh_full_packet_context"
    if family in {"qwen", "mistral"}:
        return "fresh_request_per_candidate"
    return "not_recorded"


def _quote_invalid_count(row: Mapping[str, Any], target: str | None) -> int:
    if not isinstance(target, str):
        return 0
    invalid = 0
    for error in row.get("_errors", []):
        if not isinstance(error, Mapping):
            continue
        quote = error.get("target_exact_quote")
        if isinstance(quote, str) and quote and quote not in target:
            invalid += 1
    return invalid


def _is_pointwise_shape_error(error: Any) -> bool:
    """Return whether a validator error is a row-local pointwise shape error.

    Pointwise scalar-shape failures are retained as owner-only diagnostics so
    that a malformed score can be audited without being included in a mean.
    Every other validator error binds the result bundle to the frozen packet
    (or makes its logical rows ambiguous) and therefore quarantines the whole
    bundle before any output is written.
    """

    return bool(re.match(r"^row \d+ pointwise score shape:", str(error)))


def _validation_quarantine_errors(validation: Mapping[str, Any]) -> list[str]:
    """Return validation failures that require rejecting the whole bundle.

    ``validate_import`` deliberately reports invalid qualification scalar
    shapes as warnings, because qualification evidence/critical diagnostics
    remain useful.  It reports invalid pointwise scalar shapes as errors.  A
    pointwise shape error is row-local and can safely be retained in the long
    table with ``score_eligible_for_mean=False``; binding, duplicate, target,
    source, quote, and structural errors cannot.
    """

    status = str(validation.get("status", "")).upper()
    raw_errors = validation.get("errors", [])
    errors = [str(error) for error in raw_errors] if isinstance(raw_errors, list) else [str(raw_errors)]
    non_shape_errors = [error for error in errors if not _is_pointwise_shape_error(error)]
    if non_shape_errors:
        return non_shape_errors
    if status == "FAIL" and not errors:
        return ["validator returned FAIL without an error detail"]
    if status not in {"PASS", "FAIL"}:
        return [f"validator returned unexpected status {status!r}"]
    if status == "PASS" and errors:
        return errors
    return []


def _pointwise_row(
    row: Mapping[str, Any],
    bundle_path: Path,
    judge_metadata: Mapping[str, Any],
    mapping: Mapping[str, Any],
    packet_candidates: Mapping[tuple[str, str], str],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    nested = row.get("pointwise") if isinstance(row.get("pointwise"), Mapping) else {}
    candidate_id = str(row.get("_candidate_id") or "")
    item_id = str(row.get("_item_id") or "")
    target = packet_candidates.get((item_id, candidate_id))
    subscores = _value(row, nested, "subscores")
    if not isinstance(subscores, Mapping):
        subscores = {}
    errors = row.get("_errors", [])
    if not isinstance(errors, list):
        errors = []
    candidate_map = mapping.get(candidate_id) if isinstance(mapping, Mapping) else {}
    if not isinstance(candidate_map, Mapping):
        candidate_map = {}
    judge_id = str(row.get("_judge_id") or judge_metadata.get("judge_id") or "")
    judge_family = str(row.get("_family") or judge_metadata.get("family") or "")
    context = _judge_context(judge_family, judge_metadata, row)
    shape_issues = _score_shape_issues(row)
    quote_invalid = _quote_invalid_count(row, target)
    eligibility_reasons: list[str] = []
    if validation is not None:
        eligibility_reasons.extend(_validation_quarantine_errors(validation))
        track_eligibility = validation.get("track_eligibility", {})
        pointwise_track = track_eligibility.get("pointwise", {}) if isinstance(track_eligibility, Mapping) else {}
        pointwise_track_status = pointwise_track.get("status") if isinstance(pointwise_track, Mapping) else None
        # A validator's ``quarantined`` track status normally reflects a
        # binding error, which is already represented above.  The one
        # row-local exception is a bundle containing only pointwise shape
        # errors: retain valid rows, while each malformed row is excluded by
        # ``shape_issues`` below.
        if pointwise_track_status == "unavailable":
            eligibility_reasons.append("pointwise track is unavailable")
        elif pointwise_track_status == "quarantined" and not validation.get("errors"):
            eligibility_reasons.append("pointwise track is quarantined")
    row_status = str(row.get("_status") or row.get("status") or "").lower()
    if row_status != "ok":
        eligibility_reasons.append(f"row status is {row_status or 'missing'}")
    if shape_issues:
        eligibility_reasons.extend(shape_issues)
    if quote_invalid:
        eligibility_reasons.append(f"{quote_invalid} exact-quote locator error(s)")
    return {
        "candidate_id": candidate_id,
        "candidate_family": candidate_map.get("candidate_family", ""),
        "model_alias": candidate_map.get("model_alias", ""),
        "source_id": row.get("_source_id", ""),
        "item_id": item_id,
        "judge_id": judge_id,
        "judge_family": judge_family,
        "judge_context_packaging": context,
        "score_0_100": _value(row, nested, "score_0_100"),
        **{key: subscores.get(key, "") for key in SCORE_KEYS},
        "critical": _value(row, nested, "critical"),
        "error_count": len(errors),
        "score_eligible_for_mean": not eligibility_reasons,
        "score_eligibility_issues": " | ".join(eligibility_reasons),
        "score_shape_status": "PASS" if not shape_issues else "FAIL",
        "score_shape_issues": " | ".join(shape_issues),
        "exact_quote_status": "PASS" if quote_invalid == 0 else "FAIL",
        "exact_quote_invalid_count": quote_invalid,
        "source_text_sha256": row.get("_source_hash", ""),
        "bundle": str(bundle_path),
        "mapping_scope": "owner_only_postjudge",
        "comparison_scope": "exploratory_pointwise_only",
    }


def _error_annotation_rows(
    row: Mapping[str, Any],
    bundle_path: Path,
    judge_metadata: Mapping[str, Any],
    mapping: Mapping[str, Any],
    packet_candidates: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Flatten original pointwise judge error objects for owner review.

    This table deliberately carries only the judge's own annotation fields;
    it does not add a reference/gold label, adjudicate error semantics, or
    duplicate source/target translations.  Binding and quote validation has
    already completed before this helper is called, so an invalid non-empty
    quote is represented as ``FAIL`` only for defensive direct callers.
    """

    errors = row.get("_errors", [])
    if not isinstance(errors, list):
        return []
    candidate_id = str(row.get("_candidate_id") or "")
    item_id = str(row.get("_item_id") or "")
    target = packet_candidates.get((item_id, candidate_id))
    candidate_map = mapping.get(candidate_id) if isinstance(mapping, Mapping) else {}
    if not isinstance(candidate_map, Mapping):
        candidate_map = {}
    judge_id = str(row.get("_judge_id") or judge_metadata.get("judge_id") or "")
    judge_family = str(row.get("_family") or judge_metadata.get("family") or "")

    annotation_rows: list[dict[str, Any]] = []
    for error_index, error in enumerate(errors):
        if isinstance(error, Mapping):
            values = error
            quote = values.get("target_exact_quote")
            severity = values.get("severity", "")
            error_type = values.get("type", "")
            source_evidence = values.get("source_evidence", "")
            source_span = values.get("source_span", "")
            target_anchor = values.get("target_anchor", "")
            confidence = values.get("confidence", "")
        else:
            # Imported Gemini objects are required to be mappings.  Keep a
            # primitive diagnostic losslessly for defensive callers rather
            # than silently dropping the original judge error string.
            quote = ""
            severity = ""
            error_type = error
            source_evidence = ""
            source_span = ""
            target_anchor = ""
            confidence = ""
        quote_status = "PASS"
        if quote:
            quote_status = "PASS" if isinstance(quote, str) and isinstance(target, str) and quote in target else "FAIL"
        annotation_rows.append({
            "item_id": row.get("_item_id", ""),
            "source_id": row.get("_source_id", ""),
            "candidate_id": candidate_id,
            "model_alias": candidate_map.get("model_alias", ""),
            "judge_id": judge_id,
            "judge_family": judge_family,
            "error_index": error_index,
            "severity": severity,
            "type": error_type,
            "target_exact_quote": quote,
            "source_evidence": source_evidence,
            "source_span": source_span,
            "target_anchor": target_anchor,
            "confidence": confidence,
            "exact_quote_status": quote_status,
            "source_text_sha256": row.get("_source_hash", ""),
            "bundle": str(bundle_path),
            "mapping_scope": "owner_only_postjudge",
        })
    return annotation_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validity_entry(path: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    tracks = validation.get("records", {}).get("tracks", {})
    shape = validation.get("score_shape", {})
    by_track = shape.get("by_track", {}) if isinstance(shape, Mapping) else {}
    quote_errors = [error for error in validation.get("errors", []) if "quote validation failed" in str(error)]
    judge_ids = validation.get("records", {}).get("judge_ids", [])
    judge_id = judge_ids[0] if len(judge_ids) == 1 else "|".join(str(value) for value in judge_ids)
    notes: list[str] = []
    qualification_shape = by_track.get("qualification", {})
    track_eligibility = validation.get("track_eligibility", {}) if isinstance(validation.get("track_eligibility"), Mapping) else {}
    pointwise_eligibility = track_eligibility.get("pointwise", {}) if isinstance(track_eligibility.get("pointwise"), Mapping) else {}
    qualification_eligibility = track_eligibility.get("qualification", {}) if isinstance(track_eligibility.get("qualification"), Mapping) else {}
    if qualification_shape.get("invalid", 0):
        notes.append(f"{qualification_shape['invalid']} qualification variants have invalid scalar score/subscore sums; raw values retained")
    if quote_errors:
        notes.append(f"{len(quote_errors)} exact-quote locator error(s) remain in this bundle")
    return {
        "judge_id": judge_id,
        "bundle": str(path),
        "source_kind": "derived_import" if ".import" in path.name else "raw",
        "validation_status": validation.get("status"),
        "pointwise_rows": tracks.get("pointwise", 0),
        "qualification_variants": tracks.get("qualification", 0),
        "pairwise_rows": tracks.get("pairwise", 0),
        "score_shape_status": shape.get("status"),
        "pointwise_track_status": pointwise_eligibility.get("status"),
        "qualification_track_status": qualification_eligibility.get("status"),
        "qualification_scalar_score_status": qualification_eligibility.get("scalar_score_status"),
        "qualification_evidence_critical_status": qualification_eligibility.get("evidence_critical_status"),
        "candidate_eligible_for_formal_scalar_qualification": qualification_eligibility.get("candidate_eligible_for_formal_scalar_qualification"),
        "pointwise_score_shape_status": "PASS" if by_track.get("pointwise", {}).get("invalid", 0) == 0 else "FAIL",
        "pointwise_score_shape_valid": by_track.get("pointwise", {}).get("valid", 0),
        "pointwise_score_shape_checked": by_track.get("pointwise", {}).get("checked", 0),
        "qualification_score_shape_status": "PASS" if qualification_shape.get("invalid", 0) == 0 else "FAIL",
        "qualification_score_shape_valid": qualification_shape.get("valid", 0),
        "qualification_score_shape_checked": qualification_shape.get("checked", 0),
        "exact_quote_status": "PASS" if not quote_errors else "FAIL",
        "exact_quote_error_count": len(quote_errors),
        "notes": " | ".join(notes),
        "mapping_scope": "not_included",
    }


def build(
    packet_path: Path,
    mapping_path: Path,
    input_paths: Sequence[Path],
    long_path: Path,
    wide_path: Path,
    translation_comparison_path: Path,
    validity_json_path: Path,
    validity_csv_path: Path,
    errors_output: Path | None = None,
) -> dict[str, Any]:
    packet = read_json(packet_path)
    packet_info = packet_index(packet)
    mapping_doc = read_json(mapping_path)
    mapping = mapping_doc.get("mapping", {}) if isinstance(mapping_doc, Mapping) else {}
    long_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    prepared_inputs: list[tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    for path in input_paths:
        metadata, rows = load_bundle(path)
        validation = validate_import(packet_path, path)
        quarantine_errors = _validation_quarantine_errors(validation)
        if quarantine_errors:
            details = " | ".join(quarantine_errors[:8])
            if len(quarantine_errors) > 8:
                details += f" | ... ({len(quarantine_errors) - 8} more)"
            raise ValueError(f"refusing to build from quarantined judge bundle {path}: {details}")
        prepared_inputs.append((path, metadata, rows, validation))

    # A wide row has one value per judge.  A repeated pointwise unit from the
    # same judge would otherwise silently overwrite whichever row was seen
    # first, while treating a second run as a new rater would inflate the
    # panel.  Reject the ambiguity before writing even the first output.
    seen_pointwise_units: dict[tuple[str, str, str], tuple[Path, int]] = {}
    for path, _metadata, rows, _validation in prepared_inputs:
        for index, row in enumerate(rows, 1):
            if row.get("_track") != "pointwise":
                continue
            key = (str(row.get("_judge_id") or ""), str(row.get("_item_id") or ""), str(row.get("_candidate_id") or ""))
            previous = seen_pointwise_units.get(key)
            if previous is not None:
                previous_path, previous_index = previous
                raise ValueError(
                    "duplicate pointwise judge unit would overwrite a wide value: "
                    f"judge/item/candidate={key!r} at {previous_path} row {previous_index} "
                    f"and {path} row {index}"
                )
            seen_pointwise_units[key] = (path, index)

    for path, metadata, rows, validation in prepared_inputs:
        validity_rows.append(_validity_entry(path, validation))
        judge_metadata = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
        for row in rows:
            if row.get("_track") != "pointwise":
                continue
            long_rows.append(_pointwise_row(row, path, judge_metadata, mapping, packet_info["candidate_targets"], validation))
            if errors_output is not None:
                error_rows.extend(_error_annotation_rows(row, path, judge_metadata, mapping, packet_info["candidate_targets"]))

    long_fields = [
        "candidate_id", "candidate_family", "model_alias", "source_id", "item_id", "judge_id", "judge_family",
        "judge_context_packaging", "score_0_100", *SCORE_KEYS, "critical", "error_count",
        "score_eligible_for_mean", "score_eligibility_issues", "score_shape_status",
        "score_shape_issues", "exact_quote_status", "exact_quote_invalid_count", "source_text_sha256", "bundle",
        "mapping_scope", "comparison_scope",
    ]
    _write_csv(long_path, long_rows, long_fields)

    judge_slug_by_id = _judge_slug_map([row["judge_id"] for row in long_rows])
    judge_slugs = sorted(judge_slug_by_id.values())
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in long_rows:
        key = (str(row["candidate_id"]), str(row["source_id"]), str(row["item_id"]))
        base = grouped.setdefault(key, {
            "candidate_id": row["candidate_id"],
            "candidate_family": row["candidate_family"],
            "model_alias": row["model_alias"],
            "source_id": row["source_id"],
            "item_id": row["item_id"],
            "mapping_scope": "owner_only_postjudge",
            "comparison_scope": "exploratory_pointwise_only",
        })
        slug = judge_slug_by_id[str(row["judge_id"])]
        base[f"{slug}_score_0_100"] = row["score_0_100"]
        base[f"{slug}_score_eligible_for_mean"] = row["score_eligible_for_mean"]
        base[f"{slug}_score_shape_status"] = row["score_shape_status"]
        base[f"{slug}_exact_quote_status"] = row["exact_quote_status"]
        base[f"{slug}_critical"] = row["critical"]

    score_fields = [f"{slug}_score_0_100" for slug in judge_slugs]
    for base in grouped.values():
        eligible_fields = [
            field for field in score_fields
            if isinstance(base.get(field), (int, float))
            and not isinstance(base.get(field), bool)
            and base.get(field.replace("_score_0_100", "_score_eligible_for_mean")) is True
        ]
        raw_fields = [
            field for field in score_fields
            if isinstance(base.get(field), (int, float)) and not isinstance(base.get(field), bool)
        ]
        scores = [float(base[field]) for field in eligible_fields]
        raw_scores = [float(base[field]) for field in raw_fields]
        base["score_mean_unweighted"] = sum(scores) / len(scores) if scores else ""
        base["score_eligible_n"] = len(scores)
        base["score_excluded_n"] = len(raw_scores) - len(scores)
        base["score_mean_policy"] = SCORE_MEAN_POLICY
        base["score_mean_raw_reported"] = sum(raw_scores) / len(raw_scores) if raw_scores else ""
        if len(judge_slugs) == 2:
            left, right = (f"{slug}_score_0_100" for slug in judge_slugs)
            left_eligible = base.get(left.replace("_score_0_100", "_score_eligible_for_mean")) is True
            right_eligible = base.get(right.replace("_score_0_100", "_score_eligible_for_mean")) is True
            if left_eligible and right_eligible and isinstance(base.get(left), (int, float)) and isinstance(base.get(right), (int, float)):
                base["score_delta_second_minus_first"] = float(base[right]) - float(base[left])
            else:
                base["score_delta_second_minus_first"] = ""

    wide_fields = ["candidate_id", "candidate_family", "model_alias", "source_id", "item_id"]
    for slug in judge_slugs:
        wide_fields.extend([f"{slug}_score_0_100", f"{slug}_score_eligible_for_mean", f"{slug}_score_shape_status", f"{slug}_exact_quote_status", f"{slug}_critical"])
    wide_fields.extend(["score_mean_unweighted", "score_mean_raw_reported", "score_eligible_n", "score_excluded_n", "score_mean_policy", "score_delta_second_minus_first", "mapping_scope", "comparison_scope"])
    _write_csv(wide_path, list(grouped.values()), wide_fields)

    # The mapping and exact translations are intentionally joined only here,
    # after judge imports.  This is an owner-only review artifact, never a
    # judge-visible packet or prompt.
    comparison_rows: list[dict[str, Any]] = []
    for base in grouped.values():
        item = packet_info["items"].get(str(base["item_id"]), {})
        target = packet_info["candidate_targets"].get((str(base["item_id"]), str(base["candidate_id"])), "")
        row = {
            "source_id": base["source_id"],
            "item_id": base["item_id"],
            "candidate_id": base["candidate_id"],
            "candidate_family": base["candidate_family"],
            "model_alias": base["model_alias"],
            "source_en": item.get("source_en", "") if isinstance(item, Mapping) else "",
            "target_zh_tw": target,
        }
        for slug in judge_slugs:
            for suffix in ("score_0_100", "score_eligible_for_mean", "score_shape_status", "exact_quote_status", "critical"):
                row[f"{slug}_{suffix}"] = base.get(f"{slug}_{suffix}", "")
        row.update({
            "score_mean_unweighted": base.get("score_mean_unweighted", ""),
            "score_mean_raw_reported": base.get("score_mean_raw_reported", ""),
            "score_eligible_n": base.get("score_eligible_n", 0),
            "score_excluded_n": base.get("score_excluded_n", 0),
            "score_mean_policy": base.get("score_mean_policy", SCORE_MEAN_POLICY),
            "mapping_scope": "owner_only_postjudge",
            "comparison_scope": "exact_frozen_translation_with_exploratory_judge_scores",
        })
        comparison_rows.append(row)
    comparison_fields = ["source_id", "item_id", "candidate_id", "candidate_family", "model_alias", "source_en", "target_zh_tw"]
    for slug in judge_slugs:
        comparison_fields.extend([f"{slug}_score_0_100", f"{slug}_score_eligible_for_mean", f"{slug}_score_shape_status", f"{slug}_exact_quote_status", f"{slug}_critical"])
    comparison_fields.extend(["score_mean_unweighted", "score_mean_raw_reported", "score_eligible_n", "score_excluded_n", "score_mean_policy", "mapping_scope", "comparison_scope"])
    _write_csv(translation_comparison_path, comparison_rows, comparison_fields)

    if errors_output is not None:
        _write_csv(errors_output, error_rows, ERROR_ANNOTATION_FIELDS)

    validity_json_path.parent.mkdir(parents=True, exist_ok=True)
    validity_json_path.write_text(json.dumps({"schema_version": "judge-pilot-validity-table-v1", "rows": validity_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(validity_csv_path, validity_rows, list(validity_rows[0].keys()) if validity_rows else [])
    result = {"long_rows": len(long_rows), "wide_rows": len(grouped), "translation_comparison_rows": len(comparison_rows), "validity_rows": len(validity_rows)}
    if errors_output is not None:
        result["error_rows"] = len(error_rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--long-output", required=True)
    parser.add_argument("--wide-output", required=True)
    parser.add_argument("--translation-comparison-output", required=True)
    parser.add_argument("--validity-json-output", required=True)
    parser.add_argument("--validity-csv-output", required=True)
    parser.add_argument("--errors-output", required=False)
    args = parser.parse_args(argv)
    result = build(
        Path(args.packet),
        Path(args.mapping),
        [Path(path) for path in args.input],
        Path(args.long_output),
        Path(args.wide_output),
        Path(args.translation_comparison_output),
        Path(args.validity_json_output),
        Path(args.validity_csv_output),
        errors_output=Path(args.errors_output) if args.errors_output else None,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
