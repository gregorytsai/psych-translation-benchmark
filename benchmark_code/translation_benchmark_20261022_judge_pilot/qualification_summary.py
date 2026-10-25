#!/usr/bin/env python3
"""Score qualification judgments against the private, frozen answer key.

This evaluator is for the study owner, not for a judge prompt.  It never
exports the clean/mutated mapping or expected error labels.  An arbitrary error
object is reported only as a mutation-flag proxy; a separate conservative
evidence check requires the canonical target/source evidence to align with the
programmed mutation.  Detection evidence is reported separately from recall
of a critical severity label: an evaluator can notice the error while choosing
``major`` rather than ``critical``.  The answer key is read-only and is never
edited or inferred from the judge output.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_pilot import _qualification_score_audit, load_bundles, read_json, write_json


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_type(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _type_matches(expected: str, actual_types: Sequence[str]) -> bool:
    expected = _normal_type(expected)
    if not expected or expected == "none":
        return False
    for actual in actual_types:
        actual = _normal_type(actual)
        if actual == expected or actual.startswith(expected + "/") or expected.startswith(actual + "/"):
            return True
    return False


def _normalized_text(value: Any) -> str:
    """Normalize only case and whitespace for source-span containment checks."""

    return " ".join(str(value or "").casefold().split())


def _normalized_contains(haystack: Any, needle: Any) -> bool:
    haystack_normalized = _normalized_text(haystack)
    needle_normalized = _normalized_text(needle)
    return bool(needle_normalized) and needle_normalized in haystack_normalized


def _changed_target_regions(clean_target: str, mutated_target: str) -> list[tuple[int, int]]:
    """Return target intervals affected by the programmed clean/mutated diff.

    A deletion has no characters in the mutated target.  The first two
    remaining characters at the deletion point are retained as an anchor so a
    judge can quote the phrase that carries the changed negation without
    making an unrelated preceding phrase look supported.  This is an evidence
    anchor, not an inferred answer label.
    """

    regions: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(None, clean_target, mutated_target, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if j1 < j2:
            regions.append((j1, j2))
        elif mutated_target:
            regions.append((j1, min(len(mutated_target), j1 + 2)))
    return regions


def _quote_overlaps_regions(quote: str, target: str, regions: Sequence[tuple[int, int]]) -> bool:
    if not quote or not target or not regions:
        return False
    start = target.find(quote)
    while start >= 0:
        end = start + len(quote)
        if any(start < region_end and end > region_start for region_start, region_end in regions):
            return True
        start = target.find(quote, start + 1)
    return False


def _evidence_assessment(
    row: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    errors: Sequence[Mapping[str, Any]],
    top_level_critical: bool,
) -> dict[str, Any]:
    """Check whether an error is tied to the private mutation, conservatively.

    An arbitrary error object is only a mutation flag proxy.  A supported
    mutation requires an exact target quote overlapping the programmed target
    diff, or (for an omission) a source span that is present in the canonical
    source.  Missing packet context is explicitly unavailable rather than a
    pass or a failure.
    """

    unavailable = {
        "evidence_checked": False,
        "evidence_supported": None,
        "evidence_status": "unavailable_context",
        "evidence_details": [],
    }
    if context is None:
        return unavailable

    target = str(context.get("mutated_target") or "")
    source = str(context.get("source_text") or "")
    regions = list(context.get("changed_target_regions") or [])
    if not errors:
        return {
            "evidence_checked": True,
            "evidence_supported": False,
            "evidence_status": "critical_flag_without_error_evidence" if top_level_critical else "no_error_evidence",
            "evidence_details": [],
        }

    details: list[dict[str, Any]] = []
    for error_index, error in enumerate(errors):
        quote = error.get("target_exact_quote")
        source_span = error.get("source_span")
        source_evidence = error.get("source_evidence")
        target_anchor = error.get("target_anchor")
        target_quote_present = isinstance(quote, str) and bool(quote) and quote in target
        target_quote_changed = target_quote_present and _quote_overlaps_regions(quote, target, regions)
        source_span_aligned = isinstance(source_span, str) and _normalized_contains(source, source_span)
        source_evidence_contains_span = (
            source_span_aligned
            and isinstance(source_evidence, str)
            and _normalized_contains(source_evidence, source_span)
        )
        target_anchor_present = isinstance(target_anchor, str) and bool(target_anchor) and target_anchor in target
        if isinstance(quote, str) and quote:
            supported = target_quote_present and target_quote_changed
            reasons: list[str] = []
            if not target_quote_present:
                reasons.append("target_quote_not_in_canonical_mutated_target")
            elif not target_quote_changed:
                reasons.append("target_quote_does_not_overlap_programmed_mutation")
            if supported:
                reasons.append("target_quote_overlaps_programmed_mutation")
        else:
            # An omission has no mutated-side substring to quote.  Requiring a
            # source span prevents a bare critical flag or an unrelated error
            # from being counted as evidence for the keyed mutation.
            supported = source_span_aligned and source_evidence_contains_span and target_anchor_present
            reasons = []
            if not source_span_aligned:
                reasons.append("omission_source_span_missing_or_not_in_canonical_source")
            if not source_evidence_contains_span:
                reasons.append("omission_source_evidence_does_not_contain_aligned_source_span")
            if not target_anchor_present:
                reasons.append("omission_target_anchor_missing_or_not_in_canonical_target")
            if supported:
                reasons.append("omission_source_and_target_anchors_align_canonical_mutation")
        details.append({
            "error_index": error_index,
            "supported": supported,
            "target_quote_present": target_quote_present,
            "target_quote_changed": target_quote_changed,
            "source_span_aligned": source_span_aligned,
            "source_evidence_contains_span": source_evidence_contains_span,
            "target_anchor_present": target_anchor_present,
            "reasons": reasons,
        })

    supported = any(detail["supported"] for detail in details)
    return {
        "evidence_checked": True,
        "evidence_supported": supported,
        "evidence_status": "supported" if supported else "unaligned_error_evidence",
        "evidence_details": details,
    }


def _assessment(row: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if row is None:
        return {"available": False, "status": "absent", "flagged": None, "top_level_critical": None, "error_severity_critical": None, "error_types": [], **_evidence_unavailable()}
    if row.get("_status") != "ok":
        return {"available": False, "status": row.get("_status"), "flagged": None, "top_level_critical": None, "error_severity_critical": None, "error_types": [], **_evidence_unavailable()}
    if row.get("abstain") is True:
        return {"available": False, "status": "abstain", "flagged": None, "top_level_critical": None, "error_severity_critical": None, "error_types": [], **_evidence_unavailable()}
    errors = [error for error in row.get("_errors", []) if isinstance(error, Mapping)]
    top_level_critical = row.get("_critical") is True
    error_severity_critical = any(_normal_type(error.get("severity")) == "critical" for error in errors)
    error_types = sorted({str(error.get("type")) for error in errors if error.get("type")})
    evidence = _evidence_assessment(row, context, errors, top_level_critical)
    return {
        "available": True,
        "status": "ok",
        # This is deliberately a flag proxy, not proven mutation detection:
        # an error object may describe an unrelated style issue.
        "flagged": bool(errors) or top_level_critical,
        "top_level_critical": top_level_critical,
        "error_severity_critical": error_severity_critical,
        "error_types": error_types,
        **evidence,
    }


def _evidence_unavailable() -> dict[str, Any]:
    return {
        "evidence_checked": False,
        "evidence_supported": None,
        "evidence_status": "unavailable_row",
        "evidence_details": [],
    }


def _build_qualification_context(
    packet: Mapping[str, Any] | None,
    qualification_key: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build owner-only evidence anchors from the canonical public packet."""

    if not isinstance(packet, Mapping):
        return {}, {"status": "unavailable", "reason": "canonical packet context was not supplied or found", "n_items": 0, "missing_item_ids": sorted(str(item_id) for item_id in qualification_key)}
    qualification = packet.get("qualification")
    items = qualification.get("items") if isinstance(qualification, Mapping) else None
    if not isinstance(items, list):
        return {}, {"status": "unavailable", "reason": "canonical packet has no qualification.items list", "n_items": 0, "missing_item_ids": sorted(str(item_id) for item_id in qualification_key)}

    contexts: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("qualification_id", item.get("item_id"))
        spec = qualification_key.get(str(item_id)) if item_id is not None else None
        if not isinstance(spec, Mapping):
            continue
        variants = item.get("variants")
        variant_text: dict[str, str] = {}
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, Mapping) and variant.get("blind_id") is not None and isinstance(variant.get("target_zh_tw"), str):
                    variant_text[str(variant["blind_id"])] = variant["target_zh_tw"]
        clean_id = str(spec.get("clean_blind_id"))
        mutated_id = str(spec.get("mutated_blind_id"))
        clean_target = variant_text.get(clean_id)
        mutated_target = variant_text.get(mutated_id)
        source_text = item.get("source_en")
        if not isinstance(source_text, str) or clean_target is None or mutated_target is None:
            continue
        contexts[str(item_id)] = {
            "source_text": source_text,
            "clean_target": clean_target,
            "mutated_target": mutated_target,
            "changed_target_regions": _changed_target_regions(clean_target, mutated_target),
        }

    missing = sorted(str(item_id) for item_id in qualification_key if str(item_id) not in contexts)
    status = "available" if not missing else ("partial" if contexts else "unavailable")
    return contexts, {"status": status, "n_items": len(contexts), "missing_item_ids": missing}


def _collapse_qualification(rows: Sequence[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("_item_id")), str(row.get("_candidate_id")), str(row.get("_judge_id")))].append(row)
    collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
    repeat_conflicts: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        usable = [row for row in group if row.get("_status") == "ok" and row.get("abstain") is not True]
        candidates = usable or [row for row in group if row.get("_status") == "ok"]
        chosen = candidates[-1] if candidates else group[-1]
        signatures = {
            (
                _assessment(row).get("flagged"),
                _assessment(row).get("top_level_critical"),
                _assessment(row).get("error_severity_critical"),
                tuple(_assessment(row).get("error_types", [])),
            )
            for row in candidates
        }
        if len(signatures) > 1:
            repeat_conflicts.append({"item_id": key[0], "blind_id": key[1], "judge_id": key[2], "n_runs": len(candidates), "note": "latest valid run used for the descriptive qualification summary; repeat disagreement is retained"})
        collapsed[key] = chosen
    return collapsed, {"logical_units": len(grouped), "collapsed_units": len(collapsed), "n_repeat_runs": sum(max(0, len(group) - 1) for group in grouped.values()), "repeat_conflicts": repeat_conflicts}


def _metric(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
    return {"estimate": _ratio(numerator, denominator), "numerator": numerator, "denominator": denominator, "definition": definition}


def _qualification_scalar_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-variant scalar-score validity without changing rows."""

    counts: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    for row in rows:
        audit = row.get("_qualification_score_audit")
        if not isinstance(audit, Mapping):
            audit = _qualification_score_audit(row)
        status = str(audit.get("status", "missing"))
        counts[status] += 1
        if status == "invalid" and len(invalid_examples) < 20:
            invalid_examples.append({
                "item_id": row.get("_item_id"),
                "candidate_id": row.get("_candidate_id"),
                "judge_id": row.get("_judge_id"),
                "score_0_100": audit.get("score_0_100"),
                "subscores": audit.get("subscores"),
                "subscores_sum": audit.get("subscores_sum"),
                "reason": audit.get("reason"),
            })
    return {
        "n_variants": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "valid": counts.get("valid", 0),
        "invalid": counts.get("invalid", 0),
        "missing": counts.get("missing", 0),
        "abstain": counts.get("abstain", 0),
        "invalid_examples": invalid_examples,
        "scalar_aggregation": {
            "status": "partial_validity" if counts.get("invalid", 0) or counts.get("missing", 0) else "ok",
            "n_excluded": counts.get("invalid", 0) + counts.get("missing", 0) + counts.get("abstain", 0),
            "note": "Invalid/missing qualification scalar scores are excluded; raw score, subscores, MQM/errors, evidence, and critical fields are not rewritten and are evaluated separately.",
        },
    }


def _judge_summary(
    judge_id: str,
    collapsed: Mapping[tuple[str, str, str], dict[str, Any]],
    answer_key: Mapping[str, Any],
    qualification_context: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    qualification_key = answer_key["qualification"]
    mutation_flag_numerator = mutation_flag_denominator = 0
    mutation_evidence_numerator = mutation_evidence_denominator = 0
    mutation_evidence_context_count = 0
    type_numerator = type_denominator = 0
    clean_false_positive = clean_denominator = 0
    critical_union_numerator = critical_top_numerator = critical_error_numerator = critical_denominator = 0
    critical_disagreement = 0
    invariance_numerator = invariance_denominator = 0
    available_variants = 0
    complete_items = 0
    detect_error_items = 0
    detect_error_mutated_available = 0
    status_counts: Counter[str] = Counter()
    item_results: list[dict[str, Any]] = []

    def get(item_id: str, blind_id: str) -> dict[str, Any]:
        row = collapsed.get((item_id, blind_id, judge_id))
        context = (qualification_context or {}).get(item_id)
        assessment = _assessment(row, context)
        status_counts[str(assessment["status"])] += 1
        return assessment

    for item_id, spec in sorted(qualification_key.items()):
        clean_id = str(spec["clean_blind_id"])
        mutated_id = str(spec["mutated_blind_id"])
        clean = get(item_id, clean_id)
        mutated = get(item_id, mutated_id)
        expected_behavior = str(spec.get("expected_behavior", "detect_error"))
        expected_type = str(spec.get("expected_error_type", "none"))
        expected_severity = _normal_type(spec.get("expected_severity", "none"))
        result = {"item_id": item_id, "clean_variant": clean_id, "mutated_variant": mutated_id, "clean_status": clean["status"], "mutated_status": mutated["status"]}
        available_variants += int(clean["available"]) + int(mutated["available"])
        complete_items += int(clean["available"] and mutated["available"])
        if expected_behavior == "detect_error":
            detect_error_items += 1
            if mutated["available"]:
                detect_error_mutated_available += 1
                mutation_flag_denominator += 1
                mutation_flag_numerator += int(bool(mutated["flagged"]))
                result["mutation_flagged"] = bool(mutated["flagged"])
                result["mutation_evidence_supported"] = mutated["evidence_supported"]
                result["mutation_evidence_status"] = mutated["evidence_status"]
                result["mutation_evidence_details"] = mutated["evidence_details"]
                if mutated["evidence_checked"]:
                    mutation_evidence_denominator += 1
                    mutation_evidence_context_count += 1
                    mutation_evidence_numerator += int(bool(mutated["evidence_supported"]))
                if expected_type != "none":
                    type_denominator += 1
                    type_numerator += int(_type_matches(expected_type, mutated["error_types"]))
                    result["error_type_matched"] = _type_matches(expected_type, mutated["error_types"])
            else:
                result["mutation_flagged"] = None
                result["mutation_evidence_supported"] = None
                result["mutation_evidence_status"] = "unavailable_row"
                result["mutation_evidence_details"] = []
                result["error_type_matched"] = None
            if clean["available"]:
                clean_denominator += 1
                clean_false_positive += int(bool(clean["flagged"]))
                result["clean_false_positive"] = bool(clean["flagged"])
            else:
                result["clean_false_positive"] = None
            if expected_severity == "critical" and mutated["available"]:
                critical_denominator += 1
                top = bool(mutated["top_level_critical"])
                error_critical = bool(mutated["error_severity_critical"])
                critical_top_numerator += int(top)
                critical_error_numerator += int(error_critical)
                critical_union_numerator += int(top or error_critical)
                critical_disagreement += int(top != error_critical)
                result.update({"top_level_critical": top, "error_severity_critical": error_critical, "critical_severity_recalled": top or error_critical})
        elif expected_behavior == "invariance":
            if clean["available"] and mutated["available"]:
                invariance_denominator += 1
                invariant = not bool(clean["flagged"]) and not bool(mutated["flagged"])
                invariance_numerator += int(invariant)
                result["invariance_correct"] = invariant
            else:
                result["invariance_correct"] = None
        item_results.append(result)

    expected_variants = len(qualification_key) * 2
    missing_variants = max(0, expected_variants - available_variants)
    missing_items = max(0, len(qualification_key) - complete_items)
    return {
        "mutation_flag_rate_proxy": _metric(
            mutation_flag_numerator,
            mutation_flag_denominator,
            "Among valid mutated detect_error variants, any explicit error object or top-level critical flag. This is a flag-rate proxy, not proven mutation/fact sensitivity, because an error may be unrelated to the programmed mutation.",
        ),
        "mutation_evidence_supported_rate": _metric(
            mutation_evidence_numerator,
            mutation_evidence_denominator,
            "Among valid mutated detect_error variants with canonical packet context, at least one target quote overlaps the programmed target diff or an omission source_span aligns the canonical source. This conservative evidence check is not run when packet context is unavailable.",
        ),
        "error_type_match": _metric(type_numerator, type_denominator, "Among valid mutated detect_error variants with a keyed type, an output error type equal to or hierarchically matching the keyed type."),
        "clean_false_positive_rate": _metric(clean_false_positive, clean_denominator, "Among valid clean detect_error variants, any explicit error object or top-level critical flag."),
        "critical_severity_recall": {
            "union_top_level_or_error_severity": _metric(critical_union_numerator, critical_denominator, "Among valid keyed critical mutations, top-level critical=true or at least one error with severity=critical."),
            "top_level_critical_flag": _metric(critical_top_numerator, critical_denominator, "Among valid keyed critical mutations, top-level critical=true."),
            "error_object_critical_severity": _metric(critical_error_numerator, critical_denominator, "Among valid keyed critical mutations, at least one error object has severity=critical."),
            "label_disagreement_count": critical_disagreement,
        },
        "invariance_accuracy": _metric(invariance_numerator, invariance_denominator, "Among valid keyed invariance items, neither anonymous variant is spuriously flagged."),
        "coverage": {
            "status": "complete" if missing_variants == 0 else "incomplete",
            "expected_variants": expected_variants,
            "available_variants": available_variants,
            "missing_variants": missing_variants,
            "variant_coverage": _metric(available_variants, expected_variants, "Available valid clean/mutated qualification variants divided by all keyed variants."),
            "expected_items": len(qualification_key),
            "complete_items": complete_items,
            "missing_items": missing_items,
            "item_pair_coverage": _metric(complete_items, len(qualification_key), "Items with both clean and mutated variants available divided by all keyed qualification items."),
            "detect_error_items": detect_error_items,
            "detect_error_mutated_available": detect_error_mutated_available,
            "detect_error_mutated_coverage": _metric(detect_error_mutated_available, detect_error_items, "Valid mutated variants available divided by keyed detect_error items."),
            "evidence_context_available": mutation_evidence_context_count,
            "evidence_check_coverage": _metric(mutation_evidence_context_count, detect_error_mutated_available, "Canonical-packet evidence checks available divided by valid mutated detect_error variants."),
        },
        "missingness": {
            "status_counts": dict(sorted(status_counts.items())),
            "n_keyed_items": len(qualification_key),
            "n_item_results": len(item_results),
            "n_missing_variants": missing_variants,
            "n_incomplete_items": missing_items,
            "note": "Missing, failed, abstained, or unavailable-context observations are not treated as agreement, zero scores, or detected mutations.",
        },
        "item_results": item_results,
    }


def _macro(per_judge: Mapping[str, Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for report in per_judge.values():
        value: Any = report
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return statistics.fmean(values) if values else None


def evaluate(
    inputs: Sequence[str | Path],
    answer_key_path: str | Path,
    packet_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata, rows = load_bundles(inputs)
    key = read_json(answer_key_path)
    if not isinstance(key, Mapping) or not isinstance(key.get("qualification"), Mapping):
        raise ValueError("private answer key must contain a qualification mapping")
    if metadata.get("packet_id") and key.get("packet_id") and metadata["packet_id"] != key["packet_id"]:
        raise ValueError(f"packet_id mismatch between imported bundle and private answer key: {metadata['packet_id']!r} vs {key['packet_id']!r}")
    qualification_rows = [row for row in rows if row.get("_track") == "qualification"]
    collapsed, collapse_audit = _collapse_qualification(qualification_rows)
    judges = sorted({str(row.get("_judge_id")) for row in qualification_rows if row.get("_judge_id") is not None})
    resolved_packet_path = Path(packet_path) if packet_path is not None else Path(answer_key_path).resolve().parent.parent / "judge_packet.json"
    packet: Mapping[str, Any] | None = None
    packet_context_audit: dict[str, Any]
    if resolved_packet_path.exists():
        packet_value = read_json(resolved_packet_path)
        if not isinstance(packet_value, Mapping):
            raise ValueError(f"canonical packet must be a JSON object: {resolved_packet_path}")
        packet = packet_value
        if packet.get("packet_id") and key.get("packet_id") and packet["packet_id"] != key["packet_id"]:
            raise ValueError(f"packet_id mismatch between canonical packet and private answer key: {packet['packet_id']!r} vs {key['packet_id']!r}")
    qualification_context, packet_context_audit = _build_qualification_context(packet, key["qualification"])
    packet_context_audit["path"] = str(resolved_packet_path) if packet is not None else None
    per_judge = {judge: _judge_summary(judge, collapsed, key, qualification_context) for judge in judges}
    for judge in judges:
        per_judge[judge]["qualification_score_audit"] = _qualification_scalar_audit([row for row in qualification_rows if str(row.get("_judge_id")) == judge])
    qualification_score_audit = _qualification_scalar_audit(qualification_rows)
    return {
        "schema_version": "judge-pilot-qualification-summary-v2",
        "status": "PROVISIONAL",
        "packet_id": metadata.get("packet_id"),
        "packet_sha256": metadata.get("packet_sha256"),
        "answer_key": {"source": str(answer_key_path), "sha256": _sha256_file(answer_key_path), "n_keyed_items": len(key["qualification"])},
        "judge_scope": {"judge_ids": judges, "n_distinct_judges": len(judges), "formal_panel": len(judges) >= 5, "note": "Qualification metrics are per-judge descriptive summaries; repeated runs are collapsed and are not new raters."},
        "evidence_context": packet_context_audit,
        "collapse_audit": collapse_audit,
        "qualification_score_audit": qualification_score_audit,
        "per_judge": per_judge,
        "panel_descriptive_macro_mean": {
            "mutation_flag_rate_proxy": _macro(per_judge, ("mutation_flag_rate_proxy", "estimate")),
            "mutation_evidence_supported_rate": _macro(per_judge, ("mutation_evidence_supported_rate", "estimate")),
            "error_type_match": _macro(per_judge, ("error_type_match", "estimate")),
            "clean_false_positive_rate": _macro(per_judge, ("clean_false_positive_rate", "estimate")),
            "critical_severity_recall_union": _macro(per_judge, ("critical_severity_recall", "union_top_level_or_error_severity", "estimate")),
            "critical_severity_recall_top_level_flag": _macro(per_judge, ("critical_severity_recall", "top_level_critical_flag", "estimate")),
            "critical_severity_recall_error_label": _macro(per_judge, ("critical_severity_recall", "error_object_critical_severity", "estimate")),
            "invariance_accuracy": _macro(per_judge, ("invariance_accuracy", "estimate")),
            "note": "Macro means are descriptive and do not provide judge-independent standard errors; the mutation-flag proxy and conservative evidence-supported rate are intentionally not substituted for critical-severity recall.",
        },
        "limitations": [
            "The private key is read-only; no expected label or severity is changed after seeing judge output.",
            "An explicit error object or top-level critical flag is only a mutation-flag proxy until target/source evidence is checked against the programmed mutation.",
            "The evidence-supported mutation rate is unavailable when canonical packet context is missing; missing/failed/abstained variants are excluded from valid denominators and shown in coverage.",
            "Critical-severity recall separately asks whether the keyed critical severity was labeled; it is not inferred from the mutation flag proxy.",
            "Error-type matching is reported separately from fact detection because locale/synonym and severity disagreements need audit rather than post-hoc relabeling.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="validated judge-result bundle; repeat for independently imported judges")
    parser.add_argument("--answer-key", required=True, help="private answer_key.private.json; never send this file to a judge")
    parser.add_argument("--packet", help="canonical judge_packet.json for conservative mutation-evidence checks (defaults to the pilot packet next to the private directory)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.input, args.answer_key, args.packet)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
