#!/usr/bin/env python3
"""Small, offline-first implementation of the multi-LLM-judge protocol.

The module deliberately has no network or model-runtime dependency.  It can
materialize the checked-in fixture, validate JSONL records, qualify a panel,
and aggregate pointwise/pairwise observations.  The statistical routines are
transparent smoke-test implementations; a production run should add
source-clustered bootstrap confidence intervals and human calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
SCHEMA_DIR = HERE / "schemas"
FIXTURE_DIR = HERE / "fixtures"
SEVERITY_ORDER = {"none": 0, "minor": 1, "major": 2, "critical": 3}


class ValidationError(ValueError):
    """A compact JSON-Schema validation error for the supported subset."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: expected a JSON object")
            value["__line_number__"] = line_number
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "__line_number__"}
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$", errors: list[str] | None = None) -> list[str]:
    """Validate the small Draft-2020-12 subset used by the checked-in schemas.

    This avoids making `jsonschema` a mandatory dependency.  If a caller wants
    full Draft-2020-12 validation, it can run the same records through its
    preferred validator after this lightweight gate.
    """

    errors = [] if errors is None else errors
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
        return errors
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string shorter than minLength")
        if "pattern" in schema:
            import re

            if re.search(schema["pattern"], value) is None:
                errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", -math.inf):
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if value > schema.get("maximum", math.inf):
            errors.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than minItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]", errors)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            for key in unknown:
                errors.append(f"{path}: unexpected property {key!r}")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}", errors)
    return errors


def validate_rows(rows: Sequence[dict[str, Any]], schema_path: Path, source_name: str) -> list[str]:
    schema = read_json(schema_path)
    errors: list[str] = []
    for row in rows:
        line = row.get("__line_number__", "?")
        clean = {key: value for key, value in row.items() if key != "__line_number__"}
        row_errors = _validate_schema(clean, schema, "$" + f" (line {line})")
        errors.extend(f"{source_name}: {message}" for message in row_errors)
    return errors


def validate_or_raise(rows: Sequence[dict[str, Any]], schema_name: str, source_name: str) -> None:
    errors = validate_rows(rows, SCHEMA_DIR / schema_name, source_name)
    if errors:
        preview = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... {len(errors) - 20} more errors"
        raise ValidationError(preview + suffix)


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _iqr(values: Sequence[float]) -> float | None:
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    return None if q1 is None or q3 is None else q3 - q1


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _metric_unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "estimate": None, "reason": reason}


def icc_two_way_random_absolute(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return ICC(2,1) and ICC(2,k), absolute agreement.

    Rows are independently rated targets and columns are judges.  The formula
    is the two-way random-effects absolute-agreement form commonly called
    ICC(A,1)/ICC(A,k), with complete rows only.
    """

    n = len(matrix)
    k = len(matrix[0]) if n else 0
    if n < 2 or k < 2:
        return {"status": "unavailable", "icc_2_1": None, "icc_2_k": None, "reason": "need at least 2 targets and 2 judges"}
    if any(len(row) != k for row in matrix):
        return {"status": "unavailable", "icc_2_1": None, "icc_2_k": None, "reason": "incomplete or ragged score matrix"}
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for row in matrix for value in row):
        return {"status": "unavailable", "icc_2_1": None, "icc_2_k": None, "reason": "non-finite score in matrix"}
    grand_mean = statistics.fmean(value for row in matrix for value in row)
    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean(matrix[i][j] for i in range(n)) for j in range(k)]
    ms_rows = k * sum((mean - grand_mean) ** 2 for mean in row_means) / (n - 1)
    ms_cols = n * sum((mean - grand_mean) ** 2 for mean in col_means) / (k - 1)
    residual = sum(
        (matrix[i][j] - row_means[i] - col_means[j] + grand_mean) ** 2
        for i in range(n)
        for j in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator_single = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    denominator_average = ms_rows + (ms_cols - ms_error) / n
    if denominator_single == 0 or denominator_average == 0:
        return {"status": "unavailable", "icc_2_1": None, "icc_2_k": None, "reason": "zero ICC denominator"}
    return {
        "status": "ok",
        "icc_2_1": (ms_rows - ms_error) / denominator_single,
        "icc_2_k": (ms_rows - ms_error) / denominator_average,
        "n_targets": n,
        "n_judges": k,
        "model": "two-way random, absolute agreement",
        "unit": "single and arithmetic mean of judges",
    }


def _distance(a: Any, b: Any, kind: str, levels: Sequence[Any] | None = None) -> float:
    if kind == "nominal":
        return 0.0 if a == b else 1.0
    order = list(levels or [])
    if a not in order or b not in order:
        return 1.0
    denominator = max(1, len(order) - 1)
    return abs(order.index(a) - order.index(b)) / denominator


def _units_and_labels(units: Sequence[Sequence[Any]]) -> tuple[list[list[Any]], list[Any]]:
    clean_units = [[label for label in unit if label is not None] for unit in units]
    clean_units = [unit for unit in clean_units if len(unit) >= 2]
    labels = []
    seen: set[Any] = set()
    for unit in clean_units:
        for label in unit:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return clean_units, labels


def krippendorff_alpha(units: Sequence[Sequence[Any]], kind: str = "nominal", levels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Pairwise coincidence implementation of Krippendorff's alpha.

    It supports missing labels and nominal/ordinal distance.  The reported
    ordinal distance is normalized absolute distance, which is explicit here
    so a production analysis can replace it with its preregistered delta.
    """

    clean_units, labels = _units_and_labels(units)
    if not clean_units or len(labels) < 2:
        return _metric_unavailable("need at least two labels and one unit with two annotations")
    observed_numerator = 0.0
    observed_denominator = 0.0
    pooled = [label for unit in clean_units for label in unit]
    for unit in clean_units:
        for first, second in itertools.permutations(unit, 2):
            observed_numerator += _distance(first, second, kind, levels)
            observed_denominator += 1.0
    expected_numerator = 0.0
    expected_denominator = 0.0
    for first, second in itertools.permutations(pooled, 2):
        expected_numerator += _distance(first, second, kind, levels)
        expected_denominator += 1.0
    observed = _ratio(observed_numerator, observed_denominator)
    expected = _ratio(expected_numerator, expected_denominator)
    if observed is None or expected is None or expected == 0:
        return _metric_unavailable("zero expected disagreement")
    return {
        "status": "ok",
        "estimate": 1.0 - observed / expected,
        "kind": kind,
        "distance": "nominal equality" if kind == "nominal" else "normalized absolute ordinal distance",
        "n_units": len(clean_units),
        "n_labels": len(pooled),
    }


def gwet_ac(units: Sequence[Sequence[Any]], kind: str = "nominal", levels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Lightweight Gwet AC1/AC2 estimate with explicit agreement terms.

    AC1 uses the multicategory Gwet chance-agreement correction.  AC2 uses
    weighted marginal chance agreement for the supplied ordinal levels.  This
    is suitable for an offline diagnostic; final reporting should cross-check
    against a pinned implementation such as irrCAC.
    """

    clean_units, labels = _units_and_labels(units)
    if not clean_units or not labels:
        return _metric_unavailable("no complete unit with two annotations")
    pairs = [(unit[i], unit[j]) for unit in clean_units for i in range(len(unit)) for j in range(i + 1, len(unit))]
    if not pairs:
        return _metric_unavailable("no annotation pairs")
    if kind == "nominal":
        observed = statistics.fmean(1.0 if first == second else 0.0 for first, second in pairs)
        counts = Counter(label for unit in clean_units for label in unit)
        total = sum(counts.values())
        probabilities = [count / total for count in counts.values()]
        category_count = len(probabilities)
        expected = sum(probability * (1.0 - probability) for probability in probabilities)
        if category_count > 1:
            expected /= category_count - 1
        metric_name = "AC1"
    else:
        order = list(levels or labels)
        observed = statistics.fmean(1.0 - _distance(first, second, "ordinal", order) for first, second in pairs)
        counts = Counter(label for unit in clean_units for label in unit)
        total = sum(counts.values())
        expected = 0.0
        for first in counts:
            for second in counts:
                expected += (counts[first] / total) * (counts[second] / total) * (1.0 - _distance(first, second, "ordinal", order))
        metric_name = "AC2"
    if expected == 1.0:
        return _metric_unavailable("zero chance-disagreement denominator")
    return {
        "status": "ok",
        "estimate": (observed - expected) / (1.0 - expected),
        "metric": metric_name,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "n_units": len(clean_units),
        "n_pairs": len(pairs),
    }


def span_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_start, first_end = int(first["target_start"]), int(first["target_end"])
    second_start, second_end = int(second["target_start"]), int(second["target_end"])
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return 0.0 if union <= 0 else intersection / union


def _greedy_span_match(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]], threshold: float = 0.5) -> tuple[int, float]:
    candidates = sorted(
        ((span_iou(left, right), left_index, right_index) for left_index, left in enumerate(first) for right_index, right in enumerate(second)),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = 0
    iou_sum = 0.0
    for iou, left_index, right_index in candidates:
        if iou < threshold or left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches += 1
        iou_sum += iou
    return matches, iou_sum


def span_pair_agreement(records: Sequence[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["item"]["item_id"], record["candidate"]["blind_id"])
        judge = record["judge"]["judge_id"]
        errors = record["pointwise"]["errors"]
        grouped[key][judge].extend(errors)
    total_matches = 0
    total_predicted = 0
    total_reference = 0
    iou_values: list[float] = []
    pair_count = 0
    for judges in grouped.values():
        for left_judge, right_judge in itertools.combinations(sorted(judges), 2):
            left, right = judges[left_judge], judges[right_judge]
            matches, iou_sum = _greedy_span_match(left, right, threshold)
            total_matches += matches
            total_predicted += len(left)
            total_reference += len(right)
            if matches:
                iou_values.extend([iou_sum / matches] * matches)
            pair_count += 1
    precision = _ratio(total_matches, total_predicted)
    recall = _ratio(total_matches, total_reference)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold_iou": threshold,
        "pair_count": pair_count,
        "matched_spans": total_matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_of_matches": _mean(iou_values),
        "status": "ok" if pair_count else "unavailable",
        "reason": None if pair_count else "no candidate with two judges",
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _subscores(score: float) -> dict[str, float]:
    first = round(score * 0.30, 2)
    second = round(score * 0.30, 2)
    third = round(score * 0.20, 2)
    fourth = round(score - first - second - third, 2)
    return {
        "terminology_locale": first,
        "semantic_safety": second,
        "fluency": third,
        "completeness_format": fourth,
    }


def _candidate_flags(item: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    errors = candidate.get("errors", [])
    error_types = {error["type"] for error in errors}
    return {
        "language_ok": True,
        "traditional_ok": candidate.get("traditional_ok", "simplified" not in candidate["translation"]),
        "numbers_ok": candidate.get("numbers_ok", "number_dose" not in " ".join(error_types)),
        "drug_terms_ok": True,
        "format_ok": True,
        "critical_deterministic": bool(candidate.get("deterministic_critical", False)),
    }


def materialize_fixture(config: dict[str, Any], anchors: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic fixture observations without invoking a judge."""

    run_id = config["run_id"]
    prompt_sha = _sha256_text(config["prompt_sha256_input"])
    judges = config["judges"]
    qualifications: list[dict[str, Any]] = []
    for anchor in anchors:
        for judge in judges:
            offset = config["qualification_score_offsets"].get(judge["judge_id"], 0)
            if anchor["expected_behavior"] == "detect_error":
                clean_score = 96 + offset
                mutated_score = 34 + offset if anchor["expected_severity"] == "critical" else 52 + offset
                qualifications.append(
                    {
                        "run_id": run_id,
                        "anchor_id": anchor["anchor_id"],
                        "judge_id": judge["judge_id"],
                        "judge_family": judge["family"],
                        "clean_score_0_100": clean_score,
                        "mutated_score_0_100": mutated_score,
                        "clean_critical": False,
                        "mutated_critical": anchor["expected_severity"] == "critical",
                        "error_detected": True,
                        "mutated_error_type": anchor["expected_error_type"],
                        "mutated_severity": anchor["expected_severity"],
                        "position_consistent": True,
                    }
                )
            else:
                qualifications.append(
                    {
                        "run_id": run_id,
                        "anchor_id": anchor["anchor_id"],
                        "judge_id": judge["judge_id"],
                        "judge_family": judge["family"],
                        "clean_score_0_100": 93 + offset,
                        "mutated_score_0_100": 92 + offset,
                        "clean_critical": False,
                        "mutated_critical": False,
                        "error_detected": False,
                        "mutated_error_type": "none",
                        "mutated_severity": "none",
                        "position_consistent": True,
                    }
                )

    pointwise: list[dict[str, Any]] = []
    for item in config["items"]:
        for candidate in item["candidates"]:
            for judge in judges:
                score = max(0, min(100, candidate["score"] + judge.get("score_offset", 0)))
                errors = candidate.get("errors", [])
                pointwise.append(
                    {
                        "run": {
                            "run_id": run_id,
                            "prompt_sha256": prompt_sha,
                            "judge_model_sha256": _sha256_text("offline-fixture:" + judge["judge_id"]),
                            "temperature": 0,
                            "seed": 0,
                            "blind_seed": config["blind_seed"],
                        },
                        "item": {
                            "item_id": item["item_id"],
                            "domain": item["domain"],
                            "input_type": item["input_type"],
                            "source_en": item["source_en"],
                            "target_locale": "zh-TW",
                        },
                        "candidate": {
                            "blind_id": candidate["candidate_id"],
                            "raw_translation_zh_tw": candidate["translation"],
                            "source_raw": item["source_en"],
                        },
                        "judge": {
                            "judge_id": judge["judge_id"],
                            "family": judge["family"],
                            "qualification_version": "offline-fixture-v1",
                        },
                        "deterministic": _candidate_flags(item, candidate),
                        "pointwise": {
                            "score_0_100": score,
                            "subscores": _subscores(score),
                            "critical": bool(candidate.get("critical", False)),
                            "errors": errors,
                        },
                        "timings": {
                            "api_elapsed_s": 0.0,
                            "prompt_tok_s": 0.0,
                            "generation_tok_s": 0.0,
                            "cold": False,
                            "timeout": False,
                        },
                    }
                )

    pairwise: list[dict[str, Any]] = []
    sensitive = set(config.get("position_sensitive", []))
    for item in config["items"]:
        candidates = item["candidates"]
        for left, right in itertools.combinations(candidates, 2):
            pair_id = f"{item['item_id']}::{left['candidate_id']}__{right['candidate_id']}"
            if left["score"] == right["score"]:
                canonical_winner = "TIE"
            else:
                canonical_winner = left["candidate_id"] if left["score"] > right["score"] else right["candidate_id"]
            for judge in judges:
                for orientation in ("AB", "BA"):
                    candidate_a, candidate_b = (
                        (left["candidate_id"], right["candidate_id"])
                        if orientation == "AB"
                        else (right["candidate_id"], left["candidate_id"])
                    )
                    sensitive_key = f"{pair_id}::{judge['judge_id']}"
                    if sensitive_key in sensitive:
                        # A deliberately position-sensitive fixture vote: the
                        # judge chooses the first displayed option in both
                        # orientations, so the canonical winner changes.
                        winner = "A"
                    elif canonical_winner == "TIE":
                        winner = "TIE"
                    else:
                        winner = "A" if canonical_winner == candidate_a else "B"
                    pairwise.append(
                        {
                            "run_id": run_id,
                            "item_id": item["item_id"],
                            "pair_id": pair_id,
                            "judge_id": judge["judge_id"],
                            "orientation": orientation,
                            "candidate_a": candidate_a,
                            "candidate_b": candidate_b,
                            "winner": winner,
                            "confidence": 0.55 if winner == "TIE" else 0.92,
                            "critical_override": False,
                            "reason_codes": ["fixture_quality_order"],
                        }
                    )
    return qualifications, pointwise, pairwise


def qualification_summary(anchors: Sequence[dict[str, Any]], observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchor_map = {anchor["anchor_id"]: anchor for anchor in anchors}
    by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row["anchor_id"] not in anchor_map:
            raise ValidationError(f"qualification observation references unknown anchor {row['anchor_id']!r}")
        by_judge[row["judge_id"]].append(row)
    results: dict[str, Any] = {}
    for judge_id, rows in sorted(by_judge.items()):
        critical_expected = [row for row in rows if anchor_map[row["anchor_id"]]["expected_severity"] == "critical"]
        critical_detected = [row for row in critical_expected if row["mutated_critical"]]
        sentinel_rows = [row for row in rows if anchor_map[row["anchor_id"]]["sentinel"]]
        sentinel_misses = [row["anchor_id"] for row in sentinel_rows if not row["error_detected"] or (anchor_map[row["anchor_id"]]["expected_severity"] == "critical" and not row["mutated_critical"])]
        clean_false_positives = sum(1 for row in rows if row["clean_critical"])
        detect_rows = [row for row in rows if anchor_map[row["anchor_id"]]["expected_behavior"] == "detect_error"]
        monotonic_rows = [row for row in detect_rows if row["clean_score_0_100"] > row["mutated_score_0_100"]]
        invariant_rows = [row for row in rows if anchor_map[row["anchor_id"]]["expected_behavior"] == "invariance"]
        invariant_ok = [row for row in invariant_rows if abs(row["clean_score_0_100"] - row["mutated_score_0_100"]) <= 5 and not row["mutated_critical"]]
        classes = sorted({anchor_map[row["anchor_id"]]["expected_error_type"] for row in detect_rows})
        class_f1: dict[str, float] = {}
        for label in classes:
            true_positive = sum(1 for row in detect_rows if anchor_map[row["anchor_id"]]["expected_error_type"] == label and row["mutated_error_type"] == label)
            false_positive = sum(1 for row in detect_rows if anchor_map[row["anchor_id"]]["expected_error_type"] != label and row["mutated_error_type"] == label)
            false_negative = sum(1 for row in detect_rows if anchor_map[row["anchor_id"]]["expected_error_type"] == label and row["mutated_error_type"] != label)
            precision = _ratio(true_positive, true_positive + false_positive)
            recall = _ratio(true_positive, true_positive + false_negative)
            class_f1[label] = 0.0 if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
        critical_recall = _ratio(len(critical_detected), len(critical_expected))
        critical_false_positive = _ratio(clean_false_positives, len(rows))
        type_macro_f1 = _mean(list(class_f1.values()))
        monotonicity = _ratio(len(monotonic_rows), len(detect_rows))
        invariance = _ratio(len(invariant_ok), len(invariant_rows))
        passed = (
            critical_recall is not None
            and critical_recall >= 0.90
            and not sentinel_misses
            and (type_macro_f1 is not None and type_macro_f1 >= 0.85)
            and (critical_false_positive is not None and critical_false_positive <= 0.10)
            and (monotonicity is not None and monotonicity >= 0.90)
            and (invariance is None or invariance >= 0.90)
            and all(row["position_consistent"] for row in rows)
        )
        results[judge_id] = {
            "n_observations": len(rows),
            "critical_recall": critical_recall,
            "critical_expected": len(critical_expected),
            "critical_detected": len(critical_detected),
            "critical_false_positive_rate": critical_false_positive,
            "sentinel_misses": sentinel_misses,
            "type_macro_f1": type_macro_f1,
            "type_f1": class_f1,
            "monotonicity": monotonicity,
            "invariance_within_5_points": invariance,
            "position_consistent": all(row["position_consistent"] for row in rows),
            "status": "PASS" if passed else "FAIL",
        }
    panel_pass = bool(results) and all(value["status"] == "PASS" for value in results.values())
    return {
        "gates": {
            "critical_recall_min": 0.90,
            "type_macro_f1_min": 0.85,
            "critical_false_positive_rate_max": 0.10,
            "monotonicity_min": 0.90,
            "invariance_difference_max": 5,
        },
        "judges": results,
        "panel_status": "PASS" if panel_pass else "FAIL",
        "note": "Known-error anchors are behavior-contract checks, not human clinical gold.",
    }


def _pair_vote_to_candidate(row: dict[str, Any]) -> str:
    if row["winner"] == "TIE":
        return "TIE"
    return row["candidate_a"] if row["winner"] == "A" else row["candidate_b"]


def swap_consistency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        groups[(row["item_id"], row["pair_id"], row["judge_id"])][row["orientation"]] = row
    complete = 0
    consistent = 0
    stable_ties = 0
    position_sensitive = 0
    tie_disagreement = 0
    incomplete = 0
    for orientations in groups.values():
        if "AB" not in orientations or "BA" not in orientations:
            incomplete += 1
            continue
        complete += 1
        first = _pair_vote_to_candidate(orientations["AB"])
        second = _pair_vote_to_candidate(orientations["BA"])
        if first == second:
            consistent += 1
            if first == "TIE":
                stable_ties += 1
        elif first != "TIE" and second != "TIE":
            position_sensitive += 1
        else:
            tie_disagreement += 1
    return {
        "complete_pairs": complete,
        "consistent_pairs": consistent,
        "swap_consistency": _ratio(consistent, complete),
        "stable_tie_pairs": stable_ties,
        "position_sensitive_pairs": position_sensitive,
        "tie_disagreement_pairs": tie_disagreement,
        "incomplete_pairs": incomplete,
        "status": "ok" if complete else "unavailable",
    }


def pairwise_copeland(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    candidates: set[str] = set()
    for row in records:
        candidates.add(row["candidate_a"])
        candidates.add(row["candidate_b"])
        grouped[(row["item_id"], row["pair_id"], row["judge_id"])][row["orientation"]] = row
    pair_votes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (item_id, pair_id, _judge_id), orientations in grouped.items():
        if "AB" not in orientations or "BA" not in orientations:
            continue
        first = _pair_vote_to_candidate(orientations["AB"])
        second = _pair_vote_to_candidate(orientations["BA"])
        if first == second:
            pair_votes[(item_id, pair_id, "consistent")].append(first)
    outcomes: list[dict[str, Any]] = []
    for (item_id, pair_id, _), votes in sorted(pair_votes.items()):
        if not votes:
            continue
        pair_candidates = set()
        for row in records:
            if row["item_id"] == item_id and row["pair_id"] == pair_id:
                pair_candidates.update((row["candidate_a"], row["candidate_b"]))
        counts = Counter(votes)
        best_count = max(counts.get(candidate, 0) for candidate in pair_candidates | {"TIE"})
        best = [candidate for candidate in pair_candidates | {"TIE"} if counts.get(candidate, 0) == best_count]
        winner = best[0] if len(best) == 1 else "TIE"
        outcomes.append({"item_id": item_id, "pair_id": pair_id, "votes": dict(sorted(counts.items())), "winner": winner, "n_votes": len(votes)})
    scores = {candidate: {"wins": 0, "losses": 0, "ties": 0, "copeland": 0.0} for candidate in sorted(candidates)}
    for outcome in outcomes:
        pair_candidates = set()
        for row in records:
            if row["item_id"] == outcome["item_id"] and row["pair_id"] == outcome["pair_id"]:
                pair_candidates.update((row["candidate_a"], row["candidate_b"]))
        if len(pair_candidates) != 2:
            continue
        left, right = sorted(pair_candidates)
        if outcome["winner"] == "TIE":
            scores[left]["ties"] += 1
            scores[right]["ties"] += 1
            scores[left]["copeland"] += 0.5
            scores[right]["copeland"] += 0.5
        else:
            loser = right if outcome["winner"] == left else left
            scores[outcome["winner"]]["wins"] += 1
            scores[loser]["losses"] += 1
            scores[outcome["winner"]]["copeland"] += 1.0
    ranking = sorted(scores, key=lambda candidate: (-scores[candidate]["copeland"], candidate))
    return {
        "status": "ok" if outcomes else "unavailable",
        "pair_outcomes": outcomes,
        "scores": scores,
        "ranking": ranking,
        "note": "Position-sensitive A/B pairs are excluded from consistent judge votes; they remain visible in swap_consistency.",
    }


def _group_pointwise_matrix(records: Sequence[dict[str, Any]], candidate_id: str | None = None) -> tuple[list[str], list[list[float]], list[tuple[str, str]]]:
    filtered = [row for row in records if candidate_id is None or row["candidate"]["blind_id"] == candidate_id]
    judges = sorted({row["judge"]["judge_id"] for row in filtered})
    by_unit: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in filtered:
        unit = (row["item"]["item_id"], row["candidate"]["blind_id"])
        by_unit[unit][row["judge"]["judge_id"]] = row["pointwise"]["score_0_100"]
    matrix: list[list[float]] = []
    units: list[tuple[str, str]] = []
    for unit in sorted(by_unit):
        values = by_unit[unit]
        if all(judge in values for judge in judges):
            matrix.append([values[judge] for judge in judges])
            units.append(unit)
    return judges, matrix, units


def _label_units(records: Sequence[dict[str, Any]], label_kind: str) -> list[list[Any]]:
    by_unit: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    judges = sorted({row["judge"]["judge_id"] for row in records})
    for row in records:
        errors = row["pointwise"]["errors"]
        if label_kind == "critical":
            label: Any = bool(row["pointwise"]["critical"])
        elif label_kind == "severity":
            label = max((SEVERITY_ORDER[error["severity"]] for error in errors), default=0)
        else:
            label = errors[0]["type"] if errors else "none"
        by_unit[(row["item"]["item_id"], row["candidate"]["blind_id"])][row["judge"]["judge_id"]] = label
    return [[values.get(judge) for judge in judges] for _, values in sorted(by_unit.items())]


def aggregate(records: Sequence[dict[str, Any]], pairwise_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValidationError("cannot aggregate an empty pointwise record set")
    candidates = sorted({row["candidate"]["blind_id"] for row in records})
    candidate_summary: dict[str, Any] = {}
    for candidate_id in candidates:
        candidate_rows = [row for row in records if row["candidate"]["blind_id"] == candidate_id]
        scores = [row["pointwise"]["score_0_100"] for row in candidate_rows]
        critical_rows = [row for row in candidate_rows if row["pointwise"]["critical"]]
        deterministic_rows = [row for row in candidate_rows if row["deterministic"]["critical_deterministic"]]
        critical_by_item = Counter(row["item"]["item_id"] for row in critical_rows)
        deterministic_by_item = Counter(row["item"]["item_id"] for row in deterministic_rows)
        judges, matrix, units = _group_pointwise_matrix(records, candidate_id)
        icc = icc_two_way_random_absolute(matrix)
        type_units = _label_units(candidate_rows, "type")
        severity_units = _label_units(candidate_rows, "severity")
        critical_units = _label_units(candidate_rows, "critical")
        if critical_by_item or deterministic_by_item:
            status = "FAIL_FOR_DEPLOYMENT" if deterministic_by_item or any(count >= 2 for count in critical_by_item.values()) or (_median(scores) or 0) < 80 else "REVIEW_REQUIRED"
        elif (_median(scores) or 0) < 90 or (_iqr(scores) or 0) > 10:
            status = "REVIEW_REQUIRED"
        else:
            status = "AUTO_PASS_FOR_BENCHMARK"
        candidate_summary[candidate_id] = {
            "n_records": len(candidate_rows),
            "n_items": len({row["item"]["item_id"] for row in candidate_rows}),
            "n_judges": len(judges),
            "score_median": _median(scores),
            "score_mean": _mean(scores),
            "score_iqr": _iqr(scores),
            "critical_vote_count": len(critical_rows),
            "critical_items": sorted(critical_by_item),
            "deterministic_critical_items": sorted(deterministic_by_item),
            "icc_2_1": icc.get("icc_2_1"),
            "icc_2_k": icc.get("icc_2_k"),
            "icc_detail": icc,
            "krippendorff_alpha_nominal_error_type": krippendorff_alpha(type_units, "nominal"),
            "krippendorff_alpha_ordinal_severity": krippendorff_alpha(severity_units, "ordinal", [0, 1, 2, 3]),
            "gwet_ac1_critical": gwet_ac(critical_units, "nominal"),
            "gwet_ac2_severity": gwet_ac(severity_units, "ordinal", [0, 1, 2, 3]),
            "deployment_status": status,
            "units_used_for_icc": [list(unit) for unit in units],
        }

    _, overall_matrix, overall_units = _group_pointwise_matrix(records, None)
    all_type_units = _label_units(records, "type")
    all_severity_units = _label_units(records, "severity")
    all_critical_units = _label_units(records, "critical")
    swap = swap_consistency(pairwise_records) if pairwise_records else {"status": "unavailable", "reason": "no pairwise records"}
    copeland = pairwise_copeland(pairwise_records) if pairwise_records else {"status": "unavailable", "reason": "no pairwise records"}
    return {
        "pointwise": candidate_summary,
        "reliability_overall": {
            "icc_2_1": icc_two_way_random_absolute(overall_matrix),
            "krippendorff_alpha_nominal_error_type": krippendorff_alpha(all_type_units, "nominal"),
            "krippendorff_alpha_ordinal_severity": krippendorff_alpha(all_severity_units, "ordinal", [0, 1, 2, 3]),
            "gwet_ac1_critical": gwet_ac(all_critical_units, "nominal"),
            "gwet_ac2_severity": gwet_ac(all_severity_units, "ordinal", [0, 1, 2, 3]),
            "matrix_units": [list(unit) for unit in overall_units],
        },
        "span_agreement": span_pair_agreement(records),
        "pairwise": {
            "swap_consistency": swap,
            "copeland": copeland,
        },
        "limitations": [
            "ICC(2,k) is reported for the arithmetic mean of k judges; it is not a reliability claim about a median.",
            "Synthetic anchors and offline fixture judgments are not human clinical gold.",
            "No bootstrap confidence intervals are inferred by this smoke harness.",
        ],
    }


def _schema_name(kind: str) -> str:
    mapping = {
        "anchor": "known_error_anchor.schema.json",
        "qualification": "qualification_observation.schema.json",
        "pointwise": "pointwise_judgment.schema.json",
        "pairwise": "pairwise_judgment.schema.json",
    }
    if kind not in mapping:
        raise ValidationError(f"unknown schema kind {kind!r}; choose {', '.join(sorted(mapping))}")
    return mapping[kind]


def run_smoke(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValidationError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors = read_jsonl(FIXTURE_DIR / "known_error_anchors.jsonl")
    config = read_json(FIXTURE_DIR / "fixture_config.json")
    validate_or_raise(anchors, _schema_name("anchor"), "known_error_anchors.jsonl")
    qualifications, pointwise, pairwise = materialize_fixture(config, anchors)
    validate_or_raise(qualifications, _schema_name("qualification"), "qualification_observations.jsonl")
    validate_or_raise(pointwise, _schema_name("pointwise"), "pointwise.jsonl")
    validate_or_raise(pairwise, _schema_name("pairwise"), "pairwise.jsonl")
    write_jsonl(output_dir / "known_error_anchors.jsonl", anchors)
    write_jsonl(output_dir / "qualification_observations.jsonl", qualifications)
    write_jsonl(output_dir / "pointwise.jsonl", pointwise)
    write_jsonl(output_dir / "pairwise.jsonl", pairwise)
    qualification = qualification_summary(anchors, qualifications)
    aggregation = aggregate(pointwise, pairwise)
    validation = {
        "anchors": {"records": len(anchors), "status": "PASS"},
        "qualification": {"records": len(qualifications), "status": "PASS"},
        "pointwise": {"records": len(pointwise), "status": "PASS"},
        "pairwise": {"records": len(pairwise), "status": "PASS"},
    }
    write_json(output_dir / "validation_report.json", validation)
    write_json(output_dir / "qualification_summary.json", qualification)
    write_json(output_dir / "aggregation.json", aggregation)
    checks = {
        "all_judges_qualify": qualification["panel_status"] == "PASS",
        "critical_sentinel_recall": all(value["critical_recall"] >= 0.90 for value in qualification["judges"].values()),
        "swap_consistency_at_least_0_90": aggregation["pairwise"]["swap_consistency"]["swap_consistency"] >= 0.90,
        "good_candidate_benchmark_pass": aggregation["pointwise"]["cand-good"]["deployment_status"] == "AUTO_PASS_FOR_BENCHMARK",
        "risky_candidate_fail_closed": aggregation["pointwise"]["cand-risky"]["deployment_status"] == "FAIL_FOR_DEPLOYMENT",
        "icc_estimate_available": aggregation["reliability_overall"]["icc_2_1"]["status"] == "ok",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "offline_only": True,
        "external_api_calls": 0,
        "models_selected_or_downloaded": False,
        "checks": checks,
        "record_counts": {"anchors": len(anchors), "qualification": len(qualifications), "pointwise": len(pointwise), "pairwise": len(pairwise)},
        "known_position_sensitive_pairs": aggregation["pairwise"]["swap_consistency"].get("position_sensitive_pairs"),
        "output_files": [
            "known_error_anchors.jsonl",
            "qualification_observations.jsonl",
            "pointwise.jsonl",
            "pairwise.jsonl",
            "validation_report.json",
            "qualification_summary.json",
            "aggregation.json",
            "smoke_test_result.json",
        ],
    }
    write_json(output_dir / "smoke_test_result.json", result)
    return result


def command_validate(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    errors = validate_rows(rows, SCHEMA_DIR / _schema_name(args.kind), args.input)
    result = {"status": "PASS" if not errors else "FAIL", "kind": args.kind, "records": len(rows), "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def command_qualify(args: argparse.Namespace) -> int:
    anchors = read_jsonl(Path(args.anchors))
    observations = read_jsonl(Path(args.observations))
    validate_or_raise(anchors, _schema_name("anchor"), args.anchors)
    validate_or_raise(observations, _schema_name("qualification"), args.observations)
    result = qualification_summary(anchors, observations)
    write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["panel_status"] == "PASS" else 1


def command_aggregate(args: argparse.Namespace) -> int:
    pointwise = read_jsonl(Path(args.pointwise))
    pairwise = read_jsonl(Path(args.pairwise)) if args.pairwise else []
    validate_or_raise(pointwise, _schema_name("pointwise"), args.pointwise)
    if pairwise:
        validate_or_raise(pairwise, _schema_name("pairwise"), args.pairwise)
    result = aggregate(pointwise, pairwise)
    write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a JSONL file against a checked-in schema")
    validate_parser.add_argument("--kind", choices=["anchor", "qualification", "pointwise", "pairwise"], required=True)
    validate_parser.add_argument("--input", required=True)
    validate_parser.set_defaults(function=command_validate)
    qualify_parser = subparsers.add_parser("qualify", help="aggregate qualification observations")
    qualify_parser.add_argument("--anchors", required=True)
    qualify_parser.add_argument("--observations", required=True)
    qualify_parser.add_argument("--output", required=True)
    qualify_parser.set_defaults(function=command_qualify)
    aggregate_parser = subparsers.add_parser("aggregate", help="aggregate pointwise and pairwise records")
    aggregate_parser.add_argument("--pointwise", required=True)
    aggregate_parser.add_argument("--pairwise")
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.set_defaults(function=command_aggregate)
    smoke_parser = subparsers.add_parser("smoke", help="materialize and run the offline fixture")
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.set_defaults(function=lambda args: (print(json.dumps(run_smoke(Path(args.output_dir)), ensure_ascii=False, indent=2)) or 0))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (OSError, ValidationError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
