#!/usr/bin/env python3
"""Auditable aggregation for the 2026-10-22 blind judge pilot.

This module is deliberately independent from the live runner and from the
older ``multi_judge_eval.py`` smoke harness.  It accepts the small, blind
JSON bundle described in ``README.md`` and keeps three distinctions that are
easy to lose in an LLM-judge pilot:

* repeated runs of one judge are repeatability observations, not extra judges;
* a source is the bootstrap unit (items translated from one source are not
  independent observations); and
* a result with fewer than five distinct judges cannot be an automatic
  five-judge pass.

Only the Python standard library is required.  Statistics that cannot be
identified from the supplied observations return an explicit ``unavailable``
object rather than a fabricated estimate.

Formula notes
-------------
``icc_two_way_random_absolute`` is the two-way random, absolute-agreement
ICC(A,1)/ICC(A,k) form; ``k`` is the observed number of distinct judges and
ICC(A,k) refers to the arithmetic mean of those k judges only.  The
Krippendorff implementation uses the coincidence matrix with each unit's
ordered pairs weighted by ``1/(n_u-1)``.  Its ordinal delta is the standard
ordinal distance based on pooled category proportions with half-weighted
endpoints, not a normalized numeric index.

Gwet AC1/AC2 uses the prevalence-anchored chance term
``p_e = T_w * sum(pi_k * (1-pi_k))``.  For AC1 the identity weight matrix is
used; AC2 uses an explicit linear or quadratic weight matrix.  The marginal
weighted product used by weighted kappa is intentionally not used as AC2.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "judge-pilot-aggregate-v1"
DEFAULT_REQUIRED_JUDGES = 5
SEVERITY_ORDER = {"none": 0, "minor": 1, "major": 2, "critical": 3}
SEVERITY_LABEL = {value: key for key, value in SEVERITY_ORDER.items()}

# Qualification score components are deliberately audited independently from
# MQM/error/critical fields.  A malformed scalar score must not erase useful
# evidence from the same anonymous judgment.
QUALIFICATION_SCORE_LIMITS = {
    "terminology_locale": 30,
    "semantic_safety": 30,
    "fluency": 20,
    "completeness_format": 20,
}


class ImportError(ValueError):
    """Raised when a bundle cannot be normalized without guessing."""


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON content after removing a self-referential packet digest."""

    def strip_digest_fields(child: Any) -> Any:
        if isinstance(child, dict):
            return {key: strip_digest_fields(value) for key, value in child.items() if key not in {"packet_sha256", "packet_hash", "content_sha256"}}
        if isinstance(child, list):
            return [strip_digest_fields(value) for value in child]
        return child

    value = strip_digest_fields(value)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _nested(mapping: Mapping[str, Any], *paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return default


def _as_status(row: Mapping[str, Any]) -> str:
    status = _first(row, "status", "record_status", "result_status", default="ok")
    if status is None:
        return "missing"
    status = str(status).lower()
    aliases = {"success": "ok", "valid": "ok", "error": "failed", "failure": "failed", "na": "missing"}
    return aliases.get(status, status)


def _source_hash_from_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower()):
        return value.lower()
    return sha256_text(value)


def packet_index(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a conservative blind index from common packet layouts.

    The canonical packet is intentionally allowed to evolve.  We recursively
    inspect item-shaped objects, but never infer a candidate/model mapping.
    """

    item_map: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    pair_ids: set[str] = set()
    source_hashes: dict[str, str] = {}
    source_ids: dict[str, str] = {}
    candidate_targets: dict[tuple[str, str], str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            blind_id = _first(value, "blind_id", "candidate_id")
            if isinstance(blind_id, str) and blind_id:
                candidate_ids.add(blind_id)
            item_id = _first(value, "item_id", "qualification_id", "id")
            if isinstance(item_id, str) and item_id and (
                any(key in value for key in ("source", "source_en", "source_text", "source_text_sha256", "source_hash"))
                or any(key in value for key in ("candidates", "candidate_ids", "variants", "candidate_a", "candidate_b", "translation"))
            ):
                source_value = _first(value, "source_text_sha256", "source_hash", "source_sha256")
                if source_value is None:
                    source_value = _first(value, "source_en", "source_text", "source", "source_raw")
                source_hash = _source_hash_from_value(source_value)
                if source_hash:
                    previous = source_hashes.get(item_id)
                    if previous and previous != source_hash:
                        # Keep the first value; validation reports the conflict.
                        source_hashes[item_id] = "__CONFLICT__"
                    else:
                        source_hashes[item_id] = source_hash
                source_id = _first(value, "source_id", "source_cluster", "source_group", "source_label")
                if isinstance(source_id, str) and source_id:
                    source_ids[item_id] = source_id
                item_map.setdefault(item_id, value)
                candidates = _first(value, "candidate_ids", default=[])
                if isinstance(candidates, list):
                    candidate_ids.update(str(candidate) for candidate in candidates)
                candidate_objects = _first(value, "candidates", "translations", "variants", default=[])
                if isinstance(candidate_objects, list):
                    for candidate in candidate_objects:
                        if isinstance(candidate, Mapping):
                            candidate_id = _first(candidate, "candidate_id", "blind_id", "id")
                            if isinstance(candidate_id, str):
                                candidate_ids.add(candidate_id)
                                target = _first(candidate, "target_zh_tw", "target", "translation")
                                if isinstance(target, str):
                                    candidate_targets[(item_id, candidate_id)] = target
                        elif isinstance(candidate, str):
                            candidate_ids.add(candidate)
                for candidate_key in ("candidate_a", "candidate_b"):
                    candidate = value.get(candidate_key)
                    if isinstance(candidate, Mapping):
                        candidate_id = _first(candidate, "candidate_id", "blind_id", "id")
                        target = _first(candidate, "target_zh_tw", "target", "translation")
                        if isinstance(candidate_id, str):
                            candidate_ids.add(candidate_id)
                            if isinstance(target, str):
                                candidate_targets[(item_id, candidate_id)] = target
            pair_id = _first(value, "pair_id")
            if isinstance(pair_id, str) and pair_id:
                pair_ids.add(pair_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(packet)
    return {
        "item_ids": set(item_map),
        "candidate_ids": candidate_ids,
        "pair_ids": pair_ids,
        "source_hashes": source_hashes,
        "source_ids": source_ids,
        "candidate_targets": candidate_targets,
        "items": item_map,
    }


def packet_hashes(packet_path: str | Path) -> set[str]:
    packet = read_json(packet_path)
    hashes = {sha256_file(packet_path), canonical_json_sha256(packet)}
    for key in ("packet_sha256", "packet_hash", "content_sha256"):
        value = packet.get(key) if isinstance(packet, Mapping) else None
        if isinstance(value, str) and len(value) == 64:
            hashes.add(value.lower())
    return hashes


def _iter_track_rows(bundle: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return metadata and flattened rows from a bundle or a JSON list."""

    if isinstance(bundle, list):
        return {}, [dict(row) for row in bundle if isinstance(row, Mapping)]
    if not isinstance(bundle, Mapping):
        raise ImportError("input must be a JSON object or an array of record objects")
    metadata = dict(bundle)
    rows: list[dict[str, Any]] = []

    def expand(value: Mapping[str, Any], default_track: str | None = None) -> list[dict[str, Any]]:
        track = value.get("track", default_track)
        judgments = value.get("judgments")
        # The local runner keeps qualification judgments under a
        # ``qualification`` envelope, while manual imports commonly put the
        # same list directly on the record.  Normalize both layouts to one
        # row per anonymous variant before validation/aggregation.  Do not
        # retain the envelope as a judge-visible field or infer any labels.
        if not isinstance(judgments, list) and isinstance(value.get("qualification"), Mapping):
            nested_qualification = value["qualification"]
            if isinstance(nested_qualification.get("judgments"), list):
                judgments = nested_qualification["judgments"]
                parent = {key: child for key, child in value.items() if key != "qualification"}
                # Preserve non-judgment metadata from the nested envelope,
                # while keeping the record-level fields authoritative.
                for key, child in nested_qualification.items():
                    if key != "judgments":
                        parent.setdefault(key, child)
            else:
                parent = None
        else:
            parent = {key: child for key, child in value.items() if key != "judgments"}
        if isinstance(judgments, list):
            expanded: list[dict[str, Any]] = []
            for judgment in judgments:
                if not isinstance(judgment, Mapping):
                    continue
                child = dict(parent)
                child.update(dict(judgment))
                child.setdefault("track", track or "qualification")
                expanded.append(child)
            return expanded
        copied = dict(value)
        if track is not None:
            copied.setdefault("track", track)
        return [copied]

    if isinstance(bundle.get("records"), list):
        for row in bundle["records"]:
            if isinstance(row, Mapping):
                rows.extend(expand(row, str(bundle.get("track")) if bundle.get("track") else None))
        return metadata, rows
    for track in ("qualification", "qualifications", "pointwise", "pointwise_records", "pairwise", "pairwise_records"):
        values = bundle.get(track)
        if not isinstance(values, list):
            continue
        canonical_track = "qualification" if track.startswith("qual") else ("pointwise" if track.startswith("point") else "pairwise")
        for row in values:
            if isinstance(row, Mapping):
                rows.extend(expand(row, canonical_track))
    return metadata, rows


def _track(row: Mapping[str, Any]) -> str:
    value = _first(row, "track", "kind", "task", default="pointwise")
    value = str(value).lower()
    if value in {"qual", "qualification", "qualifications"}:
        return "qualification"
    if value in {"pair", "pairwise", "abba"}:
        return "pairwise"
    return "pointwise"


def _qualification_score_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one flattened qualification variant without repairing it.

    The live runner emits ``qualification: {judgments: [...]}``; callers may
    also provide a direct flattened variant.  The returned object is safe to
    aggregate and records the original scalar/subscore values for audit.  A
    bad sum is a scalar-score validity problem only: critical/error/evidence
    fields remain available on the normalized row.
    """

    qualification = row.get("qualification") if isinstance(row.get("qualification"), Mapping) else {}
    score = _first(row, "score_0_100", "score", default=_first(qualification, "score_0_100", "score"))
    subscores = _first(row, "subscores", default=_first(qualification, "subscores"))
    abstain = _first(row, "abstain", default=_first(qualification, "abstain"))
    result: dict[str, Any] = {
        "status": "missing",
        "score_0_100": score,
        "subscores": dict(subscores) if isinstance(subscores, Mapping) else subscores,
        "subscores_sum": None,
        "reason": None,
    }
    if abstain is True:
        if score is None and subscores is None:
            result.update({"status": "abstain", "reason": "judge_abstained"})
        else:
            result.update({"status": "invalid", "reason": "abstain_with_score_or_subscores"})
        return result
    if score is None and subscores is None:
        result["reason"] = "score_and_subscores_missing"
        return result
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or not 0 <= float(score) <= 100:
        result.update({"status": "invalid", "reason": "score_missing_or_outside_0_100"})
        return result
    if not isinstance(subscores, Mapping):
        result.update({"status": "missing", "reason": "subscores_missing"})
        return result
    invalid_components: list[str] = []
    values: list[float] = []
    for key, maximum in QUALIFICATION_SCORE_LIMITS.items():
        value = subscores.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= maximum:
            invalid_components.append(f"{key}_missing_or_outside_0_{maximum}")
        else:
            values.append(float(value))
    if invalid_components:
        result.update({"status": "invalid", "reason": "invalid_subscores", "invalid_components": invalid_components})
        return result
    total = sum(values)
    result["subscores_sum"] = total
    if not math.isclose(total, float(score), rel_tol=0.0, abs_tol=1e-6):
        result.update({"status": "invalid", "reason": "subscores_sum_mismatch"})
        return result
    result["status"] = "valid"
    return result


def _normalize_row(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    track = _track(row)
    item = row.get("item") if isinstance(row.get("item"), Mapping) else {}
    candidate = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
    judge = row.get("judge") if isinstance(row.get("judge"), Mapping) else {}
    pointwise = row.get("pointwise") if isinstance(row.get("pointwise"), Mapping) else {}
    qualification = row.get("qualification") if isinstance(row.get("qualification"), Mapping) else {}
    deterministic = row.get("deterministic") if isinstance(row.get("deterministic"), Mapping) else {}
    source_value = _first(
        row,
        "source_text_sha256",
        "source_hash",
        "source_sha256",
        default=_first(item, "source_text_sha256", "source_hash", "source_sha256"),
    )
    if source_value is None:
        source_value = _first(item, "source_en", "source_text", "source", "source_raw")
    source_hash = _source_hash_from_value(source_value)
    source_id = _first(row, "source_id", "source_cluster", "source_group", default=_first(item, "source_id", "source_cluster", "source_group", "source_label"))
    item_id = _first(row, "item_id", "qualification_id", default=_first(item, "item_id", "qualification_id", "id"))
    candidate_id = _first(row, "candidate_id", "blind_id", default=_first(candidate, "candidate_id", "blind_id", "id"))
    metadata_judge = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
    metadata_judge_id = _first(metadata_judge, "judge_id", "id")
    inherited_judge_id = metadata.get("judge_id") or metadata_judge_id
    judge_id = _first(row, "judge_id", default=_first(judge, "judge_id", "id", default=inherited_judge_id))
    inherited_family = metadata.get("family") or _first(metadata_judge, "family")
    family = _first(row, "family", default=_first(judge, "family", default=inherited_family))
    run_id = _first(row, "run_id", default=_first(row.get("run"), "run_id") if isinstance(row.get("run"), Mapping) else metadata.get("run_id"))
    score = _first(row, "score_0_100", "score", default=_first(pointwise, "score_0_100", "score", default=_first(qualification, "score_0_100", "score")))
    critical = _first(row, "critical", default=_first(pointwise, "critical", default=_first(qualification, "critical")))
    errors = _first(row, "errors", default=_first(pointwise, "errors", default=_first(qualification, "errors", default=[])))
    if not isinstance(errors, list):
        errors = []
    deterministic_critical = _first(row, "critical_deterministic", default=_first(deterministic, "critical_deterministic", default=False))
    qualification_score_audit = _qualification_score_audit(row) if track == "qualification" else None
    normalized_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) else None
    # Exclude invalid/missing qualification scalar scores from downstream
    # scalar summaries, but retain the raw fields and audit object above.
    if track == "qualification" and qualification_score_audit["status"] != "valid":
        normalized_score = None
    return {
        **dict(row),
        "_track": track,
        "_status": _as_status(row),
        "_item_id": str(item_id) if item_id is not None else None,
        "_candidate_id": str(candidate_id) if candidate_id is not None else None,
        "_judge_id": str(judge_id) if judge_id is not None else None,
        "_family": str(family) if family is not None else None,
        "_run_id": str(run_id) if run_id is not None else None,
        "_source_id": str(source_id) if source_id is not None else (str(item_id) if item_id is not None else None),
        "_source_hash": source_hash,
        "_score": normalized_score,
        "_critical": bool(critical) if isinstance(critical, bool) else None,
        "_errors": errors,
        "_deterministic_critical": bool(deterministic_critical),
        "_qualification_score_audit": qualification_score_audit,
    }


def load_bundle(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = read_json(path)
    metadata, rows = _iter_track_rows(bundle)
    return metadata, [_normalize_row(row, metadata) for row in rows]


def load_bundles(paths: Sequence[str | Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge independently imported judge bundles without inventing a panel."""

    if not paths:
        raise ImportError("at least one input bundle is required")
    all_rows: list[dict[str, Any]] = []
    metadata_list: list[dict[str, Any]] = []
    for path in paths:
        metadata, rows = load_bundle(path)
        metadata_list.append(metadata)
        all_rows.extend(rows)
    packet_ids = {metadata.get("packet_id") for metadata in metadata_list if metadata.get("packet_id") is not None}
    packet_hash_values = {metadata.get("packet_sha256") or metadata.get("packet_hash") for metadata in metadata_list if metadata.get("packet_sha256") or metadata.get("packet_hash")}
    if len(packet_ids) > 1:
        raise ImportError(f"cannot merge bundles with different packet_id values: {sorted(packet_ids)}")
    if len(packet_hash_values) > 1:
        raise ImportError(f"cannot merge bundles with different packet hashes: {sorted(packet_hash_values)}")
    merged = {
        "packet_id": next(iter(packet_ids), None),
        "packet_sha256": next(iter(packet_hash_values), None),
        "judges": [metadata.get("judge") or {"judge_id": metadata.get("judge_id"), "family": metadata.get("family"), "model_label_from_ui": metadata.get("model_label_from_ui")} for metadata in metadata_list],
        "source_bundles": [str(path) for path in paths],
        "source_metadata": [metadata.get("metadata") for metadata in metadata_list if isinstance(metadata.get("metadata"), Mapping)],
    }
    fresh_chat_values = [metadata.get("metadata", {}).get("pairwise_fresh_chat_independence") for metadata in metadata_list if isinstance(metadata.get("metadata"), Mapping) and "pairwise_fresh_chat_independence" in metadata.get("metadata", {})]
    limitation_values = [metadata.get("metadata", {}).get("pairwise_limitation") for metadata in metadata_list if isinstance(metadata.get("metadata"), Mapping) and metadata.get("metadata", {}).get("pairwise_limitation")]
    if fresh_chat_values:
        merged["pairwise_fresh_chat_independence"] = fresh_chat_values[0] if len(set(fresh_chat_values)) == 1 else fresh_chat_values
    if limitation_values:
        merged["pairwise_limitation"] = limitation_values[0] if len(set(limitation_values)) == 1 else limitation_values
    return merged, all_rows


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


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


def _iqr(values: Sequence[float]) -> float | None:
    q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
    return None if q1 is None or q3 is None else q3 - q1


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unavailable", "estimate": None, "reason": reason, **extra}


def icc_two_way_random_absolute(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """ICC(2,1) and ICC(2,k), where k is the observed judge count.

    Rows are units and columns are distinct judges.  This function must be
    called on complete rows after repeat runs have been collapsed.  The
    average-measure estimate is for the arithmetic mean, never a median.
    """

    n = len(matrix)
    k = len(matrix[0]) if n else 0
    if n < 2 or k < 2:
        return unavailable("need at least 2 complete units and 2 distinct judges", n_units=n, n_judges=k)
    if any(len(row) != k for row in matrix):
        return unavailable("ragged score matrix")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for row in matrix for value in row):
        return unavailable("non-finite score in matrix")
    grand_mean = statistics.fmean(value for row in matrix for value in row)
    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean(matrix[i][j] for i in range(n)) for j in range(k)]
    ms_rows = k * sum((mean - grand_mean) ** 2 for mean in row_means) / (n - 1)
    ms_cols = n * sum((mean - grand_mean) ** 2 for mean in col_means) / (k - 1)
    residual = sum((matrix[i][j] - row_means[i] - col_means[j] + grand_mean) ** 2 for i in range(n) for j in range(k))
    ms_error = residual / ((n - 1) * (k - 1))
    denominator_single = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    denominator_average = ms_rows + (ms_cols - ms_error) / n
    if denominator_single == 0 or denominator_average == 0:
        return unavailable("zero ICC denominator", n_units=n, n_judges=k)
    return {
        "status": "ok",
        "icc_2_1": (ms_rows - ms_error) / denominator_single,
        "icc_2_k": (ms_rows - ms_error) / denominator_average,
        "n_units": n,
        "n_judges": k,
        "model": "two-way random, absolute agreement",
        "unit": "ICC(2,1) single judge; ICC(2,k) arithmetic mean of observed k judges",
        "median_reliability": "not estimated",
    }


def _ordinal_distance(first: Any, second: Any, levels: Sequence[Any], probabilities: Mapping[Any, float]) -> float:
    if first == second:
        return 0.0
    try:
        left, right = sorted((levels.index(first), levels.index(second)))
    except ValueError:
        return 1.0
    # Krippendorff's ordinal delta uses the pooled cumulative probability
    # between the categories with half weight at both endpoints:
    # [sum_{g=left..right} p_g - (p_left+p_right)/2]^2.  This is not the
    # arbitrary numeric index distance and is symmetric at category bounds.
    cumulative = sum(probabilities.get(level, 0.0) for level in levels[left : right + 1])
    endpoint_midpoint = (probabilities.get(levels[left], 0.0) + probabilities.get(levels[right], 0.0)) / 2.0
    return (cumulative - endpoint_midpoint) ** 2


def _coincidence_matrix(units: Sequence[Sequence[Any]]) -> tuple[dict[tuple[Any, Any], float], Counter[Any], int]:
    coincidence: dict[tuple[Any, Any], float] = defaultdict(float)
    pooled: Counter[Any] = Counter()
    total = 0
    for unit in units:
        labels = [label for label in unit if label is not None]
        n = len(labels)
        if n < 2:
            continue
        counts = Counter(labels)
        pooled.update(labels)
        total += n
        for first, n_first in counts.items():
            for second, n_second in counts.items():
                coincidence[(first, second)] += n_first * (n_second - (1 if first == second else 0)) / (n - 1)
    return coincidence, pooled, total


def krippendorff_alpha(units: Sequence[Sequence[Any]], kind: str = "nominal", levels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Krippendorff alpha with correct unequal-rater coincidence weighting."""

    clean_units = [[label for label in unit if label is not None] for unit in units if len([label for label in unit if label is not None]) >= 2]
    if not clean_units:
        return unavailable("need at least one unit with two non-missing annotations")
    coincidence, pooled, total = _coincidence_matrix(clean_units)
    if total < 2 or len(pooled) < 2:
        return unavailable("zero expected disagreement", n_units=len(clean_units), n_annotations=total)
    labels = list(levels) if levels is not None else list(pooled)
    probabilities = {label: count / total for label, count in pooled.items()}

    def distance(first: Any, second: Any) -> float:
        if kind == "nominal":
            return 0.0 if first == second else 1.0
        if kind == "ordinal":
            return _ordinal_distance(first, second, labels, probabilities)
        if kind == "interval":
            try:
                return (float(first) - float(second)) ** 2
            except (TypeError, ValueError):
                return 1.0
        raise ValueError(f"unsupported alpha kind {kind!r}")

    observed_disagreement = sum(value * distance(first, second) for (first, second), value in coincidence.items()) / total
    expected_denominator = total - 1
    expected_disagreement = sum(
        pooled[first] * (pooled[second] - (1 if first == second else 0)) * distance(first, second) / expected_denominator
        for first in pooled
        for second in pooled
    ) / total
    if expected_disagreement == 0:
        return unavailable("zero expected disagreement", n_units=len(clean_units), n_annotations=total)
    return {
        "status": "ok",
        "estimate": 1.0 - observed_disagreement / expected_disagreement,
        "kind": kind,
        "distance": "nominal equality" if kind == "nominal" else ("Krippendorff ordinal squared pooled cumulative midpoint probability" if kind == "ordinal" else "squared interval difference"),
        "n_units": len(clean_units),
        "n_annotations": total,
        "missing_annotations": sum(len(unit) for unit in units) - total,
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
    }


def _weight_matrix(levels: Sequence[Any], scheme: str = "quadratic") -> dict[tuple[Any, Any], float]:
    if len(levels) < 2:
        return {(level, level): 1.0 for level in levels}
    denominator = len(levels) - 1
    weights: dict[tuple[Any, Any], float] = {}
    for first in levels:
        for second in levels:
            distance = abs(levels.index(first) - levels.index(second)) / denominator
            weights[(first, second)] = 1.0 - (distance ** 2 if scheme == "quadratic" else distance)
    return weights


def gwet_ac(units: Sequence[Sequence[Any]], kind: str = "nominal", levels: Sequence[Any] | None = None, weight: str = "quadratic") -> dict[str, Any]:
    """Gwet AC1/AC2 for equal-weight units with missing annotations allowed.

    AC1 uses identity weights and AC2 uses explicit linear/quadratic weights.
    The chance term follows Gwet/irrCAC's prevalence-anchored formula, not
    the weighted-kappa expected agreement.
    """

    clean_units = [[label for label in unit if label is not None] for unit in units if len([label for label in unit if label is not None]) >= 2]
    if not clean_units:
        return unavailable("need at least one unit with two non-missing annotations")
    categories = list(levels) if levels is not None else list(dict.fromkeys(label for unit in clean_units for label in unit))
    if len(categories) < 2:
        return unavailable("need at least two categories")
    weights = {(first, second): (1.0 if first == second else 0.0) for first in categories for second in categories}
    metric = "AC1"
    if kind != "nominal":
        weights = _weight_matrix(categories, weight)
        metric = "AC2"
    observed_unit_values: list[float] = []
    observed_pair_count = 0
    unit_probabilities: list[dict[Any, float]] = []
    for unit in clean_units:
        counts = Counter(unit)
        unit_probabilities.append({category: counts.get(category, 0) / len(unit) for category in categories})
        unit_values: list[float] = []
        for left, right in itertools.combinations(range(len(unit)), 2):
            unit_values.append(weights.get((unit[left], unit[right]), 0.0))
        if unit_values:
            observed_unit_values.append(statistics.fmean(unit_values))
            observed_pair_count += len(unit_values)
    if not observed_unit_values:
        return unavailable("no annotation pairs")
    # Gwet's observed agreement is averaged over units, not over all pairs;
    # otherwise units with more available raters receive extra weight while
    # the marginal chance term remains unit-weighted.
    observed = statistics.fmean(observed_unit_values)
    pi = {category: statistics.fmean(probability[category] for probability in unit_probabilities) for category in categories}
    q = len(categories)
    tw = sum(weights.values()) / (q * (q - 1))
    expected = tw * sum(probability * (1.0 - probability) for probability in pi.values())
    if expected == 1.0:
        return unavailable("zero chance-disagreement denominator", metric=metric)
    return {
        "status": "ok",
        "estimate": (observed - expected) / (1.0 - expected),
        "metric": metric,
        "weight": "identity" if metric == "AC1" else weight,
        "observed_agreement": observed,
        "chance_agreement_gwet": expected,
        "marginal_probabilities": pi,
        "n_units": len(clean_units),
        "n_pairs": observed_pair_count,
        "n_units_with_pairs": len(observed_unit_values),
        "chance_term_note": "Gwet prevalence-anchored p_e; not weighted-kappa marginal product; observed agreement is equal-weighted across units with available annotation pairs",
    }


def _record_unit(row: Mapping[str, Any], candidate_specific: bool = True) -> tuple[str, ...]:
    return (
        str(row.get("_source_id") or row.get("_item_id")),
        str(row.get("_item_id")),
        str(row.get("_candidate_id")) if candidate_specific else "*",
    )


def _collapse_pointwise(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["_item_id"], row["_candidate_id"], row["_judge_id"])].append(row)
    collapsed: list[dict[str, Any]] = []
    repeat_disagreements: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda value: tuple(str(part) for part in value[0])):
        ok_rows = [row for row in group if row["_status"] == "ok" and row["_score"] is not None]
        status_values = Counter(row["_status"] for row in group)
        if not ok_rows:
            representative = dict(group[-1])
            representative.update({"_status": "failed" if status_values.get("failed") else "missing", "_score": None, "_critical": None, "_errors": []})
            collapsed.append(representative)
            continue
        scores = [row["_score"] for row in ok_rows if row["_score"] is not None]
        median = statistics.median(scores)
        representative = min(ok_rows, key=lambda row: (abs(float(row["_score"]) - median), str(row.get("_run_id"))))
        result = dict(representative)
        result["_score"] = median
        result["_critical"] = any(row["_critical"] is True for row in ok_rows)
        result["_deterministic_critical"] = any(row["_deterministic_critical"] for row in ok_rows)
        result["_repeat_count"] = len(ok_rows)
        result["_repeat_statuses"] = dict(status_values)
        unique_scores = sorted({round(float(score), 9) for score in scores})
        critical_values = {row["_critical"] for row in ok_rows if row["_critical"] is not None}
        if len(unique_scores) > 1 or len(critical_values) > 1:
            repeat_disagreements.append({"item_id": key[0], "candidate_id": key[1], "judge_id": key[2], "score_values": unique_scores, "critical_values": sorted(str(value) for value in critical_values)})
        collapsed.append(result)
    return collapsed, {"logical_units": len(grouped), "collapsed_units": len(collapsed), "repeat_disagreements": repeat_disagreements, "n_repeat_runs": sum(max(0, len(group) - 1) for group in grouped.values())}


def _label_from_row(row: Mapping[str, Any], label_kind: str) -> Any:
    if label_kind == "critical":
        return row.get("_critical")
    if label_kind == "severity":
        return max((SEVERITY_ORDER.get(str(error.get("severity")), 0) for error in row.get("_errors", []) if isinstance(error, Mapping)), default=0)
    types = sorted({str(error.get("type")) for error in row.get("_errors", []) if isinstance(error, Mapping) and error.get("type")})
    if not types:
        return "none"
    return types[0] if len(types) == 1 else "multiple:" + "|".join(types)


def _unit_label_matrix(rows: Sequence[dict[str, Any]], label_kind: str) -> list[list[Any]]:
    by_unit: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["_status"] != "ok" or row["_judge_id"] is None:
            continue
        by_unit[_record_unit(row)][str(row["_judge_id"])] = _label_from_row(row, label_kind)
    matrices: list[list[Any]] = []
    for values in by_unit.values():
        matrices.append([value for value in values.values() if value is not None])
    return matrices


def nominal_agreement(rows: Sequence[dict[str, Any]], label_kind: str = "type") -> dict[str, Any]:
    """Pairwise raw agreement with unequal/missing judge counts explicit."""

    by_unit: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["_status"] == "ok" and row["_judge_id"] is not None:
            by_unit[_record_unit(row)][str(row["_judge_id"])] = _label_from_row(row, label_kind)
    pair_values: list[float] = []
    complete_units = 0
    for values in by_unit.values():
        if len(values) < 2:
            continue
        complete_units += 1
        pair_values.extend(1.0 if first == second else 0.0 for first, second in itertools.combinations(values.values(), 2))
    if not pair_values:
        return unavailable("no unit has two valid distinct judges", label=label_kind)
    return {"status": "ok", "estimate": statistics.fmean(pair_values), "label": label_kind, "pair_count": len(pair_values), "units_with_two_judges": complete_units, "note": "pairwise raw agreement; missing judges are excluded and counted separately"}


def binary_agreement(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["_status"] == "ok" and row["_judge_id"] is not None and row["_critical"] is not None:
            by_unit[_record_unit(row)][str(row["_judge_id"])] = bool(row["_critical"])
    raw: list[float] = []
    positive: list[float] = []
    negative: list[float] = []
    pair_count = 0
    units = 0
    for values in by_unit.values():
        if len(values) < 2:
            continue
        units += 1
        for first, second in itertools.combinations(values.values(), 2):
            pair_count += 1
            raw.append(float(first == second))
            denominator_positive = int(first) + int(second)
            denominator_negative = int(not first) + int(not second)
            # Positive/negative agreement is undefined when its denominator
            # is zero.  Exclude that pair from the corresponding mean rather
            # than turning an all-negative panel into a spurious perfect
            # positive-agreement estimate.
            if denominator_positive:
                positive.append(2.0 * int(first and second) / denominator_positive)
            if denominator_negative:
                negative.append(2.0 * int((not first) and (not second)) / denominator_negative)
    if not raw:
        return unavailable("no unit has two valid distinct judges", pair_count=0)
    return {"status": "ok", "raw_agreement": statistics.fmean(raw), "positive_agreement": _mean(positive), "negative_agreement": _mean(negative), "pair_count": pair_count, "positive_pair_count": len(positive), "negative_pair_count": len(negative), "units_with_two_judges": units, "definition": "pairwise binary agreement; positive agreement is 2*both-positive/(positive_i+positive_j), and negative agreement is its negative-label analogue; a zero denominator is undefined and excluded from that mean"}


def _span_bounds(value: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        start, end = int(value["target_start"]), int(value["target_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def span_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_bounds = _span_bounds(first)
    second_bounds = _span_bounds(second)
    if first_bounds is None or second_bounds is None:
        return 0.0
    first_start, first_end = first_bounds
    second_start, second_end = second_bounds
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return 0.0 if union <= 0 else intersection / union


def _greedy_span_match(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> tuple[int, int, int, float, int]:
    first = [span for span in first if _span_bounds(span) is not None]
    second = [span for span in second if _span_bounds(span) is not None]
    candidates = sorted(((span_iou(left, right), left_index, right_index) for left_index, left in enumerate(first) for right_index, right in enumerate(second)), reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = 0
    exact = 0
    iou_sum = 0.0
    for iou, left_index, right_index in candidates:
        if iou < threshold or left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches += 1
        exact += int(first[left_index].get("target_start") == second[right_index].get("target_start") and first[left_index].get("target_end") == second[right_index].get("target_end"))
        iou_sum += iou
    return matches, exact, len(first), iou_sum, len(second)


def span_agreement(rows: Sequence[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    by_unit: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for row in rows:
        if row["_status"] == "ok" and row["_judge_id"] is not None:
            by_unit[_record_unit(row)][str(row["_judge_id"])] = [error for error in row.get("_errors", []) if isinstance(error, Mapping)]
    pair_stats: list[dict[str, float]] = []
    missing_units = 0
    missing_offsets = 0
    pairs_without_coordinate_spans = 0
    for values in by_unit.values():
        if len(values) < 2:
            missing_units += 1
            continue
        for first, second in itertools.combinations(values.values(), 2):
            first_spans = [error for error in first if _span_bounds(error) is not None]
            second_spans = [error for error in second if _span_bounds(error) is not None]
            missing_offsets += (len(first) - len(first_spans)) + (len(second) - len(second_spans))
            if not first_spans and not second_spans and (first or second):
                pairs_without_coordinate_spans += 1
                continue
            if first and not first_spans or second and not second_spans:
                # A row with only quote/source evidence cannot be treated as
                # a coordinate-based disagreement with another row.
                pairs_without_coordinate_spans += 1
                continue
            matches, exact, left_count, iou_sum, right_count = _greedy_span_match(first_spans, second_spans, threshold)
            precision = matches / left_count if left_count else (1.0 if right_count == 0 else 0.0)
            recall = matches / right_count if right_count else (1.0 if left_count == 0 else 0.0)
            # Zero precision and recall means two non-empty, non-overlapping
            # error sets disagree; perfect F1 is reserved for two empty sets.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else (1.0 if left_count == right_count == 0 else 0.0)
            pair_stats.append({"precision": precision, "recall": recall, "f1": f1, "exact_match_rate": exact / matches if matches else (1.0 if left_count == right_count == 0 else 0.0), "mean_iou": iou_sum / matches if matches else None})
    if not pair_stats:
        return {"status": "unavailable", "reason": "no comparable coordinate spans; missing target_start/target_end must not be treated as non-overlap", "threshold_iou": threshold, "pair_count": 0, "missing_units_without_two_judges": missing_units, "pairs_without_coordinate_spans": pairs_without_coordinate_spans, "missing_offsets": missing_offsets}
    return {"status": "partial" if missing_offsets else "ok", "threshold_iou": threshold, "pair_count": len(pair_stats), "precision": statistics.fmean(stat["precision"] for stat in pair_stats), "recall": statistics.fmean(stat["recall"] for stat in pair_stats), "f1": statistics.fmean(stat["f1"] for stat in pair_stats), "exact_match_rate": statistics.fmean(stat["exact_match_rate"] for stat in pair_stats), "mean_iou_of_matches": _mean([stat["mean_iou"] for stat in pair_stats if stat["mean_iou"] is not None]), "missing_units_without_two_judges": missing_units, "pairs_without_coordinate_spans": pairs_without_coordinate_spans, "missing_offsets": missing_offsets, "note": "Only errors with valid positive-length target_start/target_end are compared; missing/zero-length offsets are reported as partial and never scored as non-overlap."}


def _span_consensus(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["_status"] == "ok":
            by_unit[_record_unit(row)].append(row)
    consensus, disputed, singleton, total, missing_offsets, coordinate_spans = 0, 0, 0, 0, 0, 0
    for unit_rows in by_unit.values():
        clusters: list[dict[str, Any]] = []
        for row in unit_rows:
            judge_id = row.get("_judge_id")
            for error in row.get("_errors", []):
                if not isinstance(error, Mapping):
                    continue
                total += 1
                if _span_bounds(error) is None:
                    missing_offsets += 1
                    continue
                coordinate_spans += 1
                best = None
                best_iou = 0.0
                for cluster in clusters:
                    iou = span_iou(error, cluster["representative"])
                    if iou >= 0.5 and iou > best_iou:
                        best, best_iou = cluster, iou
                if best is None:
                    clusters.append({"representative": dict(error), "judges": {judge_id}, "max_severity": SEVERITY_ORDER.get(str(error.get("severity")), 0)})
                else:
                    best["judges"].add(judge_id)
                    best["max_severity"] = max(best["max_severity"], SEVERITY_ORDER.get(str(error.get("severity")), 0))
        available_judges = len({row.get("_judge_id") for row in unit_rows})
        # Strict majority; for an even panel ceil(n/2) is only half.
        majority = max(2, available_judges // 2 + 1)
        for cluster in clusters:
            coverage = len(cluster["judges"])
            if coverage >= majority:
                consensus += 1
            elif coverage >= 2:
                disputed += 1
            else:
                singleton += 1
    return {"status": "partial" if missing_offsets else "ok", "raw_error_spans": total, "coordinate_error_spans": coordinate_spans, "missing_offsets": missing_offsets, "panel_consensus_error_clusters": consensus, "disputed_error_clusters": disputed, "singleton_error_clusters": singleton, "definition": "target-side IoU >= 0.50 over positive-length coordinate spans; one judge contributes at most once per cluster; strict majority is floor(available judges/2)+1, minimum 2; missing/zero-length offsets are excluded and reported"}


def source_cluster_bootstrap(values_by_source: Mapping[str, Sequence[float]], statistic: str = "mean", replicates: int = 2000, seed: int = 20261022) -> dict[str, Any]:
    sources = [source for source, values in values_by_source.items() if _finite(values)]
    if len(sources) < 2:
        return unavailable("need at least 2 source clusters", n_source_clusters=len(sources), bootstrap_unit="source_id")
    cleaned = {source: _finite(values) for source, values in values_by_source.items() if _finite(values)}

    def summarize(source_sample: Sequence[str]) -> float | None:
        values = [value for source in source_sample for value in cleaned[source]]
        if not values:
            return None
        return _mean(values) if statistic == "mean" else _median(values)

    observed = summarize(sources)
    rng = random.Random(seed)
    estimates = [estimate for estimate in (summarize([rng.choice(sources) for _ in sources]) for _ in range(max(1, replicates))) if estimate is not None]
    if observed is None or not estimates:
        return unavailable("bootstrap statistic undefined", n_source_clusters=len(sources), bootstrap_unit="source_id")
    unstable = len(sources) < 5
    return {"status": "ok", "estimate": observed, "ci_95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)], "n_source_clusters": len(sources), "bootstrap_replicates": len(estimates), "bootstrap_unit": "source_id", "statistic": statistic, "stability": "unstable_small_source_scope" if unstable else "descriptive", "warning": "only 3 source clusters: percentile bootstrap intervals are descriptive and can be narrow/unstable" if len(sources) == 3 else ("few source clusters; interpret cautiously" if unstable else None)}


def _group_score_matrix(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[float]], list[tuple[str, str, str]]]:
    judges = sorted({str(row["_judge_id"]) for row in rows if row["_status"] == "ok" and row["_judge_id"] is not None and row["_score"] is not None})
    by_unit: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["_status"] == "ok" and row["_judge_id"] is not None and row["_score"] is not None:
            by_unit[_record_unit(row)][str(row["_judge_id"])] = float(row["_score"])
    matrix, units = [], []
    for unit, values in sorted(by_unit.items()):
        if len(values) == len(judges):
            matrix.append([values[judge] for judge in judges])
            units.append(unit)
    return judges, matrix, units


def kendall_w(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tie-aware Kendall W on judge rankings per source/item.

    This is a concordance diagnostic over complete candidate-by-judge score
    tables.  It is not a claim that the ranking is correct.
    """

    by_source_item: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["_status"] == "ok" and row["_judge_id"] is not None and row["_score"] is not None:
            by_source_item[(str(row["_source_id"]), str(row["_item_id"]))][str(row["_judge_id"])][str(row["_candidate_id"])] = float(row["_score"])
    estimates: list[float] = []
    detail: list[dict[str, Any]] = []
    for (source_id, item_id), judge_scores in sorted(by_source_item.items()):
        judges = sorted(judge_scores)
        candidates = sorted(set().union(*(scores.keys() for scores in judge_scores.values())))
        if len(judges) < 2 or len(candidates) < 2 or any(set(judge_scores[judge]) != set(candidates) for judge in judges):
            continue
        rank_rows: list[list[float]] = []
        tie_correction = 0.0
        for judge in judges:
            ordered = sorted(candidates, key=lambda candidate: (-judge_scores[judge][candidate], candidate))
            ranks: dict[str, float] = {}
            index = 0
            while index < len(ordered):
                end = index + 1
                while end < len(ordered) and judge_scores[judge][ordered[end]] == judge_scores[judge][ordered[index]]:
                    end += 1
                rank = (index + 1 + end) / 2
                for candidate in ordered[index:end]:
                    ranks[candidate] = rank
                tie_size = end - index
                tie_correction += tie_size ** 3 - tie_size
                index = end
            rank_rows.append([ranks[candidate] for candidate in candidates])
        m, n = len(judges), len(candidates)
        rank_sums = [sum(row[index] for row in rank_rows) for index in range(n)]
        mean_rank = m * (n + 1) / 2
        s_value = sum((rank_sum - mean_rank) ** 2 for rank_sum in rank_sums)
        denominator = m * m * (n ** 3 - n) - m * tie_correction
        if denominator <= 0:
            continue
        estimate = max(0.0, min(1.0, 12 * s_value / denominator))
        estimates.append(estimate)
        detail.append({"source_id": source_id, "item_id": item_id, "n_judges": m, "n_candidates": n, "estimate": estimate})
    if not estimates:
        return unavailable("need complete rankings for at least two judges and two candidates")
    return {"status": "ok", "estimate": statistics.fmean(estimates), "per_source_item": detail, "n_rankings": len(estimates), "scope": "source/item complete candidate rankings; tie-corrected Kendall W", "note": "concordance only, not correctness or source-independent validity"}


def _pair_winner(row: Mapping[str, Any]) -> str | None:
    if row.get("_status") != "ok":
        return None
    winner = row.get("winner", row.get("pairwise", {}).get("winner") if isinstance(row.get("pairwise"), Mapping) else None)
    if winner in {"TIE", "tie", "Tie"}:
        return "TIE"
    candidate_a = row.get("candidate_a")
    candidate_b = row.get("candidate_b")
    if winner == "A":
        return str(candidate_a) if candidate_a is not None else None
    if winner == "B":
        return str(candidate_b) if candidate_b is not None else None
    if isinstance(winner, str) and winner:
        return winner
    return None


def _collapse_pairwise(rows: Sequence[dict[str, Any]]) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["_status"] != "ok":
            continue
        orientation = str(row.get("orientation", "")).upper()
        if orientation not in {"AB", "BA"}:
            continue
        groups[(str(row["_item_id"]), str(row.get("pair_id")), str(row["_judge_id"]), orientation)].append(row)
    collapsed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    repeat_conflicts: list[dict[str, Any]] = []
    for key, group in groups.items():
        winners = [_pair_winner(row) for row in group]
        winners = [winner for winner in winners if winner is not None]
        if not winners:
            continue
        counts = Counter(winners)
        best_count = max(counts.values())
        best = sorted(winner for winner, count in counts.items() if count == best_count)
        if len(best) != 1:
            repeat_conflicts.append({"item_id": key[0], "pair_id": key[1], "judge_id": key[2], "orientation": key[3], "winners": dict(counts)})
            continue
        result = dict(group[0])
        result["_canonical_winner"] = best[0]
        result["_repeat_count"] = len(winners)
        collapsed[key] = result
    return collapsed, repeat_conflicts


def _comparison_components(candidate_ids: Iterable[str], pair_outcomes: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Return connected components of the observed, canonical pair graph."""

    nodes = {str(candidate) for candidate in candidate_ids}
    adjacency: dict[str, set[str]] = {candidate: set() for candidate in nodes}
    for outcome in pair_outcomes:
        pair_candidates = [str(candidate) for candidate in outcome.get("candidates", [])]
        if len(pair_candidates) != 2:
            continue
        left, right = pair_candidates
        nodes.update((left, right))
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components: list[list[str]] = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: set[str] = set()
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0] if component else "")


def _component_rankings(scores: Mapping[str, Mapping[str, Any]], components: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for component in components:
        ranking = sorted(component, key=lambda candidate: (-float(scores[candidate]["copeland"]), candidate))
        rankings.append({"candidates": list(component), "ranking": ranking, "copeland_scores": {candidate: dict(scores[candidate]) for candidate in ranking}})
    return rankings


def pairwise_abba_copeland(rows: Sequence[dict[str, Any]], candidate_universe: Iterable[str] | None = None) -> dict[str, Any]:
    collapsed, repeat_conflicts = _collapse_pairwise(rows)
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    candidates: set[str] = set()
    for key, row in collapsed.items():
        item_id, pair_id, judge_id, orientation = key
        candidate_a, candidate_b = str(row.get("candidate_a")), str(row.get("candidate_b"))
        candidates.update((candidate_a, candidate_b))
        groups[(item_id, pair_id, judge_id)][orientation] = row
    consistent_votes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts = {"complete": 0, "consistent": 0, "position_sensitive": 0, "tie_disagreement": 0, "incomplete": 0}
    for (item_id, pair_id, judge_id), orientations in groups.items():
        if "AB" not in orientations or "BA" not in orientations:
            counts["incomplete"] += 1
            continue
        counts["complete"] += 1
        first = orientations["AB"]["_canonical_winner"]
        second = orientations["BA"]["_canonical_winner"]
        if first == second:
            counts["consistent"] += 1
            consistent_votes[(item_id, pair_id)].append({"winner": first, "judge_id": judge_id, "family": orientations["AB"].get("_family")})
        elif first != "TIE" and second != "TIE":
            counts["position_sensitive"] += 1
        else:
            counts["tie_disagreement"] += 1
    pair_outcomes: list[dict[str, Any]] = []
    family_balanced_outcomes: list[dict[str, Any]] = []
    for key, vote_details in sorted(consistent_votes.items()):
        votes = [detail["winner"] for detail in vote_details]
        vote_counts = Counter(votes)
        best_count = max(vote_counts.values())
        winners = sorted(candidate for candidate, count in vote_counts.items() if count == best_count)
        winner = winners[0] if len(winners) == 1 else "TIE"
        pair_candidates: set[str] = set()
        for collapsed_key, collapsed_row in collapsed.items():
            if collapsed_key[0] == key[0] and collapsed_key[1] == key[1]:
                pair_candidates.update((str(collapsed_row.get("candidate_a")), str(collapsed_row.get("candidate_b"))))
        pair_outcomes.append({"item_id": key[0], "pair_id": key[1], "winner": winner, "candidates": sorted(pair_candidates), "votes": dict(sorted(vote_counts.items())), "n_distinct_judges": len(votes)})
        family_votes: dict[str, list[str]] = defaultdict(list)
        for detail in vote_details:
            if detail.get("family"):
                family_votes[str(detail["family"])].append(detail["winner"])
        if family_votes:
            balanced_candidate_scores: dict[str, float] = defaultdict(float)
            balanced_tie_score = 0.0
            for family, family_winners in family_votes.items():
                family_counts = Counter(family_winners)
                family_total = len(family_winners)
                for candidate in pair_candidates:
                    balanced_candidate_scores[candidate] += family_counts.get(candidate, 0) / family_total
                # Keep TIE as an explicit third outcome during family
                # balancing.  It is deliberately not added to
                # ``family_vote_scores``: that candidate-only mapping feeds
                # the observed comparison graph below.  The separate score
                # preserves all family vote mass without changing the
                # primary candidate-vote rule or introducing half-votes.
                balanced_tie_score += family_counts.get("TIE", 0) / family_total
            n_families = len(family_votes)
            balanced_candidate_scores = {candidate: value / n_families for candidate, value in balanced_candidate_scores.items()}
            balanced_tie_score /= n_families
            balanced_outcome_scores = {**balanced_candidate_scores, "TIE": balanced_tie_score}
            balanced_best = max(balanced_outcome_scores.values())
            balanced_winners = sorted(outcome for outcome, value in balanced_outcome_scores.items() if math.isclose(value, balanced_best))
            family_balanced_outcomes.append({"item_id": key[0], "pair_id": key[1], "winner": balanced_winners[0] if len(balanced_winners) == 1 else "TIE", "candidates": sorted(pair_candidates), "family_vote_scores": balanced_candidate_scores, "family_tie_vote_score": balanced_tie_score, "family_outcome_vote_scores": balanced_outcome_scores, "n_families": n_families})
    all_candidates = set(candidates)
    if candidate_universe is not None:
        all_candidates.update(str(candidate) for candidate in candidate_universe)
    scores = {candidate: {"wins": 0, "losses": 0, "ties": 0, "copeland": 0.0} for candidate in sorted(all_candidates)}
    for outcome in pair_outcomes:
        pair_candidates = set(outcome.get("candidates", []))
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
    balanced_scores = {candidate: {"wins": 0, "losses": 0, "ties": 0, "copeland": 0.0} for candidate in sorted(all_candidates)}
    for outcome in family_balanced_outcomes:
        pair_candidates = set(outcome["family_vote_scores"])
        if len(pair_candidates) != 2:
            continue
        left, right = sorted(pair_candidates)
        if outcome["winner"] == "TIE":
            balanced_scores[left]["ties"] += 1
            balanced_scores[right]["ties"] += 1
            balanced_scores[left]["copeland"] += 0.5
            balanced_scores[right]["copeland"] += 0.5
        else:
            loser = right if outcome["winner"] == left else left
            balanced_scores[outcome["winner"]]["wins"] += 1
            balanced_scores[loser]["losses"] += 1
            balanced_scores[outcome["winner"]]["copeland"] += 1.0
    balanced_ranking = sorted(balanced_scores, key=lambda candidate: (-balanced_scores[candidate]["copeland"], candidate))
    components = _comparison_components(all_candidates, pair_outcomes)
    balanced_components = _comparison_components(all_candidates, family_balanced_outcomes)
    global_available = len(components) == 1 and len(components[0]) == len(all_candidates) and bool(pair_outcomes)
    balanced_global_available = len(balanced_components) == 1 and len(balanced_components[0]) == len(all_candidates) and bool(family_balanced_outcomes)
    coverage = {
        "candidate_universe": sorted(all_candidates),
        "n_candidates_in_universe": len(all_candidates),
        "n_candidates_observed_in_pair_rows": len(candidates),
        "n_candidates_with_consistent_pair_outcome": len({candidate for outcome in pair_outcomes for candidate in outcome.get("candidates", [])}),
        "n_pair_outcomes": len(pair_outcomes),
        "n_connected_components": len(components),
        "connected_components": components,
        "global_ranking_available": global_available,
        "global_ranking_reason": None if global_available else "pairwise graph is disconnected or does not cover the candidate universe; report selected pair outcomes and component rankings only",
    }
    balanced_coverage = {
        "n_connected_components": len(balanced_components),
        "connected_components": balanced_components,
        "global_ranking_available": balanced_global_available,
        "global_ranking_reason": None if balanced_global_available else "family-balanced pairwise graph is disconnected or incomplete; no global family-balanced ranking",
    }
    component_rankings = _component_rankings(scores, components)
    balanced_component_rankings = _component_rankings(balanced_scores, balanced_components)
    return {"status": "ok" if pair_outcomes else "unavailable", "swap_consistency": _ratio(counts["consistent"], counts["complete"]), "swap_counts": counts, "position_sensitive_pairs_excluded": counts["position_sensitive"], "repeat_conflicts": repeat_conflicts, "pair_outcomes": pair_outcomes, "copeland_scores": scores, "ranking": ranking if global_available else None, "global_ranking": ranking if global_available else unavailable(coverage["global_ranking_reason"]), "component_rankings": component_rankings, "coverage": coverage, "family_balanced_pair_outcomes": family_balanced_outcomes, "family_balanced_copeland_scores": balanced_scores, "family_balanced_ranking": balanced_ranking if balanced_global_available else None, "family_balanced_global_ranking": balanced_ranking if balanced_global_available else unavailable(balanced_coverage["global_ranking_reason"]), "family_balanced_component_rankings": balanced_component_rankings, "family_balanced_coverage": balanced_coverage, "family_balanced_weighting": "each family 1/F; within-family judge votes 1/n_f; only descriptive sensitivity with missing family labels excluded", "note": "AB and BA winners are mapped back to canonical candidate IDs; position-sensitive pairs abstain from Copeland; disconnected graphs are reported per component, never as a global candidate ranking"}


def _candidate_rows(rows: Sequence[dict[str, Any]], candidate_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("_candidate_id") == candidate_id]


def family_balanced_mean(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean with family weight 1/F and judge weight 1/n_f.

    A judge first contributes its mean over the available valid units.  Each
    family then contributes the mean of its judges, so a family with two
    judges cannot receive twice the weight of a family with one.  This is a
    sensitivity/primary descriptive weighting, not a correction for truth or
    a substitute for a crossed hierarchical model.
    """

    family_judge_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("_status") != "ok" or row.get("_score") is None or row.get("_judge_id") is None:
            continue
        family = row.get("_family")
        if not family:
            return unavailable("family metadata missing; family-balanced weighting is not identifiable")
        family_judge_values[str(family)][str(row["_judge_id"])].append(float(row["_score"]))
    if not family_judge_values:
        return unavailable("no valid scores")
    family_means: dict[str, float] = {}
    judge_counts: dict[str, int] = {}
    for family, judge_values in family_judge_values.items():
        judge_means = [statistics.fmean(values) for values in judge_values.values() if values]
        if judge_means:
            family_means[family] = statistics.fmean(judge_means)
            judge_counts[family] = len(judge_means)
    if not family_means:
        return unavailable("no family has a valid judge score")
    return {"status": "ok", "estimate": statistics.fmean(list(family_means.values())), "n_families": len(family_means), "family_means": family_means, "judges_per_family": judge_counts, "weighting": "each family 1/F; within family each distinct judge 1/n_f"}


def _ranking_from_rows(rows: Sequence[dict[str, Any]], exclude_family: str | None = None) -> list[str]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("_status") != "ok" or row.get("_score") is None or row.get("_candidate_id") is None:
            continue
        if exclude_family is not None and row.get("_family") == exclude_family:
            continue
        values[str(row["_candidate_id"])].append(float(row["_score"]))
    return sorted(values, key=lambda candidate: (-statistics.fmean(values[candidate]), candidate))


def family_leave_one_out(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if any(row.get("_status") == "ok" and row.get("_score") is not None and not row.get("_family") for row in rows):
        return unavailable("family metadata missing for at least one valid judge row; leave-one-family-out would be ambiguous")
    families = sorted({str(row.get("_family")) for row in rows if row.get("_family")})
    if len(families) < 2:
        return unavailable("need at least two distinct family labels")
    baseline = _ranking_from_rows(rows)
    omissions: dict[str, Any] = {}
    for family in families:
        ranking = _ranking_from_rows(rows, family)
        positions = {candidate: index + 1 for index, candidate in enumerate(baseline)}
        omitted_positions = {candidate: index + 1 for index, candidate in enumerate(ranking)}
        omissions[family] = {"ranking": ranking, "top1": ranking[0] if ranking else None, "top1_changed": bool(ranking and baseline and ranking[0] != baseline[0]), "rank_delta_from_baseline": {candidate: omitted_positions.get(candidate) - positions[candidate] for candidate in omitted_positions if candidate in positions}}
    return {"status": "ok", "baseline_ranking": baseline, "leave_one_family_out": omissions, "n_families": len(families), "note": "family omission is a sensitivity analysis; it is not a family-cluster robust CI"}


def _missingness(rows: Sequence[dict[str, Any]], candidate_id: str, observed_judges: Sequence[str] | None = None) -> dict[str, Any]:
    candidate_rows = _candidate_rows(rows, candidate_id)
    status_counts = Counter(row["_status"] for row in candidate_rows)
    logical_units = {(row["_item_id"], row["_judge_id"]) for row in candidate_rows}
    ok_judges = sorted({str(row["_judge_id"]) for row in candidate_rows if row["_status"] == "ok" and row["_judge_id"] is not None})
    return {"status_counts": dict(sorted(status_counts.items())), "n_rows": len(candidate_rows), "n_ok_rows": status_counts.get("ok", 0), "n_failed_rows": status_counts.get("failed", 0), "n_missing_rows": status_counts.get("missing", 0), "n_distinct_ok_judges": len(ok_judges), "ok_judges": ok_judges, "logical_item_judge_units": len(logical_units), "expected_judges": list(observed_judges or []), "note": "missing/failed observations are not silently treated as agreement or zero scores"}


def _qualification_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    qualification = [row for row in rows if row["_track"] == "qualification"]
    if not qualification:
        return {"status": "unavailable", "reason": "no qualification observations imported", "n_records": 0}
    by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_counts: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    valid_scalar_scores: list[float] = []
    for row in qualification:
        by_judge[str(row.get("_judge_id"))].append(row)
        audit = row.get("_qualification_score_audit")
        if not isinstance(audit, Mapping):
            audit = _qualification_score_audit(row)
        audit_status = str(audit.get("status", "missing"))
        audit_counts[audit_status] += 1
        if audit_status == "valid" and row.get("_score") is not None:
            valid_scalar_scores.append(float(row["_score"]))
        elif audit_status == "invalid" and len(invalid_examples) < 20:
            invalid_examples.append({
                "item_id": row.get("_item_id"),
                "candidate_id": row.get("_candidate_id"),
                "judge_id": row.get("_judge_id"),
                "run_id": row.get("_run_id"),
                "score_0_100": audit.get("score_0_100"),
                "subscores": audit.get("subscores"),
                "subscores_sum": audit.get("subscores_sum"),
                "reason": audit.get("reason"),
            })
    score_audit = {
        "n_variants": len(qualification),
        "status_counts": dict(sorted(audit_counts.items())),
        "valid": audit_counts.get("valid", 0),
        "invalid": audit_counts.get("invalid", 0),
        "missing": audit_counts.get("missing", 0),
        "abstain": audit_counts.get("abstain", 0),
        "invalid_examples": invalid_examples,
        "scalar_score_aggregation": {
            "status": "partial_validity" if audit_counts.get("invalid", 0) or audit_counts.get("missing", 0) else "ok",
            "n_valid": len(valid_scalar_scores),
            "n_excluded": len(qualification) - len(valid_scalar_scores),
            "mean_valid_only": _mean(valid_scalar_scores),
            "note": "Only variants whose score_0_100 equals the sum of all four subscores are included; raw invalid scores remain in imported rows. Error/evidence/critical fields are assessed separately.",
        },
    }
    judge_summaries: dict[str, dict[str, Any]] = {}
    for judge, values in sorted(by_judge.items()):
        statuses: Counter[str] = Counter()
        for row in values:
            audit = row.get("_qualification_score_audit")
            if not isinstance(audit, Mapping):
                audit = _qualification_score_audit(row)
            statuses[str(audit.get("status", "missing"))] += 1
        judge_summaries[judge] = {"n_records": len(values), "status_counts": dict(Counter(row["_status"] for row in values)), "score_audit": dict(sorted(statuses.items()))}
    return {"status": "imported_not_scored_without_hidden_anchor_key", "n_records": len(qualification), "judges": judge_summaries, "score_audit": score_audit, "note": "Qualification pass/fail requires the hidden known-error key; this aggregate does not infer it from judge output. Scalar qualification scores with invalid sums are excluded from scalar summaries; MQM/error/evidence/critical fields remain available."}


def aggregate(rows: Sequence[dict[str, Any]], metadata: Mapping[str, Any] | None = None, required_judges: int = DEFAULT_REQUIRED_JUDGES, bootstrap_replicates: int = 2000, seed: int = 20261022) -> dict[str, Any]:
    metadata = metadata or {}
    normalized = [row if "_track" in row else _normalize_row(row, metadata) for row in rows]
    pointwise_raw = [row for row in normalized if row["_track"] == "pointwise"]
    pairwise_raw = [row for row in normalized if row["_track"] == "pairwise"]
    collapsed, collapse_audit = _collapse_pointwise(pointwise_raw)
    observed_judges = sorted({str(row["_judge_id"]) for row in collapsed if row.get("_judge_id") is not None})
    observed_families = sorted({str(row.get("_family")) for row in collapsed if row.get("_family")})
    # Do not silently derive family independence from score correlation.  A
    # shared source difficulty term makes raw family correlation confounded.
    family_dependency = {"status": "not_estimated", "reason": "raw judge-score correlation is confounded by shared source difficulty; family independence requires source-centered residuals and enough source clusters", "n_source_clusters": len({row["_source_id"] for row in collapsed if row.get("_source_id")})}
    candidate_ids = sorted({str(row["_candidate_id"]) for row in collapsed if row.get("_candidate_id") is not None})
    pointwise_summary: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        candidate_rows = [row for row in collapsed if row["_candidate_id"] == candidate_id]
        valid_scores = [float(row["_score"]) for row in candidate_rows if row["_status"] == "ok" and row["_score"] is not None]
        judge_ids = sorted({str(row["_judge_id"]) for row in candidate_rows if row["_status"] == "ok" and row.get("_judge_id") is not None})
        _, matrix, matrix_units = _group_score_matrix(candidate_rows)
        score_by_source: dict[str, list[float]] = defaultdict(list)
        critical_by_source: dict[str, list[float]] = defaultdict(list)
        for row in candidate_rows:
            if row["_status"] == "ok" and row["_score"] is not None:
                score_by_source[str(row["_source_id"])].append(float(row["_score"]))
                if row.get("_critical") is not None:
                    critical_by_source[str(row["_source_id"])].append(float(bool(row["_critical"])))
        balanced_score_by_source: dict[str, list[float]] = {}
        for source_id in sorted(score_by_source):
            balanced = family_balanced_mean([row for row in candidate_rows if str(row.get("_source_id")) == source_id])
            if balanced.get("status") == "ok":
                balanced_score_by_source[source_id] = [float(balanced["estimate"])]
        critical_votes = {str(row["_item_id"]): [] for row in candidate_rows if row["_status"] == "ok"}
        for row in candidate_rows:
            if row["_status"] == "ok" and row.get("_critical") is not None:
                critical_votes.setdefault(str(row["_item_id"]), []).append((str(row["_judge_id"]), bool(row["_critical"])))
        critical_by_item = {item: {judge for judge, flag in values if flag} for item, values in critical_votes.items()}
        deterministic_items = sorted({str(row["_item_id"]) for row in candidate_rows if row.get("_deterministic_critical")})
        n_judges = len(judge_ids)
        missing = _missingness(candidate_rows, candidate_id, observed_judges)
        if not valid_scores:
            status = "NO_VALID_JUDGMENTS"
        elif n_judges < required_judges:
            status = "INSUFFICIENT_PANEL"
        elif deterministic_items or any(len(v) >= 2 for v in critical_by_item.values()):
            status = "FAIL_FOR_DEPLOYMENT"
        elif any(len(v) == 1 for v in critical_by_item.values()) or missing["n_failed_rows"] or missing["n_missing_rows"]:
            status = "REVIEW_REQUIRED"
        elif (_median(valid_scores) or 0) < 80:
            status = "FAIL_FOR_DEPLOYMENT"
        elif (_median(valid_scores) or 0) < 90 or (_iqr(valid_scores) or 0) > 10:
            status = "REVIEW_REQUIRED"
        else:
            status = "AUTO_PASS_FOR_BENCHMARK"
        pointwise_summary[candidate_id] = {
            "n_items": len({row["_item_id"] for row in candidate_rows}),
            "n_valid_scores": len(valid_scores),
            "n_distinct_judges": n_judges,
            "score_median": _median(valid_scores),
            "score_mean": _mean(valid_scores),
            "score_iqr": _iqr(valid_scores),
            "score_mean_family_balanced": family_balanced_mean(candidate_rows),
            "score_mean_family_balanced_source_bootstrap_ci": source_cluster_bootstrap(balanced_score_by_source, "mean", bootstrap_replicates, seed + 2),
            "score_mean_source_bootstrap_ci": source_cluster_bootstrap(score_by_source, "mean", bootstrap_replicates, seed),
            "critical_rate_source_bootstrap_ci": source_cluster_bootstrap(critical_by_source, "mean", bootstrap_replicates, seed + 1),
            "critical_vote_count_distinct_judges": sum(len(judges) for judges in critical_by_item.values()),
            "critical_items": sorted(item for item, judges in critical_by_item.items() if judges),
            "critical_judges_by_item": {item: sorted(judges) for item, judges in sorted(critical_by_item.items()) if judges},
            "deterministic_critical_items": deterministic_items,
            "icc_2_1_2_k": icc_two_way_random_absolute(matrix),
            "nominal_error_type_agreement": nominal_agreement(candidate_rows, "type"),
            "critical_agreement": binary_agreement(candidate_rows),
            "krippendorff_alpha_nominal_error_type": krippendorff_alpha(_unit_label_matrix(candidate_rows, "type"), "nominal"),
            "krippendorff_alpha_ordinal_severity": krippendorff_alpha(_unit_label_matrix(candidate_rows, "severity"), "ordinal", [0, 1, 2, 3]),
            "gwet_ac1_critical": gwet_ac(_unit_label_matrix(candidate_rows, "critical"), "nominal", [False, True]),
            "gwet_ac2_severity": gwet_ac(_unit_label_matrix(candidate_rows, "severity"), "ordinal", [0, 1, 2, 3], "quadratic"),
            "mqm_consensus": _span_consensus(candidate_rows),
            "span_agreement": span_agreement(candidate_rows),
            "missingness": missing,
            "deployment_status": status,
            "formal_panel_eligible": n_judges >= required_judges,
            "matrix_units_for_icc": [list(unit) for unit in matrix_units],
        }
    _, overall_matrix, overall_units = _group_score_matrix(collapsed)
    source_ids = sorted({str(row["_source_id"]) for row in collapsed if row.get("_source_id")})
    pairwise_summary = pairwise_abba_copeland(pairwise_raw, candidate_ids) if pairwise_raw else {"status": "unavailable", "reason": "no pairwise observations imported"}
    pairwise_fresh_chat = _nested(metadata, ("metadata", "pairwise_fresh_chat_independence"), ("pairwise_fresh_chat_independence",), default=None)
    if isinstance(pairwise_fresh_chat, list):
        pairwise_fresh_chat = False if any(value is False for value in pairwise_fresh_chat) else (True if pairwise_fresh_chat and all(value is True for value in pairwise_fresh_chat) else None)
    pairwise_context_limit = _nested(metadata, ("metadata", "pairwise_limitation"), ("pairwise_limitation",), default=None)
    pairwise_context = {"fresh_chat_independence": pairwise_fresh_chat, "limitation": pairwise_context_limit, "note": "AB/BA swap consistency is not an independent-chat estimate when the source bundle records shared context or prior exposure" if pairwise_fresh_chat is False else None}
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": metadata.get("packet_id"),
        "packet_sha256": metadata.get("packet_sha256"),
        "judge_scope": {"observed_judges": observed_judges, "n_distinct_judges": len(observed_judges), "observed_families": observed_families, "required_for_auto_pass": required_judges, "formal_panel": len(observed_judges) >= required_judges, "note": "Judge IDs are raters; run_id/repeat is not an additional judge. No automatic pass is emitted for a panel below the required count."},
        "sample_scope": {"n_source_clusters": len(source_ids), "source_ids": source_ids, "bootstrap_unit": "source_id", "n_pointwise_records_raw": len(pointwise_raw), "n_pointwise_units_collapsed": len(collapsed), "warning": "only 3 source clusters: percentile bootstrap intervals are descriptive and unstable/narrow; 21 translations from these sources are correlated" if len(source_ids) == 3 else ("few source clusters; interpret cautiously" if len(source_ids) < 5 else None)},
        "qualification": _qualification_summary(normalized),
        "pointwise": pointwise_summary,
        "reliability_overall": {"icc_2_1_2_k": icc_two_way_random_absolute(overall_matrix), "krippendorff_alpha_nominal_error_type": krippendorff_alpha(_unit_label_matrix(collapsed, "type"), "nominal"), "krippendorff_alpha_ordinal_severity": krippendorff_alpha(_unit_label_matrix(collapsed, "severity"), "ordinal", [0, 1, 2, 3]), "gwet_ac1_critical": gwet_ac(_unit_label_matrix(collapsed, "critical"), "nominal", [False, True]), "gwet_ac2_severity": gwet_ac(_unit_label_matrix(collapsed, "severity"), "ordinal", [0, 1, 2, 3], "quadratic"), "kendall_w": kendall_w(collapsed), "span_agreement": span_agreement(collapsed), "matrix_units_for_icc": [list(unit) for unit in overall_units], "scope_note": "Descriptive agreement/reliability over source-correlated units; ICC(2,k) is arithmetic mean only, not median."},
        "pairwise": pairwise_summary,
        "pairwise_context": pairwise_context,
        "family_sensitivity": {"weighting": "primary descriptive sensitivity: family 1/F, within-family judge 1/n_f, alongside unweighted rows", "family_balanced_ranking": sorted(candidate_ids, key=lambda candidate: -(pointwise_summary[candidate]["score_mean_family_balanced"].get("estimate") if pointwise_summary[candidate]["score_mean_family_balanced"].get("status") == "ok" else float("-inf"))), "leave_one_family_out": family_leave_one_out(collapsed), "hierarchical_crossed_model": {"status": "not_fit", "reason": "three source clusters and one judge per family in the current pilot do not identify crossed family/judge variance components; a proposed source×candidate×judge model requires a larger held-out calibration design"}},
        "mqm_consensus_overall": _span_consensus(collapsed),
        "missingness": {"raw_status_counts": dict(Counter(row["_status"] for row in normalized)), "collapse_audit": collapse_audit, "failed_or_missing_rows": [{"track": row["_track"], "item_id": row["_item_id"], "candidate_id": row["_candidate_id"], "judge_id": row["_judge_id"], "run_id": row["_run_id"], "status": row["_status"], "reason": row.get("reason")} for row in normalized if row["_status"] != "ok"]},
        "family_dependency": family_dependency,
        "limitations": ["Only validated judge imports in this aggregate are represented; unimported or quarantined runs are excluded rather than imputed.", "Local qualification and pointwise observations are separate tracks; qualification pass/fail is not inferred without the hidden key.", "A high agreement coefficient is reliability only, not translation correctness or clinical safety.", "UI-displayed model version is user-supplied metadata and is not verified by a prompt self-report; UI latency is not comparable to API timing."] + (["This source bundle records shared pairwise context rather than independent fresh AB/BA chats; treat swap consistency as context-contaminated descriptive evidence."] if pairwise_fresh_chat is False else []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="validated judge-result JSON bundle; repeat for independently imported judges")
    parser.add_argument("--output", required=True, help="aggregation JSON output")
    parser.add_argument("--required-judges", type=int, default=DEFAULT_REQUIRED_JUDGES)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    try:
        metadata, rows = load_bundles(args.input)
        result = aggregate(rows, metadata, args.required_judges, args.bootstrap_replicates, args.seed)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ImportError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
