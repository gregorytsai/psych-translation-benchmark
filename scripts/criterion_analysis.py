#!/usr/bin/env python3
"""Offline criterion/ranking analysis for the public translation benchmark.

This script intentionally uses only the public CSV exports and numpy.  It does
not call a model, access the website, or modify any benchmark input file.

The inferential unit is a source cluster (12 clusters), not a judge.  Judges
are averaged within each model/item first.  Exact paired sign-flip tests then
enumerate all 2**12 signs.  The script is deterministic and writes CSV, JSON,
and Traditional-Chinese Markdown outputs in verification/criteria by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


CRITERIA = [
    "completeness_format",
    "fluency",
    "semantic_safety",
    "terminology_locale",
]
CRITERION_ZH = {
    "completeness_format": "完整性／格式",
    "fluency": "流暢度",
    "semantic_safety": "語意／安全",
    "terminology_locale": "術語／臺灣在地化",
}
# The rubric's four weighted maxima sum to 100; a row's total is its earned
# points, not 100 for every row.
MAXIMA = {
    "completeness_format": 20.0,
    "fluency": 20.0,
    "semantic_safety": 30.0,
    "terminology_locale": 30.0,
}
MODEL_ORDER_HINT = [
    "gemma4-12b",
    "gemma4-26b",
    "gemma4-31b-qat",
    "gemma4-e4b",
    "translategemma-12b",
    "translategemma-27b",
    "translategemma-4b",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def number(value: str | int | float | None) -> float:
    if value is None or value == "":
        raise ValueError("empty numeric value")
    return float(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def tie_midrank(values: dict[str, float], descending: bool = True) -> dict[str, float]:
    """Competition ranks with tied values receiving the middle rank."""
    ordered = sorted(values.items(), key=lambda kv: (-kv[1] if descending else kv[1], kv[0]))
    out: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        rank = (i + 1 + j) / 2.0
        for key, _ in ordered[i:j]:
            out[key] = rank
        i = j
    return out


def median_item_ranks(item_values: dict[str, dict[str, float]], model_ids: list[str]) -> dict[str, float]:
    per_model: dict[str, list[float]] = defaultdict(list)
    for values in item_values.values():
        ranks = tie_midrank({m: values[m] for m in model_ids})
        for m in model_ids:
            per_model[m].append(ranks[m])
    return {m: float(median(per_model[m])) for m in model_ids}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p values in original order."""
    order = sorted(range(len(p_values)), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order, start=1):
        candidate = min(1.0, (m - rank + 1) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def exact_signflip(contributions: np.ndarray, signs: np.ndarray) -> tuple[float, float, int, float]:
    """Two-sided exact sign-flip p for a cluster contribution vector.

    All 4096 sign rows are retained, including duplicated permutations caused
    by zero contributions.  The statistic is the absolute sum.
    """
    obs = float(np.sum(contributions))
    perm_stats = signs @ contributions
    threshold = abs(obs) - 1e-12
    p = float(np.count_nonzero(np.abs(perm_stats) >= threshold) / signs.shape[0])
    zeros = int(np.count_nonzero(np.isclose(contributions, 0.0, rtol=0.0, atol=1e-12)))
    return obs, p, zeros, float(np.mean(np.abs(perm_stats)))


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 3) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |\n"
    head += "|" + "|".join("---" for _ in columns) + "|\n"
    body = []
    for row in rows:
        vals = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(value, digits))
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return head + "\n".join(body) + ("\n" if body else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "data",
        help="Public docs/data directory containing the two CSV inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("verification") / "criteria",
        help="Directory for generated analysis outputs.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pointwise_path = data_dir / "pointwise_results_long.csv"
    comparison_path = data_dir / "translation_comparison.csv"
    pointwise = read_csv(pointwise_path)
    comparison = read_csv(comparison_path)

    # The published comparison file is the authoritative common-case roster:
    # text-15 is intentionally excluded because its TranslateGemma-4B panel is
    # incomplete.  This also checks that an input refresh cannot silently
    # change the common-case definition.
    eligible_text_rows = [
        r for r in comparison
        if r.get("source_kind") == "text" and r.get("common_case_eligible") == "true"
    ]
    common_ids = sorted({r["material_id"] for r in eligible_text_rows})
    excluded_text_ids = sorted({r["material_id"] for r in comparison if r.get("source_kind") == "text"} - set(common_ids))
    if len(common_ids) != 29 or excluded_text_ids != ["text-15"]:
        raise RuntimeError(f"unexpected common text roster: {len(common_ids)} eligible, excluded={excluded_text_ids}")

    # Preserve the public model order where possible, then append any new ID in
    # stable lexical order so the script fails loudly only on actual schema
    # changes, not on CSV row order.
    model_names = {r["model_id"]: r.get("model_name", r["model_id"]) for r in comparison}
    model_ids_seen = set(model_names)
    model_ids = [m for m in MODEL_ORDER_HINT if m in model_ids_seen]
    model_ids.extend(sorted(model_ids_seen - set(model_ids)))
    if len(model_ids) != 7:
        raise RuntimeError(f"expected 7 public models, got {model_ids}")
    judge_slots = sorted({r["judge_slot"] for r in pointwise})
    if judge_slots != ["judge-01", "judge-02", "judge-03", "judge-04", "judge-05"]:
        raise RuntimeError(f"unexpected judge roster: {judge_slots}")

    comparison_by_item_model = {
        (r["material_id"], r["model_id"]): r for r in comparison
    }
    cluster_by_item: dict[str, str] = {}
    for material_id in common_ids:
        rows = [r for r in eligible_text_rows if r["material_id"] == material_id]
        clusters = {r.get("source_cluster", "") for r in rows}
        if len(clusters) != 1 or not next(iter(clusters)):
            raise RuntimeError(f"source cluster missing/ambiguous for {material_id}: {clusters}")
        cluster_by_item[material_id] = next(iter(clusters))
    clusters = sorted(set(cluster_by_item.values()))
    if len(clusters) != 12:
        raise RuntimeError(f"expected 12 source clusters, got {clusters}")
    cluster_items = {c: [i for i in common_ids if cluster_by_item[i] == c] for c in clusters}

    # Select exactly the 29*7*5 authoritative text cells.  Keep validation
    # counts separately so malformed rows are reported rather than silently
    # dropped.  There is no source_cluster column in pointwise_results_long;
    # the comparison file supplies that join.
    common_set = set(common_ids)
    text_rows = [r for r in pointwise if r.get("source_kind") == "text" and r.get("material_id") in common_set]
    expected_text_rows = len(common_ids) * len(model_ids) * len(judge_slots)
    if len(text_rows) != expected_text_rows:
        raise RuntimeError(f"expected {expected_text_rows} common text judge cells, got {len(text_rows)}")

    # Verify one cell per item/model/judge and preserve all raw rubric values.
    cell_counts = defaultdict(int)
    bad_duplicate_keys: list[tuple[str, str, str]] = []
    cell_values: dict[tuple[str, str, str], dict[str, float]] = {}
    score_values: dict[tuple[str, str, str], float] = {}
    invalid_rows: list[dict[str, str]] = []
    bound_violations: list[dict[str, Any]] = []
    score_sum_mismatches: list[dict[str, Any]] = []
    for row in text_rows:
        key = (row["material_id"], row["model_id"], row["judge_slot"])
        cell_counts[key] += 1
        if cell_counts[key] > 1:
            bad_duplicate_keys.append(key)
        try:
            vals = {c: number(row[c]) for c in CRITERIA}
            score = number(row["score_0_100"])
            cell_values[key] = vals
            score_values[key] = score
            if row.get("scalar_valid") != "true" or row.get("schema_valid") != "true" or row.get("validation_valid") != "true" or row.get("status") != "ok":
                invalid_rows.append({"material_id": row["material_id"], "model_id": row["model_id"], "judge_slot": row["judge_slot"], "status": row.get("status", "")})
            if any(vals[c] < 0 or vals[c] > MAXIMA[c] for c in CRITERIA):
                bound_violations.append({"material_id": row["material_id"], "model_id": row["model_id"], "judge_slot": row["judge_slot"], **vals})
            if not math.isclose(sum(vals.values()), score, rel_tol=0.0, abs_tol=1e-9):
                score_sum_mismatches.append({"material_id": row["material_id"], "model_id": row["model_id"], "judge_slot": row["judge_slot"], "rubric_sum": sum(vals.values()), "score_0_100": score})
        except (TypeError, ValueError, KeyError):
            invalid_rows.append({"material_id": row.get("material_id", ""), "model_id": row.get("model_id", ""), "judge_slot": row.get("judge_slot", ""), "status": "non_numeric_or_missing"})

    expected_keys = {(i, m, j) for i in common_ids for m in model_ids for j in judge_slots}
    missing_keys = sorted(expected_keys - set(cell_values))
    duplicate_count = len(bad_duplicate_keys)
    if missing_keys or duplicate_count or invalid_rows or bound_violations or score_sum_mismatches:
        raise RuntimeError(
            "common text validation failed: "
            f"missing={len(missing_keys)}, duplicates={duplicate_count}, invalid={len(invalid_rows)}, "
            f"bounds={len(bound_violations)}, score_sum={len(score_sum_mismatches)}"
        )

    # Judge averaging happens before item averaging.  This gives each of the 29
    # text items equal weight, while no judge is treated as an independent
    # inferential replicate.
    item_panel_raw: dict[tuple[str, str], dict[str, float]] = {}
    item_panel_norm: dict[tuple[str, str], dict[str, float]] = {}
    item_panel_total: dict[tuple[str, str], float] = {}
    for material_id in common_ids:
        for model_id in model_ids:
            vals_by_criterion: dict[str, float] = {}
            for criterion in CRITERIA:
                vals_by_criterion[criterion] = float(np.mean([
                    cell_values[(material_id, model_id, j)][criterion] for j in judge_slots
                ]))
            item_panel_raw[(material_id, model_id)] = vals_by_criterion
            item_panel_norm[(material_id, model_id)] = {c: vals_by_criterion[c] / MAXIMA[c] * 100.0 for c in CRITERIA}
            item_panel_total[(material_id, model_id)] = float(np.mean([
                score_values[(material_id, model_id, j)] for j in judge_slots
            ]))

    criterion_table: list[dict[str, Any]] = []
    criterion_rank_maps: dict[str, dict[str, float]] = {}
    criterion_median_rank_maps: dict[str, dict[str, float]] = {}
    for criterion in CRITERIA:
        raw_means = {
            m: float(np.mean([item_panel_raw[(i, m)][criterion] for i in common_ids])) for m in model_ids
        }
        norm_means = {m: raw_means[m] / MAXIMA[criterion] * 100.0 for m in model_ids}
        ranks = tie_midrank(norm_means)
        item_values = {i: {m: item_panel_norm[(i, m)][criterion] for m in model_ids} for i in common_ids}
        median_ranks = median_item_ranks(item_values, model_ids)
        criterion_rank_maps[criterion] = ranks
        criterion_median_rank_maps[criterion] = median_ranks
        for m in model_ids:
            criterion_table.append({
                "model_id": m,
                "model_name": model_names[m],
                "criterion": criterion,
                "criterion_zh": CRITERION_ZH[criterion],
                "raw_mean": raw_means[m],
                "raw_max": MAXIMA[criterion],
                "normalized_mean_100": norm_means[m],
                "aggregate_rank": ranks[m],
                "median_item_rank": median_ranks[m],
                "n_items": len(common_ids),
                "n_judges_averaged": len(judge_slots),
            })

    # Total score rankings by each judge, plus the pooled five-judge panel.
    judge_total_table: list[dict[str, Any]] = []
    for judge in judge_slots + ["all-5-panel"]:
        per_item: dict[str, dict[str, float]] = {}
        if judge == "all-5-panel":
            for item in common_ids:
                per_item[item] = {m: item_panel_total[(item, m)] for m in model_ids}
        else:
            for item in common_ids:
                per_item[item] = {m: score_values[(item, m, judge)] for m in model_ids}
        means = {m: float(np.mean([per_item[i][m] for i in common_ids])) for m in model_ids}
        ranks = tie_midrank(means)
        medians = median_item_ranks(per_item, model_ids)
        for m in model_ids:
            judge_total_table.append({
                "judge_slot": judge,
                "model_id": m,
                "model_name": model_names[m],
                "total_score_0_100_mean": means[m],
                "aggregate_rank": ranks[m],
                "median_item_rank": medians[m],
                "n_items": len(common_ids),
                "note": "5-judge panel mean first, then 29-item mean" if judge == "all-5-panel" else "one judge, 29-item mean",
            })

    # Exact 12-cluster paired sign-flip tests.  The cluster contribution is the
    # sum of item differences divided by 29 (not a mean of 12 cluster means),
    # preserving the main table's equal item weight.
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(clusters))), dtype=float)
    pair_records: list[dict[str, Any]] = []
    pair_contrib: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for a_index, model_a in enumerate(model_ids):
        for model_b in model_ids[a_index + 1:]:
            for criterion in CRITERIA:
                contribution = np.asarray([
                    sum(item_panel_norm[(item, model_a)][criterion] - item_panel_norm[(item, model_b)][criterion] for item in cluster_items[c]) / len(common_ids)
                    for c in clusters
                ], dtype=float)
                pair_contrib[(model_a, model_b, criterion)] = {"vector": contribution}
                obs, p_raw, zero_count, perm_abs_mean = exact_signflip(contribution, signs)
                pair_records.append({
                    "model_a": model_a,
                    "model_a_name": model_names[model_a],
                    "model_b": model_b,
                    "model_b_name": model_names[model_b],
                    "criterion": criterion,
                    "criterion_zh": CRITERION_ZH[criterion],
                    "difference_a_minus_b_normalized100": obs,
                    "p_raw_exact_two_sided": p_raw,
                    "holm84_p": None,
                    "holm84_reject_alpha05": None,
                    "n_clusters": len(clusters),
                    "n_permutations": int(signs.shape[0]),
                    "zero_difference_clusters": zero_count,
                    "permutation_abs_stat_mean": perm_abs_mean,
                })
    pair_adjusted = holm_adjust([float(r["p_raw_exact_two_sided"]) for r in pair_records])
    for record, adjusted in zip(pair_records, pair_adjusted):
        record["holm84_p"] = adjusted
        record["holm84_reject_alpha05"] = bool(adjusted <= 0.05)

    # Exploratory global interaction.  Use normalized-to-100 criterion panels
    # so each rubric dimension has the same 0-100 display scale.  For each item
    # double-center the 7x4 model-by-criterion panel, sum by source cluster,
    # divide by 29, then use the sum of squared interaction entries as the
    # statistic.  (Using a mean instead would only multiply T by 28 and leave
    # the exact p unchanged; the reported T follows the independent QA table.)
    interaction_item: dict[str, np.ndarray] = {}
    for item in common_ids:
        matrix = np.asarray([[item_panel_norm[(item, m)][c] for c in CRITERIA] for m in model_ids], dtype=float)
        interaction_item[item] = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + matrix.mean()
    interaction_cluster = np.asarray([
        np.sum([interaction_item[item] for item in cluster_items[c]], axis=0) / len(common_ids) for c in clusters
    ], dtype=float)
    interaction_observed_matrix = interaction_cluster.sum(axis=0)
    interaction_observed_stat = float(np.sum(interaction_observed_matrix ** 2))
    interaction_perm = signs @ interaction_cluster.reshape(len(clusters), -1)
    interaction_perm_stat = np.sum(interaction_perm ** 2, axis=1)
    interaction_p = float(np.count_nonzero(interaction_perm_stat >= interaction_observed_stat - 1e-12) / signs.shape[0])

    # Post-hoc rank sensitivity: for each item and criterion, replace the
    # panel means with 1=best midranks (ties receive the average rank), then
    # apply the same model-by-criterion double-centering and cluster sign-flip.
    # This directly probes whether the ordering pattern changes by criterion;
    # it is not a p value for any particular pairwise rank switch.
    rank_interaction_item: dict[str, np.ndarray] = {}
    for item in common_ids:
        rank_matrix = np.asarray([
            [tie_midrank({m: item_panel_norm[(item, m)][criterion] for m in model_ids}, descending=False)[model] for criterion in CRITERIA]
            for model in model_ids
        ], dtype=float)
        rank_interaction_item[item] = rank_matrix - rank_matrix.mean(axis=1, keepdims=True) - rank_matrix.mean(axis=0, keepdims=True) + rank_matrix.mean()
    rank_interaction_cluster = np.asarray([
        np.sum([rank_interaction_item[item] for item in cluster_items[c]], axis=0) / len(common_ids) for c in clusters
    ], dtype=float)
    rank_interaction_observed_matrix = rank_interaction_cluster.sum(axis=0)
    rank_interaction_observed_stat = float(np.sum(rank_interaction_observed_matrix ** 2))
    rank_interaction_perm = signs @ rank_interaction_cluster.reshape(len(clusters), -1)
    rank_interaction_perm_stat = np.sum(rank_interaction_perm ** 2, axis=1)
    rank_interaction_p = float(np.count_nonzero(rank_interaction_perm_stat >= rank_interaction_observed_stat - 1e-12) / signs.shape[0])

    # 21 model pairs x 6 unordered criterion pairs.  A reverse is descriptive
    # evidence that the model ordering flips between the two normalized
    # criteria; inferential evidence additionally requires Holm-126 <= .05.
    criterion_pair_records: list[dict[str, Any]] = []
    criterion_pairs = list(itertools.combinations(CRITERIA, 2))
    for a_index, model_a in enumerate(model_ids):
        for model_b in model_ids[a_index + 1:]:
            for c1, c2 in criterion_pairs:
                contribution = np.asarray([
                    sum(
                        (item_panel_norm[(item, model_a)][c1] - item_panel_norm[(item, model_b)][c1])
                        - (item_panel_norm[(item, model_a)][c2] - item_panel_norm[(item, model_b)][c2])
                        for item in cluster_items[c]
                    ) / len(common_ids)
                    for c in clusters
                ], dtype=float)
                diff_c1 = float(sum(
                    item_panel_norm[(item, model_a)][c1] - item_panel_norm[(item, model_b)][c1]
                    for item in common_ids
                ) / len(common_ids))
                diff_c2 = float(sum(
                    item_panel_norm[(item, model_a)][c2] - item_panel_norm[(item, model_b)][c2]
                    for item in common_ids
                ) / len(common_ids))
                reverse = bool(diff_c1 * diff_c2 < -1e-12)
                obs, p_raw, zero_count, perm_abs_mean = exact_signflip(contribution, signs)
                criterion_pair_records.append({
                    "model_a": model_a,
                    "model_a_name": model_names[model_a],
                    "model_b": model_b,
                    "model_b_name": model_names[model_b],
                    "criterion_1": c1,
                    "criterion_1_zh": CRITERION_ZH[c1],
                    "criterion_2": c2,
                    "criterion_2_zh": CRITERION_ZH[c2],
                    "difference_c1_a_minus_b_normalized100": diff_c1,
                    "difference_c2_a_minus_b_normalized100": diff_c2,
                    "difference_of_differences_c1_minus_c2": obs,
                    "descriptive_reverse_sign": reverse,
                    "p_raw_exact_two_sided": p_raw,
                    "holm126_p": None,
                    "holm126_reject_alpha05": None,
                    "n_clusters": len(clusters),
                    "n_permutations": int(signs.shape[0]),
                    "zero_difference_clusters": zero_count,
                    "permutation_abs_stat_mean": perm_abs_mean,
                })
    criterion_pair_adjusted = holm_adjust([float(r["p_raw_exact_two_sided"]) for r in criterion_pair_records])
    for record, adjusted in zip(criterion_pair_records, criterion_pair_adjusted):
        record["holm126_p"] = adjusted
        record["holm126_reject_alpha05"] = bool(adjusted <= 0.05)

    # Image scores are descriptive only.  Images have four evaluable judges;
    # judge-03's 28 unsupported_by_design rows are deliberately retained in the
    # validation counts and excluded from these means.
    image_rows = [r for r in pointwise if r.get("source_kind") == "image"]
    image_valid_rows = [r for r in image_rows if r.get("scalar_valid") == "true" and r.get("status") == "ok"]
    image_ids = sorted({r["material_id"] for r in image_valid_rows})
    image_judges = sorted({r["judge_slot"] for r in image_valid_rows})
    image_item_model_judge: dict[tuple[str, str, str], float] = {}
    for r in image_valid_rows:
        image_item_model_judge[(r["material_id"], r["model_id"], r["judge_slot"])] = number(r["score_0_100"])
    image_panel: dict[tuple[str, str], float] = {}
    for item in image_ids:
        for model in model_ids:
            values = [image_item_model_judge[(item, model, judge)] for judge in image_judges]
            image_panel[(item, model)] = float(np.mean(values))
    image_mean = {m: float(np.mean([image_panel[(i, m)] for i in image_ids])) for m in model_ids}
    image_rank = tie_midrank(image_mean)
    image_item_rank = {
        item: tie_midrank({m: image_panel[(item, m)] for m in model_ids}) for item in image_ids
    }
    image_table: list[dict[str, Any]] = []
    for m in model_ids:
        image_table.append({
            "model_id": m,
            "model_name": model_names[m],
            "image_panel_mean_0_100": image_mean[m],
            "aggregate_rank": image_rank[m],
            "median_image_rank": float(median([image_item_rank[item][m] for item in image_ids])),
            "n_images": len(image_ids),
            "n_evaluable_judges": len(image_judges),
        })
    for item in image_ids:
        for m in model_ids:
            image_table.append({
                "model_id": m,
                "model_name": model_names[m],
                "image_id": item,
                "image_panel_mean_0_100": image_panel[(item, m)],
                "aggregate_rank": image_item_rank[item][m],
                "median_image_rank": "",
                "n_images": 1,
                "n_evaluable_judges": len(image_judges),
            })

    criterion_csv_fields = [
        "model_id", "model_name", "criterion", "criterion_zh", "raw_mean", "raw_max",
        "normalized_mean_100", "aggregate_rank", "median_item_rank", "n_items", "n_judges_averaged",
    ]
    judge_csv_fields = [
        "judge_slot", "model_id", "model_name", "total_score_0_100_mean", "aggregate_rank",
        "median_item_rank", "n_items", "note",
    ]
    pair_csv_fields = list(pair_records[0].keys())
    criterion_pair_csv_fields = list(criterion_pair_records[0].keys())
    image_csv_fields = [
        "model_id", "model_name", "image_id", "image_panel_mean_0_100", "aggregate_rank",
        "median_image_rank", "n_images", "n_evaluable_judges",
    ]
    write_csv(output_dir / "criterion_rank_table.csv", criterion_table, criterion_csv_fields)
    write_csv(output_dir / "judge_total_ranks.csv", judge_total_table, judge_csv_fields)
    write_csv(output_dir / "pairwise_tests_84.csv", pair_records, pair_csv_fields)
    write_csv(output_dir / "criterion_pair_tests_126.csv", criterion_pair_records, criterion_pair_csv_fields)
    write_csv(output_dir / "image_rank_table.csv", image_table, image_csv_fields)
    write_csv(output_dir / "source_cluster_items.csv", [
        {"source_cluster": c, "n_items": len(cluster_items[c]), "material_ids": ",".join(cluster_items[c])} for c in clusters
    ], ["source_cluster", "n_items", "material_ids"])

    # Compact model x criterion quick table for readers who want one row per
    # model, without replacing the long criterion table above.
    quick_rows: list[dict[str, Any]] = []
    pooled_total_rows = [r for r in judge_total_table if r["judge_slot"] == "all-5-panel"]
    pooled_total_by_model = {r["model_id"]: r for r in pooled_total_rows}
    for m in model_ids:
        row: dict[str, Any] = {"model_id": m, "model_name": model_names[m]}
        for c in CRITERIA:
            rec = next(r for r in criterion_table if r["model_id"] == m and r["criterion"] == c)
            short = c.replace("_", "")
            row[f"{c}_raw"] = rec["raw_mean"]
            row[f"{c}_norm100"] = rec["normalized_mean_100"]
            row[f"{c}_rank"] = rec["aggregate_rank"]
            row[f"{c}_median_item_rank"] = rec["median_item_rank"]
        row["total_score_0_100_mean"] = pooled_total_by_model[m]["total_score_0_100_mean"]
        row["total_score_rank"] = pooled_total_by_model[m]["aggregate_rank"]
        quick_rows.append(row)
    quick_fields = list(quick_rows[0].keys())
    write_csv(output_dir / "quick_rank_table.csv", quick_rows, quick_fields)

    significant_pair = [r for r in pair_records if r["holm84_reject_alpha05"]]
    reverse_rows = [r for r in criterion_pair_records if r["descriptive_reverse_sign"]]
    significant_reverse = [r for r in reverse_rows if r["holm126_reject_alpha05"]]
    min_pair_raw = min(float(r["p_raw_exact_two_sided"]) for r in pair_records)
    min_pair_holm = min(float(r["holm84_p"]) for r in pair_records)
    min_criterion_pair_raw = min(float(r["p_raw_exact_two_sided"]) for r in criterion_pair_records)
    min_criterion_pair_holm = min(float(r["holm126_p"]) for r in criterion_pair_records)
    result_json = {
        "schema": "criterion-analysis-20261027-v1",
        "run": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": "criterion_analysis.py",
            "data_dir": "psych-translation-benchmark/docs/data",
            "input_files": ["pointwise_results_long.csv", "translation_comparison.csv"],
            "offline_no_model_calls": True,
            "website_modified": False,
        },
        "sample": {
            "common_text_items": len(common_ids),
            "excluded_text_ids": excluded_text_ids,
            "models": [{"model_id": m, "model_name": model_names[m]} for m in model_ids],
            "judges": judge_slots,
            "source_clusters": clusters,
            "source_cluster_item_counts": {c: len(cluster_items[c]) for c in clusters},
            "common_text_cells": len(text_rows),
        },
        "score_schema": {
            "criterion_order": CRITERIA,
            "criterion_zh": CRITERION_ZH,
            "criterion_raw_maxima": MAXIMA,
            "raw_maxima_sum": sum(MAXIMA.values()),
            "normalization": "criterion normalized_mean_100 = raw mean / criterion raw maximum * 100",
            "row_total": "score_0_100 equals the sum of four earned raw criterion points; it is not expected to equal 100 for every row",
            "validation": {
                "expected_common_text_cells": expected_text_rows,
                "observed_common_text_cells": len(text_rows),
                "all_scalar_schema_validation_status_ok": True,
                "bounds_violations": len(bound_violations),
                "score_equals_sum_rows": len(text_rows) - len(score_sum_mismatches),
                "score_sum_mismatches": len(score_sum_mismatches),
                "score_min": min(score_values.values()),
                "score_max": max(score_values.values()),
                "score_100_rows": sum(math.isclose(v, 100.0, abs_tol=1e-12) for v in score_values.values()),
            },
        },
        "criterion_rank_table": criterion_table,
        "judge_total_rank_table": judge_total_table,
        "pairwise_tests": {
            "n_tests": len(pair_records),
            "correction": "Holm across 84 tests (21 model pairs x 4 criteria)",
            "n_holm_reject_alpha05": len(significant_pair),
            "minimum_observed_raw_p": min_pair_raw,
            "minimum_observed_holm_p": min_pair_holm,
            "theoretical_two_sided_signflip_p_resolution": 2.0 / signs.shape[0],
            "records": pair_records,
        },
        "global_interaction": {
            "criterion_scale": "normalized100",
            "construction": "per-item 7x4 panel double-centered across model and criterion; item interaction matrices summed within source cluster and divided by 29",
            "statistic": "sum of squared entries of the aggregate interaction matrix",
            "observed_statistic": interaction_observed_stat,
            "p_raw_exact_two_sided": interaction_p,
            "n_clusters": len(clusters),
            "n_permutations": int(signs.shape[0]),
            "zero_difference_clusters_in_matrix": int(np.count_nonzero(np.isclose(interaction_cluster.reshape(len(clusters), -1), 0.0, atol=1e-12))),
            "interpretation": "one exploratory global interaction test; this is not a p value for discrete rank changes",
        },
        "rank_sensitivity": {
            "criterion_scale": "per-item panel means converted to 1=best midranks; ties use average rank",
            "construction": "per-item 7x4 rank panel double-centered across model and criterion; rank interaction matrices summed within source cluster and divided by 29",
            "statistic": "sum of squared entries of the aggregate rank-interaction matrix",
            "observed_statistic": rank_interaction_observed_stat,
            "p_raw_exact_two_sided": rank_interaction_p,
            "n_clusters": len(clusters),
            "n_permutations": int(signs.shape[0]),
            "interpretation": "post-hoc exploratory ranking sensitivity; this is not a p value for any individual rank switch",
        },
        "global_two_test_sensitivity": {
            "tests": ["normalized100_interaction", "rank_interaction_posthoc"],
            "raw_p_values": [interaction_p, rank_interaction_p],
            "holm2_adjusted_p_values": holm_adjust([interaction_p, rank_interaction_p]),
            "note": "Optional sensitivity only; the primary report treats normalized interaction as one global exploratory test and rank interaction as post-hoc.",
        },
        "criterion_pair_tests": {
            "n_tests": len(criterion_pair_records),
            "correction": "Holm across 126 tests (21 model pairs x 6 criterion pairs)",
            "n_descriptive_reverse_sign": len(reverse_rows),
            "n_holm_reject_alpha05": len(significant_reverse),
            "minimum_observed_raw_p": min_criterion_pair_raw,
            "minimum_observed_holm_p": min_criterion_pair_holm,
            "theoretical_two_sided_signflip_p_resolution": 2.0 / signs.shape[0],
            "minimum_p_holm_first_step_if_theoretical_min": (2.0 / signs.shape[0]) * 126,
            "records": criterion_pair_records,
        },
        "images": {
            "image_ids": image_ids,
            "evaluable_judges": image_judges,
            "valid_rows": len(image_valid_rows),
            "unsupported_by_design_rows": len(image_rows) - len(image_valid_rows),
            "expected_evaluable_cells": len(image_ids) * len(model_ids) * len(image_judges),
            "interpretation": "descriptive ranks only; four images are too few for a comparable cluster significance claim",
            "aggregate_rows": [r for r in image_table if "image_id" not in r],
        },
        "assumptions_and_limits": [
            "The five judges are averaged within each model/item and are not five independent inferential samples.",
            "Exact sign-flip tests require exchangeability/symmetry of paired source-cluster differences under the null; with 12 clusters this is a small, descriptive benchmark inference.",
            "Source clusters are the paired units; cluster contributions use sum(item differences)/29 so the main item-level weighting is preserved.",
            "Holm controls the family-wise error rate for the stated 84 and 126 exploratory families only; it does not validate the rubric or establish clinical safety.",
            "Images retain unsupported_by_design rows and are descriptive only; no image significance test is claimed.",
            "Results do not establish clinical safety, universal best model, causality, or population generalization; there is no human gold standard in this analysis.",
        ],
    }
    with (output_dir / "criterion_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(json_safe(result_json), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # Traditional-Chinese report, starting with the quick rank table.
    quick_md_rows = []
    for row in quick_rows:
        quick_md_rows.append({
            "model_name": row["model_name"],
            "completeness_format": f"{row['completeness_format_norm100']:.2f} ({row['completeness_format_rank']:.1f})",
            "fluency": f"{row['fluency_norm100']:.2f} ({row['fluency_rank']:.1f})",
            "semantic_safety": f"{row['semantic_safety_norm100']:.2f} ({row['semantic_safety_rank']:.1f})",
            "terminology_locale": f"{row['terminology_locale_norm100']:.2f} ({row['terminology_locale_rank']:.1f})",
            "total": f"{row['total_score_0_100_mean']:.3f} ({row['total_score_rank']:.1f})",
        })
    md: list[str] = []
    md.append("# 不同評分標準下的模型排名：離線核對與成對檢定\n")
    md.append("本報告只讀公開 CSV、未呼叫模型、未修改網站或發布內容。分析資料為 29 個共同文字題（明確排除 text-15）、7 模型、5 評審，共 1,015 個文字評分 cell；文字題按 12 個 source cluster 配對。\n")
    md.append("## 先看快速排名表\n")
    md.append("括號為該標準的整體名次；前四欄是各分項轉成 0–100 的均值，最後是五評審總分（`score_0_100`）均值。這是描述性排名，並不等同於顯著差異。\n")
    md.append(markdown_table(quick_md_rows, [
        ("model_name", "模型"),
        ("completeness_format", "完整性／格式"),
        ("fluency", "流暢度"),
        ("semantic_safety", "語意／安全"),
        ("terminology_locale", "術語／在地化"),
        ("total", "總分 0–100"),
    ], digits=2))
    md.append("\n## 分數核對\n")
    md.append(f"四個原始量尺上限為完整性／格式 20、流暢度 20、語意／安全 30、術語／在地化 30，權重上限合計 100。1,015/1,015 rows 的四項原始分數均在上限內，且 `score_0_100` 均等於四項得分相加；因此「加總 100」是量尺上限合計，不是每一筆實得分都必須為 100。本共同文字集的總分範圍為 {min(score_values.values()):.0f}–{max(score_values.values()):.0f}。\n")
    md.append("每個模型先對同一題的五位評審取平均，再對 29 題等權平均；`median_item_rank` 是逐題 midrank 的中位數。完整原始均值與名次見 `criterion_rank_table.csv`。\n")
    md.append("\n## 文字分項的成對顯著性\n")
    md.append(f"對每一個分項的 21 個模型對共 84 tests，先形成每個 source cluster 的差向量 `sum(該 cluster 題目差異)/29`，再以 12 cluster 的 2^12 = 4,096 個 sign-flip 作雙尾 exact test；零差 cluster 保留在 4,096 個排列中。Holm 校正後 alpha=.05 有 {len(significant_pair)}/{len(pair_records)} 個拒絕；本資料觀察到的最小 raw p={min_pair_raw:.6f}、最小 Holm p={min_pair_holm:.6f}。詳細 p、方向、零差數與校正值見 `pairwise_tests_84.csv`。\n")
    if significant_pair:
        md.append("Holm 後仍有證據的 pair（僅列前 20）：\n")
        sig_rows = [{"pair": f"{r['model_a_name']} − {r['model_b_name']}", "criterion": r["criterion_zh"], "difference": r["difference_a_minus_b_normalized100"], "holm_p": r["holm84_p"]} for r in sorted(significant_pair, key=lambda x: x["holm84_p"])[:20]]
        md.append(markdown_table(sig_rows, [("pair", "模型對"), ("criterion", "分項"), ("difference", "A−B（0–100）"), ("holm_p", "Holm p")]))
    else:
        md.append("本資料下，84 個分項模型對經 Holm 校正後沒有任何一個達到 alpha=.05；這不等同於模型完全相同或證明沒有實質差異。整體 interaction 有訊號與具體 pair 經多重校正未達門檻可以同時成立。\n")
    md.append("\n## 排名變化與模型差異不是同一個問題\n")
    md.append(f"以每題的 7×4 normalized-100 panel 做模型與分項雙重中心化，按 source cluster 彙總後計算 interaction 矩陣平方和，T={interaction_observed_stat:.12f}，global exact sign-flip p={interaction_p:.6f}（未校正、單一探索性 global 檢定）。這個 p 值不是離散排名變化本身的 p 值。\n")
    md.append(f"事後的 rank sensitivity 將每題每分項轉為 1=最佳的七模型 midrank，再做同一種雙重中心化與 12-cluster sign-flip：T={rank_interaction_observed_stat:.12f}、p={rank_interaction_p:.6f}。它直接反映整體排序型態是否依分項改變，但不是任何單一換位的 p 值；因此不把它混入上面的總均分名次表。\n")
    md.append(f"若把 normalized interaction 與這個事後 rank sensitivity 勉強視為兩個 global tests，兩者 Holm-2 adjusted p 都是 {max(interaction_p, rank_interaction_p):.6f}；這只是敏感度資訊，不能取代預先指定的主要檢定。\n")
    md.append(f"另外對 21 模型對 × 6 分項對的 126 個 normalized-100 差異之差做雙尾 exact test 並 Holm-126：描述上有 {len(reverse_rows)} 個模型對在兩分項間反向（A−B 符號相反），其中 {len(significant_reverse)} 個同時通過 Holm alpha=.05；最小 raw p={min_criterion_pair_raw:.6f}、最小 Holm p={min_criterion_pair_holm:.6f}。由於 4,096 個排列的雙尾 p 解析度最低為 2/4096=.000488，而 Holm-126 的第一階即至少 .061523，這個設計本身不可能以 .05 通過 126 校正；不能把 0 個通過解讀成沒有 interaction。完整結果見 `criterion_pair_tests_126.csv`；若無通過者，不能把描述性的反向排序說成有統計證據。\n")
    md.append("\n## 五評審總分排名\n")
    md.append("`judge_total_ranks.csv` 另列 judge-01 至 judge-05 的 `score_0_100` 總分排名，以及先五評審平均再按 29 題等權的 `all-5-panel`；不要把這張總分表誤當成四分項排名。\n")
    pooled_rows = [r for r in judge_total_table if r["judge_slot"] == "all-5-panel"]
    md.append(markdown_table([
        {"model_name": r["model_name"], "mean": r["total_score_0_100_mean"], "rank": r["aggregate_rank"], "median": r["median_item_rank"]} for r in pooled_rows
    ], [("model_name", "模型"), ("mean", "五評審總分均值"), ("rank", "名次"), ("median", "逐題名次中位數")]))
    md.append("\n## 圖片子集（僅描述）\n")
    md.append(f"圖片為 {len(image_ids)} 張 × {len(model_ids)} 模型 × {len(image_judges)} 位可評審 = {len(image_valid_rows)} 個有效 cell；judge-03 的 {len(image_rows) - len(image_valid_rows)} rows 以 `unsupported_by_design` 保留，不補值。只有四張圖，故不做 cluster 顯著性推論。\n")
    md.append(markdown_table([
        {"model_name": r["model_name"], "mean": r["image_panel_mean_0_100"], "rank": r["aggregate_rank"], "median": r["median_image_rank"]} for r in image_table if "image_id" not in r
    ], [("model_name", "模型"), ("mean", "圖片 panel 均值"), ("rank", "描述名次"), ("median", "圖片名次中位數")]))
    md.append("\n## 可交換性假設與限制\n")
    md.append("- 五位評審是同一題的評分者，已在 item 內平均；不能把五位評審當五個獨立樣本。\n- sign-flip 的零假設需要 paired source-cluster 差異在符號上可交換／對稱，且 12 個 cluster 作為配對單位；樣本少，結果是 benchmark 內的描述性推論。\n- Holm-84 與 Holm-126 只對各自明確列出的探索性 family 校正；不代表臨床安全、普遍最佳、因果效果或族群泛化，也沒有人工 gold。\n- 量尺正規化是正的線性轉換：同一分項內的 pairwise sign-flip p 不受單位改變；global interaction 與跨分項差異之差則明確使用 normalized-100，因其需要將不同原始上限放到同一顯示尺度。\n")
    md.append("\n## 重跑\n")
    md.append("```sh\npython scripts/criterion_analysis.py --data-dir docs/data --output-dir verification/criteria\n```\n")
    (output_dir / "criterion_analysis.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "common_text_cells": len(text_rows),
        "models": model_ids,
        "criteria": CRITERIA,
        "holm84_rejects": len(significant_pair),
        "interaction_p": interaction_p,
        "interaction_T": interaction_observed_stat,
        "rank_interaction_p": rank_interaction_p,
        "rank_interaction_T": rank_interaction_observed_stat,
        "reverse_signs": len(reverse_rows),
        "holm126_reverse_rejects": len(significant_reverse),
        "image_valid_rows": len(image_valid_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
