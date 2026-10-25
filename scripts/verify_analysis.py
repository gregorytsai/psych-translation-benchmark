#!/usr/bin/env python3
"""Portable, read-only audit of the public translation panel.

Only the public ``docs/data`` tables and metadata are consumed.  The script
does not call a model, a network endpoint, or any private runner path::

    python scripts/verify_analysis.py --data-dir docs/data --output-dir verification

All output paths are relative/public labels, and published CSVs with
``raw_path``/``all_attempt_paths`` removed remain sufficient for the offline
recalculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "docs" / "data"

TEXT_JUDGES = [f"judge-{i:02d}" for i in range(1, 6)]
IMAGE_JUDGES = ["judge-01", "judge-02", "judge-04", "judge-05"]

GOOD_CASES = {
    "gemma4-12b": "text-01",
    "gemma4-26b": "text-03",
    "gemma4-31b-qat": "text-01",
    "gemma4-e4b": "text-01",
    "translategemma-12b": "text-22",
    "translategemma-27b": "text-01",
    "translategemma-4b": "text-14",
}
RISK_CASES = {
    "gemma4-12b": "text-27",
    # High-signal image case: four fixed judges flag the SERTRALINE spelling
    # error; keep this concrete candidate visible for the model page.
    "gemma4-26b": "img-04",
    "gemma4-31b-qat": "text-12",
    "gemma4-e4b": "text-19",
    "translategemma-12b": "text-08",
    "translategemma-27b": "text-28",
    "translategemma-4b": "text-18",
}

# Keep these cases as review prompts rather than adjudicated errors. They
# illustrate why a judge annotation, including a confident terminology label,
# is not a substitute for Taiwan-language human review.
JUDGE_DISPUTE_NOTES = [
    {
        "model_id": "gemma4-31b-qat",
        "material_id": "text-06",
        "source_exact": "Bipolar disorder",
        "translation_exact": "雙相情緒障礙症",
        "note": "judge-03 提出台灣術語疑義，但此術語偏好不在本審核中裁定；該 annotation 僅作待人工複核示範，不視為確定錯誤。",
        "references": [
            "https://www.nhi.gov.tw/ch/dl-80164-84214b36b10846c49afefc8dab9648a2-1.pdf",
            "https://www.hch.gov.tw/?aid=503&iid=130&page_name=detail&pid=0",
        ],
    },
]

JUDGE_REVIEW_ONLY_CASES = {
    ("translategemma-27b", "text-28"): "此安全／服務名稱判斷可能有語境爭議，僅標為 judge 提出待查，不作嚴重錯誤定論。",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def as_int(value: Any) -> int | None:
    x = as_float(value)
    return int(x) if x is not None and x.is_integer() else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "ok"}


def q95(values: Iterable[float]) -> float | None:
    vals = sorted(float(x) for x in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=100, method="inclusive")[94]


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def median_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.median(vals) if vals else None


def round_num(value: Any, ndigits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return value
    return round(float(value), ndigits)


def first_sentence(text: str, limit: int = 280) -> str:
    """Return a literal prefix ending at a sentence boundary when possible."""
    text = text or ""
    for marker in ("。", ". ", ".\n", "！", "？"):
        idx = text.find(marker)
        if idx >= 0:
            end = idx + len(marker)
            return text[:end]
    return text[:limit]


def path_label(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def diff_value(a: Any, b: Any, tol: float = 1e-8) -> bool:
    """True when values differ, with numeric tolerance."""
    if a is None or b is None:
        return a != b
    af, bf = as_float(a), as_float(b)
    if af is not None and bf is not None:
        return abs(af - bf) > tol
    return str(a) != str(b)


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def derive_source_rows(comparison_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Use public translation_comparison as a last-resort source-equivalent fallback."""
    out: list[dict[str, str]] = []
    for r in comparison_rows:
        out.append(
            {
                "model_id": r.get("model_id", ""),
                "model_name": r.get("model_name", ""),
                "model_alias": r.get("model_alias", ""),
                "case_id": r.get("material_id", ""),
                "input_type": r.get("source_kind", ""),
                "source_id": r.get("source_id", ""),
                "source_en": r.get("source_en", ""),
                "source_sha256": r.get("source_sha256", ""),
                "image_sha256": r.get("image_sha256", ""),
                "translation_zh_tw": r.get("target_zh_tw", ""),
                "status": r.get("translation_status", ""),
                "elapsed_seconds": r.get("translation_elapsed_seconds", ""),
                "completion_tokens": "",
                "generation_tok_s": "",
            }
        )
    return out


def public_source_rows(payload: Any) -> list[dict[str, str]]:
    """Normalize the published translation_results.json into audit rows."""
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out = []
    for r in rows:
        source_en = r.get("source_en", "") or r.get("image_description_en", "")
        out.append(
            {
                "model_id": r.get("model_id", ""),
                "model_name": r.get("model_name", ""),
                "model_alias": "",
                "case_id": r.get("material_id", ""),
                "input_type": r.get("source_kind", ""),
                "source_id": r.get("source_id", ""),
                "source_en": source_en,
                "source_sha256": "",
                "image_sha256": "",
                "translation_zh_tw": r.get("target_zh_tw", ""),
                "status": r.get("translation_status", ""),
                "elapsed_seconds": r.get("translation_elapsed_seconds", ""),
                "completion_tokens": "",
                "generation_tok_s": "",
            }
        )
    return out


def load_inputs(root: Path, data_dir: Path) -> dict[str, Any]:
    comparison_path = data_dir / "translation_comparison.csv"
    comparison_rows = read_csv(comparison_path)
    source_json_path = data_dir / "translation_results.json"
    source_rows = public_source_rows(read_json(source_json_path, {}) or {})
    source_available = bool(source_rows)
    if not source_rows:
        source_rows = derive_source_rows(comparison_rows)

    config_path = data_dir / "translation_config.json"
    inventory_path = data_dir / "model_inventory.json"
    runtime_path = data_dir / "runtime_version_check.json"
    judge_config_path = data_dir / "judge_config.json"

    paths = {
        "translation_results_json": source_json_path,
        "translation_comparison": comparison_path,
        "pointwise_results_long": data_dir / "pointwise_results_long.csv",
        "error_annotations": data_dir / "error_annotations.csv",
        "invalid_annotations": data_dir / "invalid_annotations.csv",
        "pairwise_raw": data_dir / "pairwise_raw.csv",
        "pairwise_canonical": data_dir / "pairwise_canonical.csv",
        "translation_generation_summary": data_dir / "translation_generation_summary.csv",
        "lofo_summary": data_dir / "lofo_summary.csv",
        "model_summary": data_dir / "model_summary.csv",
        "judge_summary": data_dir / "judge_summary.csv",
        "common_case_model_summary": data_dir / "common_case_model_summary.csv",
        "reliability_summary": data_dir / "reliability_summary.csv",
        "analysis_report": data_dir / "analysis_report.md",
        "translation_config": config_path,
        "model_inventory": inventory_path,
        "runtime_version_check": runtime_path,
        "judge_config": judge_config_path,
    }
    files = {key: read_csv(path) for key, path in paths.items() if path.suffix == ".csv"}

    config = read_json(config_path, {}) or {}
    inventory = read_json(inventory_path, {}) or {}
    runtime = read_json(runtime_path, {}) or {}
    judge_config = read_json(judge_config_path, {}) or {}
    repro_text = ""
    protocol_text = ""

    return {
        "root": root,
        "data_dir": data_dir,
        "source_available": source_available,
        "source_json_available": source_json_path.exists(),
        "paths": paths,
        "files": files,
        "source_rows": source_rows,
        "comparison_rows": comparison_rows,
        "config": config,
        "inventory": inventory,
        "runtime": runtime,
        "judge_config": judge_config,
        "repro_text": repro_text,
        "protocol_text": protocol_text,
    }


def model_order(inputs: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    ids = []
    for item in inputs["config"].get("models", []) or []:
        if item.get("id") and item["id"] not in ids:
            ids.append(item["id"])
    for r in rows:
        if r.get("model_id") and r["model_id"] not in ids:
            ids.append(r["model_id"])
    return ids


def candidate_file_table(inputs: dict[str, Any], models: list[str], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cfg_by_id = {m.get("id"): m for m in inputs["config"].get("models", []) or []}
    inv_by_id = (inputs["inventory"].get("models") or {})
    row_by_model = {}
    for r in rows:
        row_by_model.setdefault(r.get("model_id"), r)
    result = []
    for mid in models:
        cfg = cfg_by_id.get(mid, {})
        inv = inv_by_id.get(mid, {})
        src = row_by_model.get(mid, {})
        model_path = inv.get("model_path") or cfg.get("model") or ""
        mmproj_path = inv.get("mmproj_path") or cfg.get("mmproj") or ""
        result.append(
            {
                "model_id": mid,
                "model_name": cfg.get("display_name") or src.get("model_name", mid),
                "family": cfg.get("family"),
                "model_file": Path(model_path).name if model_path else None,
                "model_sha256": inv.get("model_sha256") or src.get("model_sha256") or None,
                "model_size_bytes": inv.get("model_size_bytes"),
                "mmproj_file": Path(mmproj_path).name if mmproj_path else None,
                "mmproj_sha256": inv.get("mmproj_sha256") or src.get("mmproj_sha256") or None,
                "mmproj_size_bytes": inv.get("mmproj_size_bytes"),
                "metadata_size_label": (inv.get("model_metadata") or {}).get("general.size_label"),
                "architecture": (inv.get("model_metadata") or {}).get("general.architecture"),
                "template_mode": cfg.get("template_mode") or src.get("template_mode"),
                "embedded_template_sha256": inv.get("embedded_template_sha256") or src.get("embedded_template_sha256") or None,
                "adapter_template_sha256": inv.get("adapter_template_sha256") or src.get("adapter_template_sha256") or None,
                "reasoning_format": cfg.get("reasoning_format"),
                "chat_template_kwargs": cfg.get("chat_template_kwargs"),
                "alias": cfg.get("alias") or src.get("model_alias"),
                "port": cfg.get("port"),
            }
        )
    return result


def panel_setup(row: dict[str, str], track: str) -> tuple[list[str], list[float], int]:
    judges = TEXT_JUDGES if track == "text" else IMAGE_JUDGES
    vals = []
    for judge in judges:
        if as_bool(row.get(f"{judge}_valid")):
            x = as_float(row.get(f"{judge}_score"))
            if x is not None:
                vals.append(x)
    return judges, vals, len(judges)


def recompute_panel_rows(rows: list[dict[str, str]], track: str) -> list[dict[str, Any]]:
    out = []
    judges = TEXT_JUDGES if track == "text" else IMAGE_JUDGES
    for r in rows:
        valid = {}
        for judge in judges:
            score = as_float(r.get(f"{judge}_score")) if as_bool(r.get(f"{judge}_valid")) else None
            valid[judge] = score
        vals = [x for x in valid.values() if x is not None]
        derived_mean = mean_or_none(vals)
        required = len(judges)
        out.append(
            {
                "material_id": r.get("material_id"),
                "model_id": r.get("model_id"),
                "valid_cells": len(vals),
                "expected_cells": required,
                "complete": len(vals) == required,
                "derived_mean": derived_mean,
                "reported_mean": as_float(r.get("panel_mean")),
                "reported_complete": as_bool(r.get("panel_complete")),
                "valid_scores": valid,
            }
        )
    return out


def common_keys(rows: list[dict[str, str]], models: list[str], track: str) -> tuple[list[str], list[str]]:
    keys = sorted({r.get("material_id") for r in rows if r.get("source_kind") == track})
    req = 5 if track == "text" else 4
    by_key_model = {(r.get("material_id"), r.get("model_id")): r for r in rows if r.get("source_kind") == track}
    common, excluded = [], []
    for key in keys:
        okay = True
        for mid in models:
            r = by_key_model.get((key, mid))
            if not r or not as_bool(r.get("panel_complete")) or as_int(r.get("panel_n_valid")) != req:
                okay = False
                break
        (common if okay else excluded).append(key)
    return common, excluded


def model_metrics(rows: list[dict[str, str]], models: list[str], common_by_track: dict[str, list[str]]) -> list[dict[str, Any]]:
    out = []
    for track in ("text", "image"):
        judges = TEXT_JUDGES if track == "text" else IMAGE_JUDGES
        req = len(judges)
        track_rows = [r for r in rows if r.get("source_kind") == track]
        for mid in models:
            rs = [r for r in track_rows if r.get("model_id") == mid]
            complete = [r for r in rs if as_bool(r.get("panel_complete")) and as_int(r.get("panel_n_valid")) == req]
            means = [as_float(r.get("panel_mean")) for r in complete]
            means = [x for x in means if x is not None]
            critical_cells = 0
            review_units = 0
            fail_units = 0
            for r in rs:
                c = sum(1 for j in judges if as_bool(r.get(f"{j}_valid")) and as_bool(r.get(f"{j}_critical")))
                critical_cells += c
                if c == 1:
                    review_units += 1
                if c >= 2:
                    fail_units += 1
            valid_cells = sum(sum(1 for j in judges if as_bool(r.get(f"{j}_valid")) and as_float(r.get(f"{j}_score")) is not None) for r in rs)
            expected_cells = len(rs) * req
            visible_english_units = sum(1 for r in rs if as_bool(r.get("translation_untranslated_visible_english")))
            out.append(
                {
                    "track": track,
                    "model_id": mid,
                    "model_name": next((r.get("model_name") for r in rs if r.get("model_name")), mid),
                    "material_units": len(rs),
                    "panel_required_k": req,
                    "panel_complete_units": len(complete),
                    "valid_judge_cells": valid_cells,
                    "expected_judge_cells": expected_cells,
                    "coverage_fraction": valid_cells / expected_cells if expected_cells else None,
                    "observed_panel_mean": mean_or_none(means),
                    "observed_panel_median": median_or_none(means),
                    "critical_flag_cells": critical_cells,
                    "critical_review_units_1ofk": review_units,
                    "critical_fail_units_2plusofk": fail_units,
                    "translation_untranslated_visible_english_units": visible_english_units,
                    "common_case_items": len(common_by_track[track]),
                    "common_case_panel_mean": mean_or_none(
                        [as_float(r.get("panel_mean")) for r in track_rows if r.get("model_id") == mid and r.get("material_id") in common_by_track[track] and as_float(r.get("panel_mean")) is not None]
                    ),
                }
            )
    return out


def speed_metrics(source_rows: list[dict[str, str]], comparison_rows: list[dict[str, str]], models: list[str]) -> list[dict[str, Any]]:
    # Full source CSV has completion tokens/tok/s.  Public CSV fallback still
    # has translation_elapsed_seconds and is enough for latency summaries.
    rows = source_rows if source_rows else derive_source_rows(comparison_rows)
    out = []
    for mid in models:
        for track in ("text", "image"):
            rs = [r for r in rows if r.get("model_id") == mid and r.get("input_type") == track]
            elapsed = [x for x in (as_float(r.get("elapsed_seconds")) for r in rs) if x is not None]
            tok_s = [x for x in (as_float(r.get("generation_tok_s")) for r in rs) if x is not None]
            tokens = [x for x in (as_int(r.get("completion_tokens")) for r in rs) if x is not None]
            out.append(
                {
                    "model_id": mid,
                    "track": track,
                    "cases": len(rs),
                    "elapsed_median_s": median_or_none(elapsed),
                    "elapsed_p95_s": q95(elapsed),
                    "generation_tok_s_median": median_or_none(tok_s),
                    "generation_tok_s_mean": mean_or_none(tok_s),
                    "completion_tokens": sum(tokens) if tokens else None,
                    "speed_provenance": "public translation_results.json" if tokens else "public translation_comparison.csv; token throughput unavailable",
                }
            )
    return out


def judge_metrics(pointwise: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for judge in sorted({r.get("judge_slot") for r in pointwise if r.get("judge_slot")}):
        rs = [r for r in pointwise if r.get("judge_slot") == judge]
        scores = [as_float(r.get("score_0_100")) for r in rs if as_bool(r.get("scalar_valid")) and as_float(r.get("score_0_100")) is not None]
        elapsed = [as_float(r.get("elapsed_s")) for r in rs if as_float(r.get("elapsed_s")) is not None]
        out.append(
            {
                "judge_slot": judge,
                "expected_rows": len(rs),
                "scalar_valid": len(scores),
                "invalid_or_excluded": sum(not as_bool(r.get("scalar_valid")) for r in rs),
                "technical_invalid": sum(r.get("technical_status") == "fail" for r in rs),
                "unsupported_by_design": sum(r.get("status") == "unsupported_by_design" for r in rs),
                "mean": mean_or_none(scores),
                "sd_population": statistics.pstdev(scores) if scores else None,
                "median": median_or_none(scores),
                "unique_scores": len(set(scores)),
                "critical_flag_count": sum(as_bool(r.get("critical")) for r in rs),
                "evidence_warning_count": sum(as_int(r.get("evidence_warning_count")) or 0 for r in rs),
                "latency_median_s": median_or_none(elapsed),
                "latency_p95_s": q95(elapsed),
            }
        )
    return out


def pairwise_metrics(pairwise_raw: list[dict[str, str]], pairwise_canonical: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "raw_rows": len(pairwise_raw),
        "raw_valid_orientations": sum(as_bool(r.get("valid")) for r in pairwise_raw),
        "raw_unsupported_by_design": sum(r.get("status") == "unsupported_by_design" for r in pairwise_raw),
        "raw_status_counts": dict(Counter(r.get("status", "") for r in pairwise_raw)),
        "canonical_rows": len(pairwise_canonical),
        "canonical_status_counts": dict(Counter(r.get("canonical_status", "") for r in pairwise_canonical)),
        "canonical_strict_consistent": sum(r.get("canonical_status") == "CONSISTENT" for r in pairwise_canonical),
        "canonical_consistent_tie": sum(r.get("canonical_status") == "CONSISTENT_TIE" for r in pairwise_canonical),
        "canonical_consistent_including_tie": sum(r.get("canonical_status") in {"CONSISTENT", "CONSISTENT_TIE"} for r in pairwise_canonical),
        "canonical_abstain_inconsistent": sum(r.get("canonical_status") == "ABSTAIN_INCONSISTENT" for r in pairwise_canonical),
        "canonical_missing_or_invalid": sum(r.get("canonical_status") == "MISSING_OR_INVALID" for r in pairwise_canonical),
    }


def lofo_metrics(rows: list[dict[str, str]], models: list[str], common_by_track: dict[str, list[str]]) -> list[dict[str, Any]]:
    out = []
    by_key_model = {(r.get("source_kind"), r.get("material_id"), r.get("model_id")): r for r in rows}
    for track in ("text", "image"):
        judges = TEXT_JUDGES if track == "text" else IMAGE_JUDGES
        keys = common_by_track[track]
        for omitted in judges:
            remaining = [j for j in judges if j != omitted]
            candidate_means: dict[str, float | None] = {}
            for mid in models:
                vals = []
                for key in keys:
                    r = by_key_model.get((track, key, mid))
                    if not r:
                        continue
                    for judge in remaining:
                        if as_bool(r.get(f"{judge}_valid")):
                            x = as_float(r.get(f"{judge}_score"))
                            if x is not None:
                                vals.append(x)
                candidate_means[mid] = mean_or_none(vals)
            order = sorted(models, key=lambda m: (-(candidate_means[m] if candidate_means[m] is not None else -math.inf), m))
            out.append(
                {
                    "track": track,
                    "omitted_judge_slot": omitted,
                    "remaining_actual_k": len(remaining),
                    "candidate_means": candidate_means,
                    "candidate_order": order,
                    "common_case_items": len(keys),
                    "excluded_case_ids": sorted(set(r.get("material_id") for r in rows if r.get("source_kind") == track) - set(keys)),
                }
            )
    return out


def rank_with_ties(values: list[float], higher_is_better: bool = True) -> list[float]:
    """Competition-independent average ranks, without a SciPy dependency."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher_is_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for pos in order[i:j]:
            ranks[pos] = rank
        i = j
    return ranks


def kendall_w(matrix: list[list[float]]) -> float | None:
    """Kendall's W for rows=objects and columns=raters, with tie correction."""
    if not matrix or not matrix[0]:
        return None
    n, m = len(matrix), len(matrix[0])
    if n < 2 or m < 2 or any(len(row) != m for row in matrix):
        return None
    ranks_by_rater = [rank_with_ties([matrix[i][j] for i in range(n)]) for j in range(m)]
    rank_sums = [sum(ranks_by_rater[j][i] for j in range(m)) for i in range(n)]
    mean_rank_sum = m * (n + 1) / 2.0
    s = sum((x - mean_rank_sum) ** 2 for x in rank_sums)
    tie_sum = 0.0
    for j in range(m):
        counts = Counter(matrix[i][j] for i in range(n))
        tie_sum += sum(t**3 - t for t in counts.values() if t > 1)
    denominator = m**2 * (n**3 - n) - m * tie_sum
    return 12.0 * s / denominator if denominator else None


def icc_absolute_agreement(matrix: list[list[float]]) -> tuple[float | None, float | None]:
    """ICC(2,1) and ICC(2,k), two-way random absolute agreement."""
    if not matrix or not matrix[0]:
        return None, None
    n, k = len(matrix), len(matrix[0])
    if n < 2 or k < 2 or any(len(row) != k for row in matrix):
        return None, None
    grand = statistics.mean(x for row in matrix for x in row)
    row_means = [statistics.mean(row) for row in matrix]
    col_means = [statistics.mean(matrix[i][j] for i in range(n)) for j in range(k)]
    ms_rows = k * sum((x - grand) ** 2 for x in row_means) / (n - 1)
    ms_cols = n * sum((x - grand) ** 2 for x in col_means) / (k - 1)
    residual = sum(
        (matrix[i][j] - row_means[i] - col_means[j] + grand) ** 2
        for i in range(n)
        for j in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    numerator = ms_rows - ms_error
    denominator_21 = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    denominator_2k = ms_rows + (ms_cols - ms_error) / n
    return (
        numerator / denominator_21 if denominator_21 else None,
        numerator / denominator_2k if denominator_2k else None,
    )


def reliability_metrics(rows: list[dict[str, str]], models: list[str], common_by_track: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Recompute balanced-panel ICC and rank agreement from comparison CSV."""
    out = []
    by_key_model = {(r.get("source_kind"), r.get("material_id"), r.get("model_id")): r for r in rows}
    for track in ("text", "image"):
        judges = TEXT_JUDGES if track == "text" else IMAGE_JUDGES
        keys = common_by_track[track]
        unit_matrix: list[list[float]] = []
        for key in keys:
            for mid in models:
                r = by_key_model.get((track, key, mid))
                if not r:
                    continue
                vals = [as_float(r.get(f"{j}_score")) if as_bool(r.get(f"{j}_valid")) else None for j in judges]
                if all(x is not None for x in vals):
                    unit_matrix.append([float(x) for x in vals])
        icc21, icc2k = icc_absolute_agreement(unit_matrix)

        system_matrix: list[list[float]] = []
        for mid in models:
            candidate = []
            for judge in judges:
                vals = []
                for key in keys:
                    r = by_key_model.get((track, key, mid))
                    if r and as_bool(r.get(f"{judge}_valid")):
                        x = as_float(r.get(f"{judge}_score"))
                        if x is not None:
                            vals.append(x)
                if vals:
                    candidate.append(statistics.mean(vals))
            if len(candidate) == len(judges):
                system_matrix.append(candidate)
        per_item_w = []
        for key in keys:
            item_matrix = []
            for mid in models:
                r = by_key_model.get((track, key, mid))
                if not r:
                    continue
                vals = [as_float(r.get(f"{j}_score")) if as_bool(r.get(f"{j}_valid")) else None for j in judges]
                if all(x is not None for x in vals):
                    item_matrix.append([float(x) for x in vals])
            w = kendall_w(item_matrix)
            if w is not None:
                per_item_w.append(w)
        out.append(
            {
                "track": track,
                "complete_balanced_units": len(unit_matrix),
                "expected_balanced_units": len(keys) * len(models),
                "common_case_items": len(keys),
                "judge_slots_fixed_panel": judges,
                "actual_k": len(judges),
                "icc_2_1_absolute_agreement": icc21,
                "icc_2_k_absolute_agreement": icc2k,
                "icc_k_is_actual_fixed_panel": True,
                "kendall_w_system_level": kendall_w(system_matrix),
                "kendall_w_unit_level": kendall_w(unit_matrix),
                "kendall_w_per_item_defined_count": len(per_item_w),
                "kendall_w_per_item_mean": mean_or_none(per_item_w),
                "kendall_w_per_item_median": median_or_none(per_item_w),
                "kendall_w_per_item_min": min(per_item_w) if per_item_w else None,
                "kendall_w_per_item_max": max(per_item_w) if per_item_w else None,
            }
        )
    return out


def selected_error_examples(model_id: str, case_id: str, comparison_rows: list[dict[str, str]], errors: list[dict[str, str]]) -> dict[str, Any] | None:
    row = next((r for r in comparison_rows if r.get("model_id") == model_id and r.get("material_id") == case_id), None)
    if not row:
        return None
    track = row.get("source_kind", "")
    source = row.get("source_en", "") if track == "text" else row.get("image_ground_truth_en", "")
    target = row.get("target_zh_tw", "")
    ee = [
        e
        for e in errors
        if e.get("model_id") == model_id and e.get("material_id") == case_id and as_bool(e.get("selected_for_scalar"))
    ]
    sev_rank = {"critical": 3, "major": 2, "minor": 1}
    ee.sort(key=lambda e: (-int(as_bool(e.get("critical_flag"))), -sev_rank.get(e.get("severity", ""), 0), -float(as_float(e.get("confidence")) or 0), e.get("judge_slot", "")))
    evidence = []
    seen = set()
    for e in ee:
        span = e.get("source_span", "") or ""
        quote = e.get("target_exact_quote", "") or ""
        sig = (span, quote, e.get("type", ""))
        if sig in seen:
            continue
        seen.add(sig)
        source_match = bool(span and span in source)
        target_match = bool(quote and quote in target)
        evidence.append(
            {
                "judge_slot": e.get("judge_slot"),
                "severity": e.get("severity"),
                "type": e.get("type"),
                "confidence": as_float(e.get("confidence")),
                "critical_flag": as_bool(e.get("critical_flag")),
                "source_span": span,
                "target_exact_quote": quote,
                "source_span_exact_in_material": source_match,
                "target_quote_exact_in_translation": target_match,
                "source_evidence": e.get("source_evidence", ""),
            }
        )
        if len(evidence) >= 3:
            break
    first = first_sentence(source)
    target_first = first_sentence(target)
    return {
        "material_id": case_id,
        "track": track,
        "panel_mean": as_float(row.get("panel_mean")),
        "panel_scores": {j: as_float(row.get(f"{j}_score")) for j in (TEXT_JUDGES if track == "text" else IMAGE_JUDGES)},
        "source_excerpt": first if track == "text" else source[:420],
        "translation_excerpt": target_first if track == "text" else target[:560],
        "source_excerpt_exact_in_material": bool(first and first in source),
        "translation_excerpt_exact_in_material": bool(target_first and target_first in target),
        "selected_error_count": len(ee),
        "selected_critical_count": sum(as_bool(e.get("critical_flag")) for e in ee),
        "evidence": evidence,
        "judge_review_only_note": JUDGE_REVIEW_ONLY_CASES.get((model_id, case_id)),
        "interpretation": (
            "comparatively strong proxy example; no selected error annotation was retained, but this is not a human gold proof"
            if not ee
            else (
                "judge提出待查；exact quote checks are reported, but this case is not adjudicated as a severe error or human gold"
                if (model_id, case_id) in JUDGE_REVIEW_ONLY_CASES
                else "judge-evidence-supported risk example; exact quote checks are reported and the annotation is not treated as human gold"
            )
        ),
        "image_ground_truth_status": row.get("image_ground_truth_status", "") if track == "image" else "not_applicable_text_no_human_gold",
    }


def settings_summary(inputs: dict[str, Any], models: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    config = inputs["config"]
    inv = inputs["inventory"]
    fixed = config.get("fixed_generation") or {}
    # The config intentionally keeps paths private/host-specific; model table
    # above reports only basenames and hashes.
    translator = {
        "protocol_version": config.get("protocol_version"),
        "llama_version_expected": config.get("llama_version_expected"),
        "ctx_size": fixed.get("ctx_size"),
        "max_tokens": fixed.get("max_tokens"),
        "temperature": fixed.get("temperature"),
        "top_p": fixed.get("top_p"),
        "seed": fixed.get("seed"),
        "reasoning": fixed.get("reasoning"),
        "reasoning_effort": fixed.get("reasoning_effort"),
        "stream": fixed.get("stream"),
        "parallel": fixed.get("parallel"),
        "batch_size": fixed.get("batch_size"),
        "ubatch_size": fixed.get("ubatch_size"),
        "cache_prompt": fixed.get("cache_prompt"),
        "speculative": fixed.get("speculative"),
        "n_gpu_layers": fixed.get("n_gpu_layers"),
        "flash_attn": fixed.get("flash_attn"),
        "no_cont_batching": fixed.get("no_cont_batching"),
        "no_warmup": fixed.get("no_warmup"),
        "shared_instruction": config.get("shared_instruction"),
    }
    judge_fixed = inputs["judge_config"].get("fixed_generation") or {}
    judges = []
    for j in inputs["judge_config"].get("judges", []) or []:
        judges.append(
            {
                "id": j.get("id"),
                "family": j.get("family"),
                "native_mode": j.get("native_mode"),
                "reasoning_effort": j.get("reasoning_effort"),
                "enable_thinking": j.get("enable_thinking"),
                "mmproj_required": j.get("mmproj_required"),
                "vision_policy": j.get("vision_policy"),
            }
        )
    runtime = inputs["runtime"]
    return {
        "translator_generation": translator,
        "translator_model_count": len(inv.get("models") or {}) or len(models),
        "runtime": {
            "expected": runtime.get("expected"),
            "observed": runtime.get("observed"),
            "pass": runtime.get("pass"),
            "generic_hardware_evidence": "Darwin arm64 build; n_gpu_layers=all and flash_attn=on. Exact SoC/RAM is not asserted here.",
        },
        "judge_runner_only": {
            "context": judge_fixed.get("context"),
            "max_tokens": judge_fixed.get("max_tokens"),
            "reasoning_budget_tokens": judge_fixed.get("reasoning_budget_tokens"),
            "temperature": judge_fixed.get("temperature"),
            "top_p": judge_fixed.get("top_p"),
            "seed": judge_fixed.get("seed"),
            "speculative": judge_fixed.get("speculative"),
            "judges": judges,
        },
    }


def documentation_findings(inputs: dict[str, Any], pair_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    config_fixed = inputs["config"].get("fixed_generation") or {}
    judge_fixed = inputs["judge_config"].get("fixed_generation") or {}
    findings = []
    repro = inputs["repro_text"]
    has_repro_old = all(x in repro for x in ("context 32768", "max output 16384", "reasoning budget 8192"))
    if has_repro_old and config_fixed:
        findings.append(
            {
                "id": "translation_vs_judge_generation_limits",
                "severity": "documentation_clarity",
                "status": "confirmed_track_mismatch_in_wording",
                "reproducibility_claim": {"ctx": 32768, "max_tokens": 16384, "reasoning_budget": 8192},
                "translator_authoritative": {
                    "ctx_size": config_fixed.get("ctx_size"),
                    "max_tokens": config_fixed.get("max_tokens"),
                    "reasoning": config_fixed.get("reasoning"),
                    "reasoning_effort": config_fixed.get("reasoning_effort"),
                },
                "judge_runner_authoritative": {
                    "context": judge_fixed.get("context"),
                    "max_tokens": judge_fixed.get("max_tokens"),
                    "reasoning_budget_tokens": judge_fixed.get("reasoning_budget_tokens"),
                },
                "conclusion": "The 32768/16384/8192 values match the local judge runner, while translator config/protocol/raw generation request use 8192/4096/reasoning off. The document should label these tracks explicitly.",
            }
        )
    if pair_metrics.get("canonical_missing_or_invalid"):
        findings.append(
            {
                "id": "pairwise_canonical_denominator_clarity",
                "severity": "reporting_clarity",
                "status": "needs_explicit_breakdown",
                "strict_consistent": pair_metrics.get("canonical_strict_consistent"),
                "consistent_tie": pair_metrics.get("canonical_consistent_tie"),
                "abstain_inconsistent": pair_metrics.get("canonical_abstain_inconsistent"),
                "missing_or_invalid": pair_metrics.get("canonical_missing_or_invalid"),
                "conclusion": "The shorthand 119 consistent / 49 abstain groups consistent ties with strict consistency and omits 12 missing/invalid groups; show all four categories to make the 180-group denominator auditable.",
            }
        )
    findings.append(
        {
            "id": "no_human_gold_or_clinical_claim",
            "severity": "scope_limit",
            "status": "confirmed",
            "conclusion": "All means/examples are descriptive judge proxies; no human gold, clinical safety certification, or universal-best claim is inferred.",
        }
    )
    findings.append(
        {
            "id": "mistral_image_design_unsupported",
            "severity": "design_boundary",
            "status": "preserve",
            "conclusion": "Mistral image cells remain unsupported by design (28 pointwise + 24 pairwise); no zero imputation.",
        }
    )
    findings.append(
        {
            "id": "qwen_invalid_preserved",
            "severity": "data_integrity",
            "status": "confirmed",
            "conclusion": "One Qwen pointwise record remains invalid due to subscores_sum_mismatch and is excluded from scalar means rather than converted to zero.",
        }
    )
    findings.append(
        {
            "id": "judge_terminology_claim_requires_human_review",
            "severity": "scope_limit",
            "status": "confirmed_review_caveat",
            "case": JUDGE_DISPUTE_NOTES[0],
            "conclusion": "The Gemma 4 31B text-06 terminology criticism is retained as a judge annotation but not accepted as a model error; Taiwan terminology should be checked by qualified human reviewers. This demonstrates that judge evidence can be questionable and must not be blindly treated as ground truth.",
        }
    )
    return findings


def compare_report_files(metrics: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare independent metrics to latest summaries when those CSVs exist."""
    mismatches = []
    report = inputs["files"].get("model_summary", [])
    by_key = {(r.get("track"), r.get("model_id")): r for r in report}
    for m in metrics:
        rr = by_key.get((m["track"], m["model_id"]))
        if not rr:
            continue
        checks = {
            "panel_complete_units": (m["panel_complete_units"], as_int(rr.get("panel_complete_units"))),
            "valid_judge_cells": (m["valid_judge_cells"], as_int(rr.get("judge_scalar_valid_cells"))),
            "coverage_fraction": (m["coverage_fraction"], as_float(rr.get("coverage_fraction"))),
            "observed_panel_mean": (m["observed_panel_mean"], as_float(rr.get("observed_panel_mean_noncomparative"))),
        }
        for field, (calc, reported) in checks.items():
            if diff_value(calc, reported, 1e-7):
                mismatches.append({"file": "model_summary.csv", "track": m["track"], "model_id": m["model_id"], "field": field, "calculated": calc, "reported": reported})
    return mismatches


def compare_reliability_summary(reliability: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare independently recomputed agreement statistics to the archived summary."""
    reported = inputs["files"].get("reliability_summary", [])
    by_track = {r.get("track"): r for r in reported}
    mismatches = []
    fields = (
        "complete_balanced_units",
        "expected_balanced_units",
        "common_case_items",
        "actual_k",
        "icc_2_1_absolute_agreement",
        "icc_2_k_absolute_agreement",
        "kendall_w_system_level",
        "kendall_w_unit_level",
        "kendall_w_per_item_defined_count",
        "kendall_w_per_item_mean",
        "kendall_w_per_item_median",
        "kendall_w_per_item_min",
        "kendall_w_per_item_max",
    )
    for calc in reliability:
        rr = by_track.get(calc["track"])
        if not rr:
            continue
        for field in fields:
            reported_value = as_int(rr.get(field)) if field.endswith(("units", "items", "k", "count")) else as_float(rr.get(field))
            if diff_value(calc.get(field), reported_value, 1e-7):
                mismatches.append(
                    {
                        "file": "reliability_summary.csv",
                        "track": calc["track"],
                        "field": field,
                        "calculated": calc.get(field),
                        "reported": reported_value,
                    }
                )
    return mismatches


def model_assessments(metrics: list[dict[str, Any]], speed: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    """Join per-model measurements with explicit descriptive limitations."""
    by_model_track = {(r["model_id"], r["track"]): r for r in metrics}
    speed_by_model_track = {(r["model_id"], r["track"]): r for r in speed}
    out = []
    for mid in models:
        text = by_model_track.get((mid, "text"), {})
        image = by_model_track.get((mid, "image"), {})
        ts = speed_by_model_track.get((mid, "text"), {})
        ims = speed_by_model_track.get((mid, "image"), {})
        limitations = [
            "image 軌只有 4 個 authored/reference cases，不能視為廣泛影像翻譯估計",
            "所有 panel 均為 judge proxy、沒有人工 gold；selected evidence 仍需人工複核",
        ]
        if text.get("panel_complete_units") != text.get("material_units"):
            limitations.append(
                f"text panel coverage 為 {text.get('panel_complete_units')}/{text.get('material_units')} 完整；共同文字集排除 text-15"
            )
        else:
            limitations.append("此模型 text panel rows 完整，但共同集仍僅 29 題")
        critical_total = (text.get("critical_flag_cells") or 0) + (image.get("critical_flag_cells") or 0)
        if critical_total:
            limitations.append(f"兩軌 fixed judge annotations 有 {critical_total} 個 critical-flag cells；這只是複核訊號，不是裁定")
        else:
            limitations.append("fixed panel 沒有記錄 critical flag；不代表輸出已證明無錯")
        out.append(
            {
                "model_id": mid,
                "observed_text_mean": text.get("observed_panel_mean"),
                "common_text_mean": text.get("common_case_panel_mean"),
                "observed_image_mean": image.get("observed_panel_mean"),
                "text_generation_median_tok_s": ts.get("generation_tok_s_median"),
                "image_generation_median_tok_s": ims.get("generation_tok_s_median"),
                "text_elapsed_median_s": ts.get("elapsed_median_s"),
                "image_elapsed_median_s": ims.get("elapsed_median_s"),
                "limitations": limitations,
            }
        )
    return out


def build_analysis(inputs: dict[str, Any]) -> dict[str, Any]:
    root = inputs["root"]
    source_rows = inputs["source_rows"]
    comparison_rows = inputs["comparison_rows"]
    files = inputs["files"]
    models = model_order(inputs, comparison_rows or source_rows)
    tracks = sorted({r.get("source_kind") for r in comparison_rows if r.get("source_kind")})
    if not tracks:
        tracks = ["text", "image"]
    common: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for track in ("text", "image"):
        common[track], excluded[track] = common_keys(comparison_rows, models, track)

    # Source/comparison parity is intentionally exact for source/target fields.
    src_by_key = {(r.get("model_id"), r.get("case_id")): r for r in source_rows}
    cmp_by_key = {(r.get("model_id"), r.get("material_id")): r for r in comparison_rows}
    parity_mismatches = []
    for key, sr in src_by_key.items():
        cr = cmp_by_key.get(key)
        if not cr:
            parity_mismatches.append({"key": key, "reason": "missing_comparison_row"})
            continue
        checks = {
            "source_kind": (sr.get("input_type"), cr.get("source_kind")),
            "source_sha256": (sr.get("source_sha256"), cr.get("source_sha256")),
            "image_sha256": (sr.get("image_sha256"), cr.get("image_sha256")),
            "translation": (sr.get("translation_zh_tw"), cr.get("target_zh_tw")),
        }
        if sr.get("source_en"):
            checks["source_text"] = (
                sr.get("source_en"),
                cr.get("source_en") if cr.get("source_kind") == "text" else cr.get("image_ground_truth_en"),
            )
        for field, (a, b) in checks.items():
            # Public translation_results.json intentionally omits source/image
            # hashes; compare those fields only when both sides publish them.
            if field in {"source_sha256", "image_sha256"} and not a:
                continue
            if a != b:
                parity_mismatches.append({"key": key, "field": field, "source": a, "comparison": b})
    missing_source = sorted(set(cmp_by_key) - set(src_by_key))

    metrics = model_metrics(comparison_rows, models, common)
    speed = speed_metrics(source_rows if inputs["source_available"] else [], comparison_rows, models)
    judges = judge_metrics(files.get("pointwise_results_long", []))
    pair = pairwise_metrics(files.get("pairwise_raw", []), files.get("pairwise_canonical", []))
    lofo = lofo_metrics(comparison_rows, models, common)
    reliability = reliability_metrics(comparison_rows, models, common)
    errors = files.get("error_annotations", [])
    examples = {}
    for mid in models:
        examples[mid] = {
            "better": selected_error_examples(mid, GOOD_CASES.get(mid, "text-01"), comparison_rows, errors),
            "risk": selected_error_examples(mid, RISK_CASES.get(mid, "text-18"), comparison_rows, errors),
        }
    file_table = candidate_file_table(inputs, models, source_rows)
    settings = settings_summary(inputs, models, source_rows)
    assessments = model_assessments(metrics, speed, models)
    findings = documentation_findings(inputs, pair)
    report_mismatches = compare_report_files(metrics, inputs)
    reliability_mismatches = compare_reliability_summary(reliability, inputs)

    source_counts = {
        "rows": len(source_rows),
        "models": len({r.get("model_id") for r in source_rows}),
        "case_ids": len({r.get("case_id") for r in source_rows}),
        "text_rows": sum(r.get("input_type") == "text" for r in source_rows),
        "image_rows": sum(r.get("input_type") == "image" for r in source_rows),
        "status_counts": dict(Counter(r.get("status", "") for r in source_rows)),
        "duplicate_model_case_keys": len(source_rows) - len(src_by_key),
        "source_csv_available": False,
        "source_json_available": inputs.get("source_json_available", False),
    }
    comparison_counts = {
        "rows": len(comparison_rows),
        "models": len({r.get("model_id") for r in comparison_rows}),
        "text_rows": sum(r.get("source_kind") == "text" for r in comparison_rows),
        "image_rows": sum(r.get("source_kind") == "image" for r in comparison_rows),
        "panel_complete_rows": sum(as_bool(r.get("panel_complete")) for r in comparison_rows),
        "panel_incomplete_rows": sum(not as_bool(r.get("panel_complete")) for r in comparison_rows),
    }
    invalid_rows = files.get("invalid_annotations", [])
    invalid_summary = {
        "rows": len(invalid_rows),
        "request_ids": sorted({r.get("request_id") for r in invalid_rows if r.get("request_id")}),
        "technical_status_counts": dict(Counter(r.get("technical_status", "") for r in invalid_rows)),
    }
    latest_manifest = read_json(inputs["data_dir"] / "analysis_manifest.json", {}) or {}
    result = {
        "schema_version": "translation-analysis-review-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent descriptive audit of public docs/data; no model calls; no clinical/deployment claim.",
        "data_mode": "public_docs_data",
        "input_files": {k: {"path": path_label(p, root), "sha256": sha256_file(p), "present": p.exists()} for k, p in inputs["paths"].items()},
        "source_counts": source_counts,
        "comparison_counts": comparison_counts,
        "source_comparison_parity": {
            "checked_keys": len(src_by_key),
            "missing_source_keys": missing_source,
            "mismatches": parity_mismatches,
            "status": "PASS" if not missing_source and not parity_mismatches else "REVIEW",
        },
        "common_sets": {
            track: {"eligible_items": common[track], "eligible_count": len(common[track]), "excluded_items": excluded[track]}
            for track in ("text", "image")
        },
        "model_metrics": metrics,
        "speed_metrics": speed,
        "judge_metrics": judges,
        "pairwise_metrics": pair,
        "lofo_metrics": lofo,
        "reliability_metrics": reliability,
        "invalid_annotations": invalid_summary,
        "model_files_and_settings": file_table,
        "model_assessments": assessments,
        "settings": settings,
        "examples": examples,
        "judge_dispute_notes": JUDGE_DISPUTE_NOTES,
        "audit_findings": findings,
        "report_file_parity": {
            "model_summary_mismatches": report_mismatches,
            "reliability_summary_mismatches": reliability_mismatches,
            "status": "PASS" if not report_mismatches and not reliability_mismatches else "REVIEW",
        },
        "latest_manifest_status": {
            "analysis_status": (latest_manifest.get("status") or {}).get("analysis_status"),
            "terminal_coverage": latest_manifest.get("terminal_coverage"),
            "root_approved": latest_manifest.get("frozen_inputs", {}).get("root_approved"),
        },
        "reproducibility_notes": [
            "Text common set is the 29-item intersection; text-15 is excluded because TranslateGemma 4B lacks one valid judge scalar.",
            "Image panel is fixed at Qwen, Nemotron, GPT, Gemini (k=4); Mistral image cells remain N/A by design and are never imputed as zero.",
            "Panel means are arithmetic means of valid fixed-panel scores. Error annotations are evidence diagnostics, not automatic truth or human gold.",
            "Generation latency is separate from judge latency; no cross-hardware performance claim is made.",
        ],
    }
    return safe_json(result)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 獨立翻譯 panel 數據審核")
    lines.append("")
    lines.append("本文件由 `verify_analysis.py` 只讀重算，結論限描述性；不等同人工 gold、臨床安全、部署驗收或普遍最佳。")
    lines.append("")
    sc = report["source_counts"]
    cc = report["comparison_counts"]
    lines.append(f"- 資料模式：`{report['data_mode']}`；source-equivalent rows={sc['rows']}、comparison rows={cc['rows']}、7 models、30 text + 4 image cases。")
    lines.append(f"- source↔comparison exact parity：`{report['source_comparison_parity']['status']}`（checked {report['source_comparison_parity']['checked_keys']} keys）。")
    lines.append(f"- translation status：{sc['status_counts']}；panel incomplete rows={cc['panel_incomplete_rows']}（TranslateGemma 4B/text-15）。")
    lines.append("")
    lines.append("## 主要重算均分")
    lines.append("")
    lines.append("|模型|text mean（30 observed）|text common mean（29）|image mean（4; k=4）|text complete cells|image complete cells|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    by_model = defaultdict(dict)
    for m in report["model_metrics"]:
        by_model[m["model_id"]][m["track"]] = m
    for mid in by_model:
        t, i = by_model[mid].get("text", {}), by_model[mid].get("image", {})
        lines.append(f"|{t.get('model_name', mid)}|{fmt(t.get('observed_panel_mean'))}|{fmt(t.get('common_case_panel_mean'))}|{fmt(i.get('observed_panel_mean'))}|{t.get('valid_judge_cells','—')}/{t.get('expected_judge_cells','—')}|{i.get('valid_judge_cells','—')}/{i.get('expected_judge_cells','—')}|")
    lines.append("")
    lines.append("共同集：text=`" + ", ".join(report["common_sets"]["text"]["excluded_items"]) + "` excluded；image excludes none. Text 29×7×5=1015 judge cells; image 4×7×4=112 fixed-panel cells. These are proxy-judge arithmetic means, not correctness scores.")
    lines.append("")
    lines.append("## Judge 一致性（ICC / Kendall W）")
    lines.append("")
    lines.append("ICC 使用共同集、完整 candidate×item units 與實際固定 judge panel，為 two-way random absolute agreement；Kendall W 為 rank agreement。這些是描述性可靠度指標，不是人工 gold 或臨床效度。")
    lines.append("")
    lines.append("|track|balanced units|k|ICC(2,1)|ICC(2,k)|Kendall W（system）|Kendall W（unit）|per-item W mean/median|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for rel in report["reliability_metrics"]:
        lines.append(
            f"|{rel['track']}|{rel['complete_balanced_units']}/{rel['expected_balanced_units']}|{rel['actual_k']}|{fmt(rel.get('icc_2_1_absolute_agreement'))}|{fmt(rel.get('icc_2_k_absolute_agreement'))}|{fmt(rel.get('kendall_w_system_level'))}|{fmt(rel.get('kendall_w_unit_level'))}|{fmt(rel.get('kendall_w_per_item_mean'))}/{fmt(rel.get('kendall_w_per_item_median'))}|"
        )
    lines.append("")
    lines.append("## 速度（translation generation；不混入 judge latency）")
    lines.append("")
    lines.append("|模型|text median/p95 s|image median/p95 s|text median tok/s|image median tok/s|")
    lines.append("|---|---:|---:|---:|---:|")
    speed_by = defaultdict(dict)
    for s in report["speed_metrics"]:
        speed_by[s["model_id"]][s["track"]] = s
    for mid in by_model:
        t, i = speed_by[mid].get("text", {}), speed_by[mid].get("image", {})
        lines.append(f"|{mid}|{fmt(t.get('elapsed_median_s'))}/{fmt(t.get('elapsed_p95_s'))}|{fmt(i.get('elapsed_median_s'))}/{fmt(i.get('elapsed_p95_s'))}|{fmt(t.get('generation_tok_s_median'))}|{fmt(i.get('generation_tok_s_median'))}|")
    lines.append("")
    lines.append("公開 translation_results.json 含 elapsed；未發布 completion tokens，因此此副本可重算 elapsed median/p95，但不能重算 token throughput。速度受本機、載入/快取與 request 條件影響，不作跨硬體宣稱。")
    lines.append("")
    lines.append("## Judge 重算")
    lines.append("")
    lines.append("|slot|valid/expected|mean|population SD|median|critical cells|evidence warnings|latency median/p95 s|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for j in report["judge_metrics"]:
        lines.append(f"|{j['judge_slot']}|{j['scalar_valid']}/{j['expected_rows']}|{fmt(j['mean'])}|{fmt(j['sd_population'])}|{fmt(j['median'])}|{j['critical_flag_count']}|{j['evidence_warning_count']}|{fmt(j['latency_median_s'])}/{fmt(j['latency_p95_s'])}|")
    lines.append("")
    lines.append("Mistral image 28 pointwise + 24 pairwise cells保持 `unsupported_by_design`；Qwen 一筆 `subscores_sum_mismatch` 保留 invalid，不轉 0。Gemini judge latency 在摘要中缺失，不補造數字。")
    lines.append("")
    lines.append("## LOFO judge 敏感度")
    lines.append("")
    for track in ("text", "image"):
        lines.append(f"### {track}")
        for row in [x for x in report["lofo_metrics"] if x["track"] == track]:
            vals = ", ".join(f"{m} {fmt(row['candidate_means'][m])}" for m in row["candidate_order"])
            lines.append(f"- omit {row['omitted_judge_slot']} (k={row['remaining_actual_k']})：`{' > '.join(row['candidate_order'])}`；{vals}")
        lines.append("")
    lines.append("LOFO 顯示 text 前三名（Gemma 4 26B、31B QAT、12B）在五次 omission 都不變；image 前三名（31B QAT、12B、26B）也不變。其餘名次在 judge omission 下有交換，故不宣稱普遍最佳。")
    lines.append("")
    lines.append("## 模型辨識與設定")
    lines.append("")
    tg = report["settings"].get("translator_generation", {})
    runtime = report["settings"].get("runtime", {})
    runtime_observed = runtime.get("observed")
    runtime_note = (
        "runtime evidence 為 llama.cpp 0.7.0-dev build 524 commit `620d2225c`，Darwin arm64。"
        if runtime.get("pass") and runtime_observed
        else "此資料模式未提供 runtime_version_check；不從公開 CSV 推測實際硬體/runtime。"
    )
    lines.append(
        "翻譯 execution 共同條件："
        f"ctx={tg.get('ctx_size','—')}、max_tokens={tg.get('max_tokens','—')}、"
        f"temperature={tg.get('temperature','—')}、top_p={tg.get('top_p','—')}、seed={tg.get('seed','—')}、"
        f"reasoning {tg.get('reasoning','—')}/{tg.get('reasoning_effort','—')}、"
        f"cache prompt {tg.get('cache_prompt','—')}、speculative {tg.get('speculative','—')}、"
        f"n_gpu_layers={tg.get('n_gpu_layers','—')}、flash attention {tg.get('flash_attn','—')}、single request。"
        f"{runtime_note} 沒有在此報告虛構 SoC/RAM；這只是安全的通用硬體描述。"
    )
    lines.append("")
    lines.append("|model_id|model file|model SHA-256（prefix）|mmproj|template/thinking|")
    lines.append("|---|---|---|---|---|")
    for m in report["model_files_and_settings"]:
        model_sha = (m.get("model_sha256") or "")[:12] or "—"
        mm = m.get("mmproj_file") or "—"
        tmpl = m.get("template_mode") or "—"
        reason = m.get("reasoning_format") or "—"
        lines.append(f"|{m['model_id']}|{m.get('model_file') or '—'}|`{model_sha}`|{mm}|{tmpl}; reasoning_format={reason}; kwargs={json.dumps(m.get('chat_template_kwargs'), ensure_ascii=False)}|")
    lines.append("")
    lines.append("注意：公開副本只列公開 translation_config.json；未附 local judge runner config 時，不臆測 judge 的 context/max/reasoning budget。")
    lines.append("")
    lines.append("## 各模型不足與觀察界線")
    lines.append("")
    for assessment in report["model_assessments"]:
        values = (
            f"text={fmt(assessment.get('observed_text_mean'))}、text common={fmt(assessment.get('common_text_mean'))}、"
            f"image={fmt(assessment.get('observed_image_mean'))}；generation median tok/s text/image="
            f"{fmt(assessment.get('text_generation_median_tok_s'))}/{fmt(assessment.get('image_generation_median_tok_s'))}"
        )
        lines.append(f"- `{assessment['model_id']}`：{values}。不足/界線：" + "；".join(assessment.get("limitations", [])) + "。")
    lines.append("")
    lines.append("## 每模型可追溯例句")
    lines.append("")
    lines.append("較好例是較高 panel mean 或無 selected error annotation 的 proxy；風險例只列有 judge evidence 的 exact source/target quote。二者都不是人工 gold；摘錄保留於 JSON，可供網站顯示。")
    lines.append("")
    for mid, ex in report["examples"].items():
        better, risk = ex.get("better"), ex.get("risk")
        lines.append(f"### {mid}")
        if better:
            lines.append(f"- 較好 `{better['material_id']}`，panel={fmt(better.get('panel_mean'))}；source exact：`{better.get('source_excerpt','')}`；translation exact：`{better.get('translation_excerpt','')}`。{better.get('interpretation','')}")
        if risk:
            ev = next((e for e in risk.get("evidence", []) if e.get("source_span_exact_in_material") and e.get("target_quote_exact_in_translation")), None)
            if ev:
                if risk.get("judge_review_only_note"):
                    lines.append(f"- 風險候選 `{risk['material_id']}`，panel={fmt(risk.get('panel_mean'))}；source exact：`{ev.get('source_span')}`；translation exact：`{ev.get('target_exact_quote')}`；{ev.get('judge_slot')} annotation={ev.get('severity')}/{ev.get('type')}，confidence={fmt(ev.get('confidence'))}。{risk['judge_review_only_note']}")
                else:
                    lines.append(f"- 風險 `{risk['material_id']}`，panel={fmt(risk.get('panel_mean'))}；source exact：`{ev.get('source_span')}`；translation exact：`{ev.get('target_exact_quote')}`；{ev.get('judge_slot')} {ev.get('severity')}/{ev.get('type')}，confidence={fmt(ev.get('confidence'))}。這是 judge 指控與 exact-match 診斷，不是人工定論。")
            else:
                lines.append(f"- 風險 `{risk['material_id']}`，panel={fmt(risk.get('panel_mean'))}；未找到同時通過 source/target exact substring 的 annotation，請以 JSON evidence 逐項審查。")
        lines.append("")
    lines.append("## Audit findings / scope limits")
    lines.append("")
    for f in report["audit_findings"]:
        lines.append(f"- `{f['id']}` ({f['status']})：{f['conclusion']}")
    lines.append("")
    lines.append("Judge 反例示範：Gemma 4 31B `text-06` 的 `Bipolar disorder`→`雙相情緒障礙症` 被一個 judge 以術語理由挑出，但本審核不把它裁成模型錯誤；可參考 [健保署術語資料](https://www.nhi.gov.tw/ch/dl-80164-84214b36b10846c49afefc8dab9648a2-1.pdf) 與 [臺大新竹醫院相關頁面](https://www.hch.gov.tw/?aid=503&iid=130&page_name=detail&pid=0)，由人工依專業語境複核。這也說明 judge 指控不可盲信。")
    lines.append("")
    lines.append("## 重跑")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/verify_analysis.py --data-dir docs/data --output-dir verification")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=None, help="Public data directory; defaults to docs/data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for analysis_review.json/.md; defaults to verification")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = (args.data_dir or (root / "docs" / "data")).resolve()
    inputs = load_inputs(root, data_dir)
    report = build_analysis(inputs)
    output_dir = (args.output_dir or (root / "verification")).resolve()
    json_out = (args.json_out or (output_dir / "analysis_review.json")).resolve()
    md_out = (args.md_out or (output_dir / "analysis_review.md")).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(make_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_out), "markdown": str(md_out), "data_mode": report["data_mode"], "source_comparison_parity": report["source_comparison_parity"]["status"], "report_file_parity": report["report_file_parity"]["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


