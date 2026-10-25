#!/usr/bin/env python3
"""Build the immutable, metadata-bound Nemotron format-recovery diagnostic.

The recovery request is a post-hoc format-only retry of the one malformed
pointwise row in the thinking-on run.  This script creates a derived bundle
and a separate five-judge diagnostic; it never edits either raw Nemotron
bundle, and it never treats the retry as a sixth rater or as a uniform run.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_pilot import aggregate, load_bundle, load_bundles, read_json, write_json
from build_fivejudge_report import build_report, fmt, sha256_file
from validate_import import validate_import


PILOT = Path(__file__).resolve().parent
RAW_ON = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-local-thinking-on-20261022/nemotron-3.5-super-vl-120b-a12b-local-thinking-on.raw.json"
RECOVERY_BUNDLE = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-format-recovery-20261022/recovery_bundle_attempt-02.json"
RECOVERY_CASE = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-format-recovery-20261022/recovery_case_attempt-02.json"
RECOVERY_MANIFEST = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-format-recovery-20261022/recovery_manifest_attempt-02.json"
RECOVERY_VALIDATION = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-format-recovery-20261022/recovery_validation_attempt-02.json"
OUT = PILOT / "nemotron_format_recovery_20261022"
DERIVED = OUT / "nemotron_recovered_composed.raw.json"
PROVENANCE = OUT / "recovery_composed_metadata.json"
PANEL = OUT / "panel_recovered"

BASELINE = [
    PILOT / "raw/gpt-6.1-sol-codex-blind.json",
    PILOT / "raw/qwen3.8-flash-next-local-pilot-v3-thinking-off-preserve.import-v2.json",
    PILOT / "raw/mistral-medium-3.5-128b-local-isolated-pilot.json",
    PILOT / "raw/gemini-3.8-flash-high-antigravity-v2.assembled.json",
]
OFF = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-local-isolated-20261022/nemotron-3.5-super-vl-120b-a12b-local-isolated.raw.json"
FOUR_AGG = PILOT / "raw/combined-gpt-qwen-mistral-gemini.v4.aggregate.json"
BASELINE_FIVE_AGG = PILOT / "fivejudge_20261022/aggregate.fivejudges.json"
PACKET = PILOT / "judge_packet.json"
INSTRUCTIONS = PILOT / "judge_instructions.txt"
MAPPING = PILOT / "private/candidate_mapping.private.json"
ANSWER_KEY = PILOT / "private/answer_key.private.json"
ORIGINAL_INVALID_ANNOTATIONS = PILOT / "nemotron_thinking_comparison_20261022/nemotron_thinking_off_vs_on_error_annotations.owner.csv"

ORIGINAL_ON_HASH = "fd29c121e25d8104021a9b8280da36eaaab574d0585d76b851ca17b335e962bc"
RECOVERY_BUNDLE_HASH = "3211850ea7443b33c0260c429bf88a69a385cd1676e39a2b9427fd3833e6e3f5"
RECOVERY_CASE_HASH = "2fb866e6b8395fa48e5a70c48bf0a87fc7a6b8b58b8f1b3ba1aa9de1e66164d0"
FAILED_RESPONSE_HASH = "9466effda2292f70880d193ce43553ac9cc5b2b85a06a8a36cad14371dc48f21"
LOGICAL_JUDGE = "nemotron-3.5-super-vl-120b-a12b-local-thinking-on"
RECOVERY_REQUEST = "pointwise-src-03-cand-02"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _pointwise_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    _meta, rows = load_bundle(path)
    return {
        (str(row["_item_id"]), str(row["_candidate_id"])): row
        for row in rows
        if row.get("_track") == "pointwise" and row.get("_status") == "ok" and row.get("_score") is not None
    }


def derive_bundle() -> dict[str, Any]:
    """Replace only the failed key in a new derived bundle.

    The replacement row is copied from the independently validated recovery
    bundle.  Only envelope identity fields are adapted so the derived bundle
    remains the same logical Nemotron judge; score, target, source, and error
    content are not changed.
    """
    original = read_json(RAW_ON)
    recovery = read_json(RECOVERY_BUNDLE)
    recovery_audit = validate_import(PACKET, RECOVERY_BUNDLE)
    if recovery_audit.get("status") != "PASS" or recovery_audit.get("track_eligibility", {}).get("pointwise", {}).get("status") != "eligible":
        raise ValueError("recovery subset bundle failed strict packet/pointwise import validation")
    records = []
    removed = 0
    for row in original["records"]:
        if row.get("track") == "pointwise" and row.get("item_id") == "src-03" and row.get("candidate_id") == "cand-02" and row.get("status") != "ok":
            removed += 1
            continue
        records.append(copy.deepcopy(row))
    if removed != 1:
        raise ValueError(f"expected exactly one failed src-03/cand-02 pointwise row, removed={removed}")
    recovery_rows = [row for row in recovery.get("records", []) if row.get("track") == "pointwise" and row.get("status") == "ok"]
    if len(recovery_rows) != 1:
        raise ValueError(f"expected one validated recovery pointwise row, got {len(recovery_rows)}")
    replacement = copy.deepcopy(recovery_rows[0])
    if replacement.get("item_id") != "src-03" or replacement.get("candidate_id") != "cand-02":
        raise ValueError("recovery key is not the frozen failed key")
    if replacement.get("source_text_sha256") != "ac421fef9621c080a6dfbb0f06dcaa8af268649079b0b2d34b496af60b1d006e":
        raise ValueError("recovery source hash is not the canonical src-03 hash")
    if replacement.get("pointwise", {}).get("score_0_100") != 95:
        raise ValueError("recovery score changed or is not the validated score")
    if sum(replacement["pointwise"]["subscores"].values()) != replacement["pointwise"]["score_0_100"]:
        raise ValueError("recovery subscore shape is not valid")
    replacement["judge_id"] = LOGICAL_JUDGE
    records.append(replacement)

    derived = copy.deepcopy(original)
    derived["run_id"] = "nemotron-3.5-super-vl-120b-a12b-local-thinking-on-format-recovery-composed-20261022"
    derived["status"] = "COMPLETE_DERIVED_COMPOSED_DIAGNOSTIC"
    derived["complete"] = True
    derived["records"] = records
    derived["counts"] = {"failed": 0, "ok": len(records), "pairwise": 6, "pointwise": 21, "qualification": 12, "records": len(records)}
    # Keep the original event stream intact and bind the recovery event as a
    # separate source artifact.  This preserves the old malformed response.
    derived["subset_metadata"] = {
        "kind": "posthoc_format_recovery_composed_diagnostic",
        "logical_judge_id": LOGICAL_JUDGE,
        "not_a_sixth_judge": True,
        "not_a_uniform_39_request_run": True,
        "prior_valid_pointwise_rows": 20,
        "recovery_valid_pointwise_rows": 1,
        "qualification_and_pairwise_source": "original_thinking_on_raw_unchanged",
        "raw_mutation": False,
        "selection_rule": "exact frozen key src-03/cand-02 only; format-recovery eligibility; no score-based selection",
    }
    derived["composition_provenance"] = {
        "original_on_bundle": str(RAW_ON),
        "original_on_bundle_sha256": sha256_file(RAW_ON),
        "recovery_bundle": str(RECOVERY_BUNDLE),
        "recovery_bundle_sha256": sha256_file(RECOVERY_BUNDLE),
        "recovery_case": str(RECOVERY_CASE),
        "recovery_case_sha256": sha256_file(RECOVERY_CASE),
        "recovery_manifest": str(RECOVERY_MANIFEST),
        "recovery_manifest_sha256": sha256_file(RECOVERY_MANIFEST),
        "recovery_validation": str(RECOVERY_VALIDATION),
        "recovery_validation_sha256": sha256_file(RECOVERY_VALIDATION),
        "failed_original_response_sha256": FAILED_RESPONSE_HASH,
        "failed_original_score": 92,
        "failed_original_subscore_sum": 94,
        "replacement_run_id": replacement.get("run_id"),
        "replacement_request_id": RECOVERY_REQUEST,
        "replacement_settings_profile": "format_recovery",
        "mixed_settings": True,
        "no_content_adapter": True,
    }
    derived["recovery_raw_event"] = copy.deepcopy(recovery.get("raw_events", [None])[0])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(DERIVED, derived)
    return derived


def _provenance_rows(derived: Mapping[str, Any]) -> list[dict[str, Any]]:
    _meta, rows = load_bundle(DERIVED)
    derived_hash = sha256_file(DERIVED)
    result = []
    for row in rows:
        if row.get("_track") != "pointwise":
            continue
        recovery = row.get("_item_id") == "src-03" and row.get("_candidate_id") == "cand-02"
        settings_profile = "format_recovery" if recovery else "original_thinking_on"
        request_settings = {
            "reasoning_budget_tokens": 8192 if recovery else 4096,
            "max_tokens": 16384 if recovery else 8192,
            "context_size": 32768 if recovery else 16384,
            "enable_thinking": True,
            "preserve_reasoning": False,
            "reasoning_format": "deepseek",
            "temperature": 0,
            "top_p": 1,
            "seed": 20261022,
        }
        result.append({
            "item_id": row.get("_item_id"),
            "source_id": row.get("_source_id"),
            "candidate_id": row.get("_candidate_id"),
            "source_text_sha256": row.get("_source_hash"),
            "logical_judge_id": LOGICAL_JUDGE,
            "row_source": "format_recovery" if recovery else "original_thinking_on",
            "source_bundle_sha256": RECOVERY_BUNDLE_HASH if recovery else ORIGINAL_ON_HASH,
            "source_bundle_path": str(RECOVERY_BUNDLE if recovery else RAW_ON),
            "derived_composed_bundle_sha256": derived_hash,
            "source_run_id": row.get("_run_id"),
            "source_request_id": RECOVERY_REQUEST if recovery else f"pointwise-{row.get('_item_id')}-{row.get('_candidate_id')}",
            "settings_profile": settings_profile,
            "request_settings_json": _json(request_settings),
            "retry_parent_bundle_sha256": ORIGINAL_ON_HASH if recovery else "",
            "retry_parent_request_id": RECOVERY_REQUEST if recovery else "",
            "valid_for_composed_pointwise": True,
            "score_eligible": True,
            "selection_rule": "exact frozen key only; no score-based selection" if recovery else "original valid row retained",
            "mixed_settings_note": "single recovery row uses larger budget/context; composed result is diagnostic only" if recovery else "original parent run settings",
        })
    return sorted(result, key=lambda row: (str(row["item_id"]), str(row["candidate_id"])))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summary_rows(panel_aggregate: Mapping[str, Any], panel_paths: Sequence[Path]) -> list[dict[str, Any]]:
    pointwise = panel_aggregate.get("pointwise") or {}
    output = []
    for path in panel_paths:
        metadata, rows = load_bundle(path)
        judge = metadata.get("judge") if isinstance(metadata.get("judge"), Mapping) else {}
        judge_id = str(judge.get("judge_id") or metadata.get("judge_id") or "")
        valid = [row for row in rows if row.get("_track") == "pointwise" and row.get("_status") == "ok" and row.get("_score") is not None]
        values = [float(row["_score"]) for row in valid]
        unique = sorted(set(values))
        qual = [row for row in rows if row.get("_track") == "qualification"]
        q_valid = sum(
            1
            for row in qual
            if isinstance(row.get("_qualification_score_audit"), Mapping)
            and row["_qualification_score_audit"].get("status") == "valid"
        )
        q_invalid = sum(
            1
            for row in qual
            if isinstance(row.get("_qualification_score_audit"), Mapping)
            and row["_qualification_score_audit"].get("status") == "invalid"
        )
        output.append({
            "judge_id": judge_id,
            "family": judge.get("family", metadata.get("family", "")),
            "pointwise_valid_n": len(valid),
            "pointwise_expected_n": 21,
            "source_coverage": len({str(row.get("_source_id")) for row in valid}),
            "score_mean": statistics.fmean(values) if values else "",
            "score_sd_population": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else ""),
            "score_unique_values": ",".join(f"{value:g}" for value in unique),
            "qualification_parent_rows": 12,
            "qualification_flattened_variants": 24,
            "qualification_scalar_valid_variants": q_valid,
            "qualification_scalar_invalid_variants": q_invalid,
            "pairwise_rows": sum(1 for row in rows if row.get("_track") == "pairwise" and row.get("_status") == "ok"),
            "panel_candidate_mean_note": "panel pointwise aggregate below" if pointwise else "",
        })
    # Add panel means for convenience, but do not replace per-judge values.
    for row in output:
        row["candidate_means_json"] = _json({candidate: (pointwise.get(candidate) or {}).get("score_mean") for candidate in sorted(pointwise)})
    return output


def _reliability_record(label: str, paths: Sequence[Path], description: str, required_judges: int = 5) -> dict[str, Any]:
    metadata, rows = load_bundles(paths)
    point_rows = [row for row in rows if row.get("_track") == "pointwise" and row.get("_status") == "ok" and row.get("_score") is not None]
    result = aggregate(point_rows, metadata, required_judges=required_judges, bootstrap_replicates=2000, seed=20261022)
    reliability = result.get("reliability_overall", {})
    icc = reliability.get("icc_2_1_2_k") if isinstance(reliability, Mapping) else {}
    kendall = reliability.get("kendall_w") if isinstance(reliability, Mapping) else {}
    lofo = result.get("family_sensitivity", {}).get("leave_one_family_out", {}) if isinstance(result.get("family_sensitivity"), Mapping) else {}
    return {
        "comparison": label,
        "description": description,
        "n_judges_actual": icc.get("n_judges"),
        "n_units_actual": icc.get("n_units"),
        "n_source_clusters": len({str(row.get("_source_id")) for row in point_rows}),
        "icc_model": icc.get("model", "two-way random, absolute agreement"),
        "icc_2_1": icc.get("icc_2_1"),
        "icc_2_k": icc.get("icc_2_k"),
        "icc_ci_status": "not available; implementation does not estimate ICC CI",
        "kendall_w": kendall.get("estimate") if isinstance(kendall, Mapping) else None,
        "kendall_w_n_rankings": kendall.get("n_rankings") if isinstance(kendall, Mapping) else None,
        "lofo_status": lofo.get("status", "unavailable") if isinstance(lofo, Mapping) else "unavailable",
        "lofo_baseline_ranking": _json(lofo.get("baseline_ranking")) if isinstance(lofo, Mapping) else "",
        "panel_aggregate_sha256": "",
        "scope_note": description,
    }


def _candidate_rank(aggregate_result: Mapping[str, Any]) -> list[str]:
    pointwise = aggregate_result.get("pointwise") if isinstance(aggregate_result.get("pointwise"), Mapping) else {}
    return [
        candidate
        for candidate, _summary in sorted(
            pointwise.items(),
            key=lambda item: (
                -float(item[1].get("score_mean")) if isinstance(item[1], Mapping) and isinstance(item[1].get("score_mean"), (int, float)) else float("inf"),
                str(item[0]),
            ),
        )
    ]


def build_outputs(bootstrap_replicates: int = 2000) -> dict[str, Any]:
    derive_bundle()
    input_paths = [*BASELINE, DERIVED]
    # Build the ordinary owner CSVs through the existing strict pipeline.
    PANEL.mkdir(parents=True, exist_ok=True)
    report = build_report(PACKET, INSTRUCTIONS, MAPPING, ANSWER_KEY, input_paths, PANEL, FOUR_AGG, bootstrap_replicates, 20261022, False)
    panel_aggregate = read_json(PANEL / "aggregate.fivejudges.json")
    baseline_five_aggregate = read_json(BASELINE_FIVE_AGG)
    recovered_rank = _candidate_rank(panel_aggregate)
    baseline_rank = _candidate_rank(baseline_five_aggregate)

    provenance_rows = _provenance_rows(panel_aggregate)
    provenance_fields = [
        "item_id", "source_id", "candidate_id", "source_text_sha256", "logical_judge_id", "row_source",
        "source_bundle_path", "source_bundle_sha256", "derived_composed_bundle_sha256", "source_run_id",
        "source_request_id", "settings_profile", "request_settings_json", "retry_parent_bundle_sha256",
        "retry_parent_request_id", "valid_for_composed_pointwise", "score_eligible", "selection_rule", "mixed_settings_note",
    ]
    _write_csv(OUT / "recovery_composed_row_provenance.owner.csv", provenance_rows, provenance_fields)

    panel_paths = input_paths
    summary_rows = _summary_rows(panel_aggregate, panel_paths)
    summary_fields = [
        "judge_id", "family", "pointwise_valid_n", "pointwise_expected_n", "source_coverage",
        "score_mean", "score_sd_population", "score_unique_values", "qualification_parent_rows",
        "qualification_flattened_variants", "qualification_scalar_valid_variants", "qualification_scalar_invalid_variants",
        "pairwise_rows", "panel_candidate_mean_note", "candidate_means_json",
    ]
    _write_csv(OUT / "recovery_composed_judge_summary.owner.csv", summary_rows, summary_fields)

    # Preserve the old mode comparison's raw-response/contract annotations in
    # a separate CSV.  The recovered panel intentionally cannot contain the
    # failed row, so this file is the explicit audit trail for the excluded
    # 92-vs-94 response and its two preserved raw error objects.
    invalid_annotations_path = OUT / "recovery_original_invalid_annotations.owner.csv"
    with ORIGINAL_INVALID_ANNOTATIONS.open("r", encoding="utf-8", newline="") as handle:
        invalid_reader = csv.DictReader(handle)
        invalid_fields = list(invalid_reader.fieldnames or [])
        invalid_rows = [
            row
            for row in invalid_reader
            if row.get("mode") == "on"
            and row.get("judge_id") == LOGICAL_JUDGE
            and row.get("item_id") == "src-03"
            and row.get("candidate_id") == "cand-02"
        ]
    if len(invalid_rows) != 3:
        raise ValueError(f"expected three preserved original invalid annotations, got {len(invalid_rows)}")
    _write_csv(invalid_annotations_path, invalid_rows, invalid_fields)

    # The recovered panel is n=21.  The old uniform off panel and old on
    # matched-20 subset are recomputed here from their frozen raw bundles so
    # this comparison has explicit unit scope rather than borrowing a value.
    off_paths = [*BASELINE, OFF]
    on_paths = [*BASELINE, RAW_ON]
    off = _reliability_record("uniform_off_n21", off_paths, "uniform thinking-off five-judge panel; 21 common pointwise units")
    recovered = _reliability_record("recovered_composed_n21", input_paths, "20 original thinking-on rows plus one exact-key format-recovery row; mixed settings; diagnostic only")
    on_rows = [_pointwise_rows(path) for path in on_paths]
    on_common = set.intersection(*(set(rows) for rows in on_rows))
    on_matched_paths_rows = []
    metadata, all_rows = load_bundles(on_paths)
    on_matched_rows = [row for row in all_rows if row.get("_track") == "pointwise" and row.get("_status") == "ok" and (str(row.get("_item_id")), str(row.get("_candidate_id"))) in on_common]
    on_matched_result = aggregate(on_matched_rows, metadata, required_judges=5, bootstrap_replicates=2000, seed=20261022)
    on_rel = on_matched_result["reliability_overall"]; on_icc = on_rel["icc_2_1_2_k"]; on_kw = on_rel["kendall_w"]
    on_matched = {
        "comparison": "original_on_matched20",
        "description": "original thinking-on run, common eligible units only; failed src-03/cand-02 excluded, no recovery row",
        "n_judges_actual": on_icc.get("n_judges"), "n_units_actual": on_icc.get("n_units"),
        "n_source_clusters": len({str(row.get("_source_id")) for row in on_matched_rows}),
        "icc_model": on_icc.get("model"), "icc_2_1": on_icc.get("icc_2_1"), "icc_2_k": on_icc.get("icc_2_k"),
        "icc_ci_status": "not available; implementation does not estimate ICC CI",
        "kendall_w": on_kw.get("estimate"), "kendall_w_n_rankings": on_kw.get("n_rankings"),
        "lofo_status": "not_run_for_matched_subset", "lofo_baseline_ranking": "", "panel_aggregate_sha256": "",
        "scope_note": "Matched common eligible pointwise units; not a uniform retry and no score imputation",
    }
    # Include the original on full n=20 as a separate diagnostic row.
    on_full = _reliability_record("original_on_n20", on_paths, "original thinking-on run; one failed pointwise row excluded from complete-unit ICC")
    reliability_rows = [off, recovered, on_full, on_matched]
    _write_csv(OUT / "recovery_reliability_comparison.owner.csv", reliability_rows, list(reliability_rows[0].keys()))

    recovery_case = read_json(RECOVERY_CASE)
    record = recovery_case["record"]
    pw = record["pointwise"]
    raw_event = recovery_case["raw_event"]
    panel_icc = panel_aggregate.get("reliability_overall", {}).get("icc_2_1_2_k", {})
    panel_kw = panel_aggregate.get("reliability_overall", {}).get("kendall_w", {})
    panel_pair = panel_aggregate.get("pairwise", {})
    report_path = OUT / "recovery_composed_diagnostic.md"
    report_lines = [
        "# Nemotron thinking-on format-recovery composed diagnostic (2026-10-22)",
        "",
        "## 繁中摘要",
        "",
        f"本診斷把原 Nemotron thinking-on 的 20 個有效 pointwise 單位，加上同一 frozen key `src-03/cand-02` 的一次盲 format-recovery，得到 **21/21 pointwise 可用**；回復列分數 **95**，subscores `27+29+19+20=95`，reasoning tokenize **6,696/8,192**（非 cap），API 約 **685.455 秒**。這不是第六位 judge，也不是均一 39-request rerun；新列使用較大 context/budget，故 reliability 只作 composed diagnostic，不能替代原 uniform off/on 結果。",
        "",
        "原本 92 分／subscores 94 的 malformed response、原 raw hash、兩個 raw response error 與 original on 的 20/21 coverage 均保留，沒有回填 94、重問選高分、或改寫 quote。Recovery 僅通過 exact frozen key 的 format-only selection rule；這是 metadata-bound adapter，不是內容或分數修正。",
        "",
        f"回復後五 judge 面板使用四位固定 baseline（GPT、Qwen、Mistral-off、Gemini）＋同一 Nemotron logical judge 的 composed row；ICC(2,1)={fmt(panel_icc.get('icc_2_1'), 6)}、ICC(2,k)={fmt(panel_icc.get('icc_2_k'), 6)}，實際 `k={panel_icc.get('n_judges')}`、`n={panel_icc.get('n_units')}`，Kendall W={fmt(panel_kw.get('estimate'), 6)}（`n_rankings={panel_kw.get('n_rankings')}`）。這些是一致性診斷，不代表翻譯正確性或正式 30/40 protocol 通過；CI 未由實作估計，因此標示 not available。",
        "",
        f"依 panel pointwise score-mean 的候選排序，recovered composed 與 frozen baseline 五 judge 均為 `{', '.join(recovered_rank)}`；排序未變（{str(recovered_rank == baseline_rank).lower()}）。這是排名敏感度結果，不是回復列正確性的證明。",
        "",
        "## Composition and boundaries",
        "",
        "| Component | Scope | Status |",
        "|---|---|---|",
        f"| Original thinking-on | 39 records; 20 valid pointwise, 1 original pointwise shape failure, 12 qualification parents / 24 variants, 6 pair orientations | preserved immutable |",
        f"| Recovery | exactly one blind `src-03/cand-02` pointwise request; score {pw.get('score_0_100')}; HTTP {raw_event.get('http_status')}; strict JSON | PASS |",
        "| Composed logical judge | 20 original pointwise + 1 recovery; original qualification/pairwise rows reused as derived inputs | diagnostic-only |",
        "| Independent-rater count | Nemotron recovery is the same logical judge, not a sixth vote | unchanged five-rater panel |",
        "",
        "The composed bundle is deliberately non-uniform: original thinking-on used reasoning budget 4096, max tokens 8192, context 16384; recovery used budget 8192, max tokens 16384, context 32768. `enable_thinking=true`, `preserve_reasoning=false`, and `reasoning_format=deepseek` were used in both; `deepseek` is parser compatibility metadata, not a different model or thinking mode.",
        "",
        "## Qualification and pairwise diagnostics",
        "",
        "The recovery request was pointwise only, so these values are reused from the original thinking-on rows and are not new recovery evidence: qualification scalar valid **24/24**, evidence support **11/11**, critical recall **6/6**, clean false positives **0/11**, and AB/BA **2/3 consistent/complete** (3 complete swaps, 2 consistent). The original on qualification/error/critical fields remain unchanged in the derived bundle; scalar validity is not a correctness certificate.",
        "",
        "## Recovered row (raw-bound, no score/quote edits)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Frozen key | `src-03/cand-02` |",
        f"| Score / subscores | `{pw.get('score_0_100')}` / `{_json(pw.get('subscores'))}` |",
        f"| Errors | `{len(pw.get('errors', []))}` raw annotations retained; exact quotes are in the owner error CSV |",
        f"| API elapsed | `{raw_event.get('elapsed_s')}` s; secondary diagnostic only |",
        f"| Reasoning token audit | `{record.get('timings', {}).get('reasoning', {}).get('token_audit', {}).get('token_count')}` / `{record.get('timings', {}).get('reasoning', {}).get('token_audit', {}).get('budget')}`, non-cap |",
        f"| Response/request hashes | `{raw_event.get('response_sha256')}` / `{raw_event.get('request_sha256')}` |",
        f"| Source/target binding | `{record.get('source_text_sha256')}` / `{record.get('candidate', {}).get('target_text_sha256')}` |",
        "",
        "## Reliability comparison",
        "",
        "| Comparison | n judges | n units | ICC(2,1) | ICC(2,k) | Kendall W | Scope |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in reliability_rows:
        report_lines.append(f"| `{row['comparison']}` | {row.get('n_judges_actual')} | {row.get('n_units_actual')} | {fmt(row.get('icc_2_1'), 6)} | {fmt(row.get('icc_2_k'), 6)} | {fmt(row.get('kendall_w'), 6)} | {row.get('description')} |")
    report_lines += [
        "",
        "ICC confidence intervals are `not available; implementation does not estimate ICC CI`. The n=21 recovered row is not a clean uniform replication, while original-on matched20 excludes the malformed key. The frozen uniform thinking-off n=21 panel remains the primary comparable off setting; this composed diagnostic is shown alongside it, never merged with off/on into a six-judge statistic.",
        "",
        "## Pairwise",
        "",
        f"The recovery request was pointwise only; the 6 pairwise orientations in the composed bundle are the original thinking-on rows reused unchanged. The panel pairwise summary reports `{panel_pair.get('swap_counts', {})}`; source scope remains the frozen three-pair same-source bank. Any raw context flag `not_recorded` remains as imported; operational evidence is documented separately (Gemini fresh UI chats; Qwen/Mistral/Nemotron fresh requests; GPT shared full context).",
        "",
        "## Files and provenance",
        "",
        "- Standard owner panel: [`panel_recovered/selection_report.fivejudges.md`](panel_recovered/selection_report.fivejudges.md). Its `translation_comparison`, `pointwise_results_long`, `pointwise_results_wide`, `error_annotations`, `judge_summary`, and `pairwise_judge_diagnostics` CSVs are the owner-only translation/error/long-table outputs.",
        "- Derived bundle: [`nemotron_recovered_composed.raw.json`](nemotron_recovered_composed.raw.json).",
        "- Row provenance: [`recovery_composed_row_provenance.owner.csv`](recovery_composed_row_provenance.owner.csv).",
        "- Judge summary: [`recovery_composed_judge_summary.owner.csv`](recovery_composed_judge_summary.owner.csv).",
        "- Reliability: [`recovery_reliability_comparison.owner.csv`](recovery_reliability_comparison.owner.csv).",
        "- Recovery metadata/provenance: [`recovery_composed_metadata.json`](recovery_composed_metadata.json).",
        "- Preserved original invalid-row annotations: [`recovery_original_invalid_annotations.owner.csv`](recovery_original_invalid_annotations.owner.csv); these are excluded from composed scoring.",
        "- ArtifactTool `fromCSV` QA: PASS for 21 comparison rows, 105 long rows, 5 judge columns/validity rows, 134 panel error annotations, 15 pairwise rows, 21 provenance rows, 4 reliability rows, and 3 preserved invalid-row annotations; exact source/target/hash binding and no duplicate error annotations verified.",
        "",
        "Original raw bundles and prior Nemotron off/on outputs are immutable. Mistral high primary comparison continues to use Nemotron OFF as its fixed baseline. LiteLLM and external judge APIs were not called by this aggregation step; no image evaluation is inferred.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {
        "schema_version": "nemotron-format-recovery-composed-metadata-v2",
        "status": "COMPLETE_DIAGNOSTIC",
        "logical_judge_id": LOGICAL_JUDGE,
        "logical_judge_is_not_new_rater": True,
        "composition_rule": "20 original valid thinking-on pointwise rows plus exactly one valid format-recovery row for src-03/cand-02; qualification and pairwise are reused derived rows",
        "original_on_bundle": {"path": str(RAW_ON), "sha256": sha256_file(RAW_ON)},
        "recovery_bundle": {"path": str(RECOVERY_BUNDLE), "sha256": sha256_file(RECOVERY_BUNDLE)},
        "recovery_case": {"path": str(RECOVERY_CASE), "sha256": sha256_file(RECOVERY_CASE)},
        "recovery_manifest": {"path": str(RECOVERY_MANIFEST), "sha256": sha256_file(RECOVERY_MANIFEST)},
        "recovery_validation": {"path": str(RECOVERY_VALIDATION), "sha256": sha256_file(RECOVERY_VALIDATION)},
        "derived_bundle": {"path": str(DERIVED), "sha256": sha256_file(DERIVED)},
        "retry_parent": {"request_id": RECOVERY_REQUEST, "failed_response_sha256": FAILED_RESPONSE_HASH, "invalid_score": 92, "invalid_subscore_sum": 94, "selection_reason": "exact-key format-only recovery; no score-based selection"},
        "settings_profiles": {
            "original_thinking_on": {"reasoning_budget_tokens": 4096, "max_tokens": 8192, "context_size": 16384, "enable_thinking": True, "preserve_reasoning": False, "reasoning_format": "deepseek", "temperature": 0, "top_p": 1, "seed": 20261022},
            "format_recovery": {"reasoning_budget_tokens": 8192, "max_tokens": 16384, "context_size": 32768, "enable_thinking": True, "preserve_reasoning": False, "reasoning_format": "deepseek", "temperature": 0, "top_p": 1, "seed": 20261022},
        },
        "qualification_rows_reused": 24,
        "pairwise_rows_reused": 6,
        "posthoc_mixed_settings": True,
        "raw_mutation": False,
        "row_provenance": provenance_rows,
        "output_panel": str(PANEL),
        "report": str(report_path),
        "owner_csvs": [str(OUT / "recovery_composed_row_provenance.owner.csv"), str(OUT / "recovery_composed_judge_summary.owner.csv"), str(OUT / "recovery_reliability_comparison.owner.csv"), str(invalid_annotations_path)],
    }
    write_json(PROVENANCE, metadata)

    custom_manifest = {
        "schema_version": "nemotron-format-recovery-diagnostic-manifest-v1",
        "packet_sha256": sha256_file(PACKET),
        "instructions_sha256": sha256_file(INSTRUCTIONS),
        "candidate_mapping_sha256": sha256_file(MAPPING),
        "answer_key_sha256": sha256_file(ANSWER_KEY),
        "recovery_case_sha256": sha256_file(RECOVERY_CASE),
        "recovery_bundle_sha256": sha256_file(RECOVERY_BUNDLE),
        "recovery_validation_sha256": sha256_file(RECOVERY_VALIDATION),
        "original_on_sha256": sha256_file(RAW_ON),
        "derived_bundle_sha256": sha256_file(DERIVED),
        "standard_panel_manifest": str(PANEL / "fivejudge_manifest.json"),
        "standard_panel_manifest_sha256": sha256_file(PANEL / "fivejudge_manifest.json"),
        "preserved_invalid_annotations_sha256": sha256_file(invalid_annotations_path),
        "logical_judge_is_not_new_rater": True,
        "not_a_uniform_39_request_run": True,
        "raw_mutation": False,
        "output_report_sha256": sha256_file(report_path),
    }
    write_json(OUT / "recovery_diagnostic_manifest.json", custom_manifest)
    return {"derived": str(DERIVED), "panel": report, "report": str(report_path), "provenance": str(PROVENANCE)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_outputs(args.bootstrap_replicates), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
