#!/usr/bin/env python3
"""Build the non-causal local-thinking combined diagnostic.

This report is deliberately separate from the frozen five-judge baseline and
the single-family swap reports.  It binds one vote per judge family while
keeping Nemotron's 20-original-plus-one-recovery composition visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n.a."
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n.a."
    return f"{100.0 * float(value):.{digits}f}%"


def nested(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def raw_request_binding(path: Path) -> dict[str, Any]:
    """Summarize exact local request payload/message bindings without storing CoT."""

    bundle = read_json(path)
    events = bundle.get("raw_events") if isinstance(bundle, Mapping) else None
    quality: dict[str, dict[str, Any]] = {}
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping) or event.get("status") != "ok":
                continue
            request_id = str(event.get("request_id") or "")
            if not request_id.startswith(("qualification", "pointwise", "pairwise")):
                continue
            request = event.get("request") if isinstance(event.get("request"), Mapping) else {}
            messages = request.get("messages")
            signature = {
                key: request.get(key)
                for key in (
                    "model", "seed", "temperature", "top_p", "max_tokens",
                    "reasoning_budget_tokens", "reasoning_effort", "reasoning_format",
                    "response_format", "chat_template_kwargs", "stream",
                )
            }
            quality[request_id] = {
                "messages_sha256": canonical_sha(messages),
                "message_count": len(messages) if isinstance(messages, list) else None,
                "role_sequence": [item.get("role") for item in messages if isinstance(item, Mapping)] if isinstance(messages, list) else None,
                "parameter_signature": signature,
            }
    message_map = {key: value["messages_sha256"] for key, value in sorted(quality.items())}
    parameter_hashes = sorted({canonical_sha(value["parameter_signature"]) for value in quality.values()})
    return {
        "quality_request_count": len(quality),
        "request_ids": sorted(quality),
        "message_hash_by_request_id": message_map,
        "message_hash_manifest_sha256": canonical_sha(message_map),
        "parameter_signature_hashes": parameter_hashes,
        "parameter_signature_count": len(parameter_hashes),
        "representative_request": quality[sorted(quality)[0]] if quality else None,
    }


def rank_candidates(pointwise: Mapping[str, Any]) -> list[str]:
    return sorted(pointwise, key=lambda candidate: (-float(pointwise[candidate].get("score_mean") or float("-inf")), str(candidate)))


def candidate_rows(combined: Mapping[str, Any], baseline: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    combined_point = combined.get("pointwise", {}) if isinstance(combined.get("pointwise"), Mapping) else {}
    baseline_point = baseline.get("pointwise", {}) if isinstance(baseline.get("pointwise"), Mapping) else {}
    combined_rank = {candidate: index + 1 for index, candidate in enumerate(rank_candidates(combined_point))}
    baseline_rank = {candidate: index + 1 for index, candidate in enumerate(rank_candidates(baseline_point))}
    rows: list[dict[str, Any]] = []
    for candidate in sorted(set(combined_point) | set(baseline_point)):
        info = mapping.get(candidate, {}) if isinstance(mapping, Mapping) else {}
        combined_mean = nested(combined_point.get(candidate, {}), "score_mean")
        baseline_mean = nested(baseline_point.get(candidate, {}), "score_mean")
        rows.append({
            "candidate_id": candidate,
            "model_alias": info.get("model_alias", ""),
            "candidate_family": info.get("candidate_family", ""),
            "combined_rank": combined_rank.get(candidate),
            "baseline_rank": baseline_rank.get(candidate),
            "rank_delta_combined_minus_baseline": (combined_rank[candidate] - baseline_rank[candidate]) if candidate in combined_rank and candidate in baseline_rank else None,
            "combined_mean": combined_mean,
            "baseline_mean": baseline_mean,
            "mean_delta_combined_minus_baseline": (combined_mean - baseline_mean) if isinstance(combined_mean, (int, float)) and isinstance(baseline_mean, (int, float)) else None,
            "combined_valid_scores": nested(combined_point.get(candidate, {}), "n_valid_scores"),
            "baseline_valid_scores": nested(baseline_point.get(candidate, {}), "n_valid_scores"),
        })
    return sorted(rows, key=lambda row: (row.get("combined_rank") or 999, row["candidate_id"]))


def qualification_row(qualification: Mapping[str, Any], judge_id: str) -> dict[str, Any]:
    value = qualification.get("per_judge", {}).get(judge_id, {}) if isinstance(qualification.get("per_judge"), Mapping) else {}
    coverage = value.get("coverage", {}) if isinstance(value.get("coverage"), Mapping) else {}
    evidence = value.get("mutation_evidence_supported_rate", {}) if isinstance(value.get("mutation_evidence_supported_rate"), Mapping) else {}
    critical = value.get("critical_severity_recall", {}) if isinstance(value.get("critical_severity_recall"), Mapping) else {}
    critical_union = critical.get("union_top_level_or_error_severity", {}) if isinstance(critical.get("union_top_level_or_error_severity"), Mapping) else {}
    false_positive = value.get("clean_false_positive_rate", {}) if isinstance(value.get("clean_false_positive_rate"), Mapping) else {}
    error_type = value.get("error_type_match", {}) if isinstance(value.get("error_type_match"), Mapping) else {}
    mutation = value.get("mutation_flag_rate_proxy", {}) if isinstance(value.get("mutation_flag_rate_proxy"), Mapping) else {}
    return {
        "qualification_coverage": coverage.get("available_variants"),
        "qualification_status": coverage.get("status"),
        "mutation_flag_proxy": mutation.get("estimate"),
        "evidence_supported": evidence.get("estimate"),
        "evidence_numerator": evidence.get("numerator"),
        "evidence_denominator": evidence.get("denominator"),
        "critical_recall": critical_union.get("estimate"),
        "critical_numerator": critical_union.get("numerator"),
        "critical_denominator": critical_union.get("denominator"),
        "clean_false_positive": false_positive.get("estimate"),
        "clean_false_positive_numerator": false_positive.get("numerator"),
        "clean_false_positive_denominator": false_positive.get("denominator"),
        "error_type_match": error_type.get("estimate"),
        "error_type_match_numerator": error_type.get("numerator"),
        "error_type_match_denominator": error_type.get("denominator"),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base.resolve()
    out = (base / args.output_dir).resolve()
    panel = out / "panel_combined"
    combined_aggregate_path = panel / "aggregate.fivejudges.json"
    baseline_aggregate_path = base / "fivejudge_20261022/aggregate.fivejudges.json"
    combined_qualification_path = panel / "qualification_summary.fivejudges.owner.json"
    summary_path = panel / "judge_summary.fivejudges.owner.csv"
    panel_manifest_path = panel / "fivejudge_manifest.json"
    combined = read_json(combined_aggregate_path)
    baseline = read_json(baseline_aggregate_path)
    qualification = read_json(combined_qualification_path)
    summaries = {row["judge_id"]: row for row in load_csv(summary_path)}
    mapping_doc = read_json(base / "private/candidate_mapping.private.json")
    mapping = mapping_doc.get("mapping", {}) if isinstance(mapping_doc, Mapping) else {}
    panel_manifest = read_json(panel_manifest_path)

    input_paths = {
        "gpt_fixed_cloud": base / "raw/gpt-6.1-sol-codex-blind.json",
        "gemini_fixed_cloud": base / "raw/gemini-3.8-flash-high-antigravity-v2.assembled.json",
        "qwen_thinking_xhigh": base / "raw/qwen3.8-flash-next-local-thinking-xhigh-20261022/qwen3.8-flash-next-local-thinking-xhigh-20261022.raw.json",
        "mistral_thinking_high": base / "raw/mistral-medium-3.5-128b-local-thinking-high-20261022/mistral-medium-3.5-128b-local-thinking-high-20261022.raw.json",
        "nemotron_composed_recovery": base / "nemotron_format_recovery_20261022/nemotron_recovered_composed.raw.json",
    }
    input_hashes = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in input_paths.items()}

    qwen_manifest_path = base / "raw/qwen3.8-flash-next-local-thinking-xhigh-20261022/raw_manifest.json"
    qwen_template_path = base / "raw/qwen3.8-flash-next-local-thinking-xhigh-20261022/template_hash_audit.json"
    mistral_manifest_path = base / "raw/mistral-medium-3.5-128b-local-thinking-high-20261022/mistral-medium-3.5-128b-local-thinking-high-20261022.manifest.json"
    mistral_addendum_path = base / "raw/mistral-medium-3.5-128b-local-thinking-high-20261022/template_evidence_addendum.json"
    nemo_recovery_manifest_path = base / "raw/nemotron-3.5-super-vl-120b-a12b-format-recovery-20261022/recovery_manifest_attempt-02.json"
    nemo_composed_metadata_path = base / "nemotron_format_recovery_20261022/recovery_composed_metadata.json"
    nemo_original_manifest_path = base / "raw/nemotron-3.5-super-vl-120b-a12b-local-thinking-on-20261022/nemotron_thinking_on_run_manifest.json"

    qwen_manifest = read_json(qwen_manifest_path)
    qwen_template = read_json(qwen_template_path)
    mistral_manifest = read_json(mistral_manifest_path)
    mistral_addendum = read_json(mistral_addendum_path)
    nemo_recovery_manifest = read_json(nemo_recovery_manifest_path)
    nemo_composed_metadata = read_json(nemo_composed_metadata_path)
    nemo_original_manifest = read_json(nemo_original_manifest_path)

    # Request message hashes are only derived from raw request metadata.  No
    # reasoning_content is copied into the report.
    request_bindings = {
        label: raw_request_binding(path)
        for label, path in input_paths.items()
        if label in {"qwen_thinking_xhigh", "mistral_thinking_high", "nemotron_composed_recovery"}
    }
    local_message_hashes = [value["message_hash_by_request_id"] for value in request_bindings.values()]
    common_message_binding = bool(local_message_hashes) and all(value == local_message_hashes[0] for value in local_message_hashes[1:])
    common_message_manifest_sha = canonical_sha(local_message_hashes[0]) if local_message_hashes else None

    reliability = combined.get("reliability_overall", {})
    baseline_reliability = baseline.get("reliability_overall", {})
    icc = reliability.get("icc_2_1_2_k", {})
    baseline_icc = baseline_reliability.get("icc_2_1_2_k", {})
    kendall = reliability.get("kendall_w", {})
    baseline_kendall = baseline_reliability.get("kendall_w", {})
    candidate_comparison = candidate_rows(combined, baseline, mapping)

    judge_labels = {
        "gpt-6.1-sol-codex": "GPT/Codex fixed cloud",
        "gemini-3.8-flash-high-antigravity-v2": "Gemini fixed cloud",
        "qwen3.8-flash-next-local-thinking-xhigh-20261022": "Qwen xhigh local thinking",
        "mistral-medium-3.5-128b-local-thinking-high-20261022": "Mistral high local thinking",
        "nemotron-3.5-super-vl-120b-a12b-local-thinking-on": "Nemotron thinking-on composed recovery",
    }
    judge_order = [
        "gpt-6.1-sol-codex",
        "gemini-3.8-flash-high-antigravity-v2",
        "qwen3.8-flash-next-local-thinking-xhigh-20261022",
        "mistral-medium-3.5-128b-local-thinking-high-20261022",
        "nemotron-3.5-super-vl-120b-a12b-local-thinking-on",
    ]

    judge_metrics: list[dict[str, Any]] = []
    for judge_id in judge_order:
        summary = summaries.get(judge_id, {})
        qmetrics = qualification_row(qualification, judge_id)
        judge_metrics.append({
            "judge_id": judge_id,
            "label": judge_labels[judge_id],
            "family": summary.get("judge_family"),
            "provider": summary.get("provider"),
            "pointwise_valid": int(summary.get("pointwise_valid_rows") or 0),
            "pointwise_rows": int(summary.get("pointwise_rows") or 0),
            "pointwise_mean": float(summary["pointwise_mean"]) if summary.get("pointwise_mean") not in (None, "") else None,
            "pointwise_sd_population": float(summary["pointwise_score_sd_population"]) if summary.get("pointwise_score_sd_population") not in (None, "") else None,
            "pointwise_unique_values": json.loads(summary.get("pointwise_score_unique_values") or "[]"),
            "pointwise_top_tie_items": int(summary.get("pointwise_top_tie_items") or 0),
            "pointwise_top_tie_slots": int(summary.get("pointwise_top_tie_slots") or 0),
            "pointwise_error_annotations": int(summary.get("pointwise_error_annotations") or 0),
            "pointwise_critical_flags": int(summary.get("pointwise_critical_flags") or 0),
            "qualification_scalar_valid": int(summary.get("qualification_scalar_valid") or 0),
            "qualification_scalar_invalid": int(summary.get("qualification_scalar_invalid") or 0),
            "qualification_scalar_status": summary.get("qualification_scalar_status"),
            "pairwise_complete": int(summary.get("pairwise_complete_swaps") or 0),
            "pairwise_consistent": int(summary.get("pairwise_consistent") or 0),
            "pairwise_swap_consistency": float(summary["pairwise_swap_consistency"]) if summary.get("pairwise_swap_consistency") not in (None, "") else None,
            "context_packaging": summary.get("judge_context_packaging"),
            "pointwise_latency_median_s": float(summary["pointwise_latency_median_s"]) if summary.get("pointwise_latency_median_s") not in (None, "") else None,
            **qmetrics,
        })

    lofo = nested(combined, "family_sensitivity", "leave_one_family_out", "leave_one_family_out", default={})
    lofo_rows = []
    for family, value in sorted(lofo.items() if isinstance(lofo, Mapping) else []):
        lofo_rows.append({
            "family_omitted": family,
            "ranking": value.get("ranking", []),
            "top1": value.get("top1"),
            "top1_changed": value.get("top1_changed"),
            "rank_delta_from_combined": value.get("rank_delta_from_baseline", {}),
        })

    template_audits = {
        "gpt_fixed_cloud": {"status": "frozen_baseline_not_retested", "evidence": "existing validated bundle; no local template gate in this diagnostic"},
        "gemini_fixed_cloud": {"status": "frozen_baseline_not_retested", "evidence": "existing validated assembled bundle; no local template gate in this diagnostic"},
        "qwen_thinking_xhigh": {
            "status": nested(qwen_manifest, "validation", "template_hash", default=qwen_template.get("status")),
            "strict_protocol": nested(qwen_manifest, "validation", "strict_protocol"),
            "template_hash_audit": qwen_template.get("status"),
            "template_sha256": qwen_template.get("template_sha256"),
            "manifest_sha256": sha256_file(qwen_manifest_path),
            "template_audit_sha256": sha256_file(qwen_template_path),
        },
        "mistral_thinking_high": {
            "status": nested(mistral_addendum, "decision", "current_run_standalone_template_verification", default=nested(mistral_manifest, "validation", "template_verification_status")),
            "prior_independent_audit": nested(mistral_addendum, "decision", "prior_independent_local_mistral_template_audit"),
            "strict_json_and_request_fields": nested(mistral_manifest, "validation", "strict_json_and_request_fields"),
            "manifest_sha256": sha256_file(mistral_manifest_path),
            "addendum_sha256": sha256_file(mistral_addendum_path),
        },
        "nemotron_composed_recovery": {
            "status": nested(nemo_recovery_manifest, "template", "status"),
            "recovery_validation": nested(nemo_recovery_manifest, "validation", "status"),
            "original_on_render_diagnostics": nested(nemo_original_manifest, "template_and_smoke_gate", "render_diagnostics", "status"),
            "recovery_manifest_sha256": sha256_file(nemo_recovery_manifest_path),
            "composed_metadata_sha256": sha256_file(nemo_composed_metadata_path),
            "not_uniform_39_request_run": nemo_composed_metadata.get("posthoc_mixed_settings") is True,
        },
    }

    metrics = {
        "schema_version": "local-thinking-combined-diagnostic-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_COMBINED_DIAGNOSTIC",
        "panel_definition": {
            "n_votes": 5,
            "n_model_families": 5,
            "local_thinking_judges": ["qwen", "mistral", "nemotron"],
            "cloud_fixed_judges": ["gpt", "gemini"],
            "same_family_one_vote": True,
            "not_a_pure_thinking_causal_comparison": True,
            "not_primary_uniform_panel": True,
            "source_clusters": 3,
            "pointwise_units": 21,
            "qualification_parents": 12,
            "qualification_variants": 24,
            "pairwise_orientations": 6,
            "ci_status": "unavailable; only 3 source clusters",
        },
        "no_raw_mutation": True,
        "no_api_calls_in_aggregation": True,
        "source_packet_sha256": panel_manifest.get("packet_sha256"),
        "source_instructions_sha256": panel_manifest.get("instructions_sha256"),
        "candidate_mapping_sha256": panel_manifest.get("candidate_mapping_sha256"),
        "answer_key_sha256": panel_manifest.get("answer_key_sha256"),
        "input_bundles": input_hashes,
        "template_audits": template_audits,
        "request_binding": {
            "local_quality_request_count_each": [value.get("quality_request_count") for value in request_bindings.values()],
            "local_messages_exact_same_by_request_id": common_message_binding,
            "local_common_message_manifest_sha256": common_message_manifest_sha,
            "local_request_ids_exact_same": len({tuple(value.get("request_ids", [])) for value in request_bindings.values()}) == 1 if request_bindings else False,
            "note": "The local bundles use the same frozen system/user message payloads for all 39 quality request IDs. Model name, chat-template kwargs, reasoning budget/effort, max_tokens, endpoint and context settings differ as recorded below. No reasoning_content is copied here.",
        },
        "judge_settings": {
            "gpt_fixed_cloud": "frozen prior run; API request settings not re-run here",
            "gemini_fixed_cloud": "frozen prior assembled UI bundle; request settings not re-run here",
            "qwen_thinking_xhigh": qwen_manifest.get("settings"),
            "mistral_thinking_high": mistral_manifest.get("fixed_settings"),
            "nemotron_composed_recovery": {
                "composition_rule": nemo_composed_metadata.get("composition_rule"),
                "parent_settings": nested(nemo_composed_metadata, "row_provenance", default=[])[0].get("request_settings_json") if isinstance(nested(nemo_composed_metadata, "row_provenance", default=[]), list) and nested(nemo_composed_metadata, "row_provenance", default=[]) else None,
                "recovery_manifest_protocol": nested(nemo_recovery_manifest, "protocol"),
                "mixed_settings": True,
            },
        },
        "panel_metrics": {
            "combined": {
                "icc_2_1": icc.get("icc_2_1"),
                "icc_2_k": icc.get("icc_2_k"),
                "k": icc.get("n_judges"),
                "n_units": icc.get("n_units"),
                "kendall_w": kendall.get("estimate"),
                "n_rankings": kendall.get("n_rankings"),
                "pairwise_coverage": combined.get("pairwise", {}).get("coverage"),
                "family_balanced_ranking": combined.get("family_sensitivity", {}).get("family_balanced_ranking"),
            },
            "baseline_frozen": {
                "icc_2_1": baseline_icc.get("icc_2_1"),
                "icc_2_k": baseline_icc.get("icc_2_k"),
                "k": baseline_icc.get("n_judges"),
                "n_units": baseline_icc.get("n_units"),
                "kendall_w": baseline_kendall.get("estimate"),
                "n_rankings": baseline_kendall.get("n_rankings"),
            },
            "differences_combined_minus_baseline": {
                "icc_2_1": (icc.get("icc_2_1") - baseline_icc.get("icc_2_1")) if isinstance(icc.get("icc_2_1"), (int, float)) and isinstance(baseline_icc.get("icc_2_1"), (int, float)) else None,
                "icc_2_k": (icc.get("icc_2_k") - baseline_icc.get("icc_2_k")) if isinstance(icc.get("icc_2_k"), (int, float)) and isinstance(baseline_icc.get("icc_2_k"), (int, float)) else None,
                "kendall_w": (kendall.get("estimate") - baseline_kendall.get("estimate")) if isinstance(kendall.get("estimate"), (int, float)) and isinstance(baseline_kendall.get("estimate"), (int, float)) else None,
            },
        },
        "judge_metrics": judge_metrics,
        "candidate_ranking_comparison": candidate_comparison,
        "family_lofo": lofo_rows,
        "qualification_macro": qualification.get("panel_descriptive_macro_mean"),
        "limitations": [
            "This panel swaps three local families at once relative to the frozen baseline and therefore cannot identify a pure thinking-on effect.",
            "Nemotron is a derived 20-original-thinking-on plus one exact-key format-recovery composition. It is not a uniform 39-request rerun and is not a sixth rater.",
            "Mistral current-run template verification remains PENDING_PROVENANCE_INCOMPLETE; the prior independent template audit is retained but not promoted.",
            "Only three source clusters are available. ICC/Kendall/LOFO are descriptive; no confidence interval is available and no correctness conclusion follows from higher agreement.",
            "Pairwise data are three fixed pairs from one source cluster with a disconnected candidate graph. No global pairwise ranking is claimed.",
            "Mistral targeted stability retest checkpoint is not included and cannot replace the original 39-request high run.",
        ],
    }

    metrics_path = out / "combined_metrics.json"
    manifest_path = out / "combined_diagnostic_manifest.json"
    report_path = out / "local_thinking_combined_diagnostic.md"
    artifacttool_qa_path = out / "artifacttool_qa.json"
    write_json(metrics_path, metrics)
    manifest = {
        "schema_version": "local-thinking-combined-diagnostic-manifest-v1",
        "metrics_sha256": sha256_file(metrics_path),
        "report_path": str(report_path),
        "report_sha256_after_write": None,
        "panel_dir": str(panel),
        "panel_manifest_sha256": sha256_file(panel_manifest_path),
        "panel_csvs": sorted(str(path) for path in panel.glob("*.csv")),
        "artifacttool_qa_path": str(artifacttool_qa_path) if artifacttool_qa_path.exists() else None,
        "artifacttool_qa_sha256": sha256_file(artifacttool_qa_path) if artifacttool_qa_path.exists() else None,
        "source_bundles": input_hashes,
        "recovery_manifest_sha256": sha256_file(nemo_recovery_manifest_path),
        "recovery_composed_metadata_sha256": sha256_file(nemo_composed_metadata_path),
        "raw_mutation": False,
        "no_llm_calls": True,
        "n_votes": 5,
        "same_family_one_vote": True,
        "non_uniform_nemotron": True,
        "ci_available": False,
    }
    write_json(manifest_path, manifest)

    lines: list[str] = []
    lines.extend([
        "# Local-thinking combined diagnostic（3 本地 thinking + 2 固定雲端）",
        "",
        "更新：2026-10-23（Asia/Taipei）",
        "",
        "> **定位：探索性診斷，不是正式 panel、不是真正的純 thinking 因果比較，也不取代 baseline 或 single-swap 結果。** 每個 judge family 只計一票：GPT/Codex 與 Gemini 沿用既有 frozen 結果；Qwen 使用 xhigh thinking；Mistral 使用原始 uniform high 39-request run；Nemotron 使用 20 個原始 thinking-on pointwise valid rows 加 1 個 exact-key format-recovery row 的 derived composition。Nemotron 因而是 mixed-settings，不能稱 uniform 39-request rerun，也不是第六位 judge。",
        "",
        "## 繁中摘要",
        "",
        f"此合併診斷仍是 `{icc.get('n_units', 21)} 個 point units × {icc.get('n_judges', 5)} 票`、3 個 source clusters、12 個 qualification parents／24 variants、3 pairs × AB/BA。ICC(2,1)={fmt(icc.get('icc_2_1'), 6)}、ICC(2,5)={fmt(icc.get('icc_2_k'), 6)}、Kendall W={fmt(kendall.get('estimate'), 6)}；CI **不可用**，不能把 agreement 當 correctness。相對 frozen baseline 的 .313362/.695294/W .640146，這個非單因子組合變為 .469662/.815768/W .715111，只能描述為敏感度差異。",
        "",
        f"候選排序維持：`{' > '.join(candidate['model_alias'] or candidate['candidate_id'] for candidate in candidate_comparison)}`；排序未因本診斷組合改變，但候選均分有變動。LOFO 仍只有 family omission sensitivity，且每 family 僅一個 judge；pairwise graph 仍 disconnected，不提供全 7-candidate pairwise ranking。",
        "",
        "品質判讀優先看 evidence／critical／clean false-positive、相互一致與合理同分，不以分數變異或 ICC 單獨挑選模式。Mistral high 的 current-run template status 仍是 `PENDING_PROVENANCE_INCOMPLETE`；Qwen template/strict protocol PASS；Nemotron recovery case PASS，但 composed panel 保留 mixed-settings 限制。Mistral targeted stability checkpoint 不在本次 panel，未取代原始 39-request high 結果。",
        "",
        "## Panel binding and settings",
        "",
        "| Vote | Family / operation | Raw / derived source | Thinking / settings evidence | Template status |",
        "|---|---|---|---|---|",
        "| GPT/Codex | `openai` / frozen cloud baseline | frozen raw bundle | prior full-packet context; not re-run here | frozen baseline, no local template gate |",
        "| Gemini | `google-gemini` / frozen cloud baseline | frozen assembled bundle | prior UI operation; not re-run here | frozen baseline, no local template gate |",
        "| Qwen | `qwen` / local xhigh | full 39-record raw | `enable_thinking=true`, `preserve_thinking=false`, `reasoning_effort=xhigh`, reasoning budget/max output 8192/16384, context 32768, seed 20261022 | `PASS` strict protocol + template hash |",
        "| Mistral | `mistral` / local high | full 39-record raw | thinking high, `preserve_reasoning=false`, reasoning budget/max output 8192/16384, context 32768, seed 20261022 | `PENDING_PROVENANCE_INCOMPLETE`; prior independent audit PASS is not promoted |",
        "| Nemotron | `nvidia-nemotron` / derived local recovery | 20 old thinking-on + 1 recovery, qualification/pairwise reused | parent reasoning budget 4096 / max output 8192 / context 16384; recovery budget 8192 / max output 16384 / context 32768; seed 20261022 | recovery template/validation PASS; composed panel mixed-settings |",
        "",
        "The local quality-request payloads use the same frozen message bodies for all 39 request IDs. Model names, reasoning parameters, context limits, endpoints, and Nemotron recovery settings differ. The exact binding is recorded as hashes without copying chain-of-thought text.",
        "",
        "## Judge-level quality and discrimination diagnostics",
        "",
        "`valid/rows` and qualification scalar validity are separate. Error-annotation counts are raw owner annotations, not correctness labels. Score SD/unique values describe scale compression or spread only.",
        "",
        "| Judge | Point mean | SD(pop) | Unique | Point errors / critical | Qualification scalar | Evidence support | Critical recall | Clean FP | ABBA 一致／完整 | Point latency median (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in judge_metrics:
        lines.append(
            f"| {row['label']} | {fmt(row['pointwise_mean'], 3)} | {fmt(row['pointwise_sd_population'], 3)} | {len(row['pointwise_unique_values'])} | {row['pointwise_error_annotations']} / {row['pointwise_critical_flags']} | {row['qualification_scalar_valid']}/{row['qualification_scalar_valid'] + row['qualification_scalar_invalid']} | {pct(row['evidence_supported'])} ({row['evidence_numerator']}/{row['evidence_denominator']}) | {pct(row['critical_recall'])} ({row['critical_numerator']}/{row['critical_denominator']}) | {pct(row['clean_false_positive'])} ({row['clean_false_positive_numerator']}/{row['clean_false_positive_denominator']}) | {row['pairwise_consistent']}/{row['pairwise_complete']} | {fmt(row['pointwise_latency_median_s'], 3)} |"
        )
    lines.extend([
        "",
        "Mistral token audit recorded 3 boundary-adjacent reasoning counts at/over the nominal 8192 budget; the audit treats tokenizer/tag offsets conservatively and does not establish native-cap failure. Qwen token audit had 39/39 successful offline tokenizations with no cap candidates. Nemotron composed rows mix the parent and recovery budgets by design.",
        "",
        "## Panel reliability and baseline sensitivity",
        "",
        "| Panel | ICC(2,1) | ICC(2,k) | k | n units | Kendall W | n rankings | CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| Frozen baseline | {fmt(baseline_icc.get('icc_2_1'), 6)} | {fmt(baseline_icc.get('icc_2_k'), 6)} | {baseline_icc.get('n_judges')} | {baseline_icc.get('n_units')} | {fmt(baseline_kendall.get('estimate'), 6)} | {baseline_kendall.get('n_rankings')} | unavailable |",
        f"| Combined diagnostic | {fmt(icc.get('icc_2_1'), 6)} | {fmt(icc.get('icc_2_k'), 6)} | {icc.get('n_judges')} | {icc.get('n_units')} | {fmt(kendall.get('estimate'), 6)} | {kendall.get('n_rankings')} | unavailable |",
        "",
        f"Difference combined minus baseline: ICC(2,1) {fmt(metrics['panel_metrics']['differences_combined_minus_baseline']['icc_2_1'], 6)}, ICC(2,k) {fmt(metrics['panel_metrics']['differences_combined_minus_baseline']['icc_2_k'], 6)}, Kendall W {fmt(metrics['panel_metrics']['differences_combined_minus_baseline']['kendall_w'], 6)}. The combined panel changes Qwen, Mistral, and Nemotron simultaneously, so these deltas are not a causal effect estimate.",
        "",
        "### Candidate ranking and coverage",
        "",
        "| Candidate | Family | Baseline rank / mean | Combined rank / mean | Rank delta | Mean delta | Combined valid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in candidate_comparison:
        lines.append(f"| {row['model_alias']} (`{row['candidate_id']}`) | `{row['candidate_family']}` | {row['baseline_rank']} / {fmt(row['baseline_mean'], 2)} | {row['combined_rank']} / {fmt(row['combined_mean'], 2)} | {fmt(row['rank_delta_combined_minus_baseline'])} | {fmt(row['mean_delta_combined_minus_baseline'], 2)} | {row['combined_valid_scores']} |")
    lines.extend([
        "",
        "All five votes cover 21/21 point units in this composed panel. This is not an imputation statement: the Nemotron derived bundle has one valid recovery row selected by exact request key, while its invalid original 92-versus-94 response remains preserved in the recovery annotations and is not backfilled.",
        "",
        "## Qualification, pairwise, and LOFO",
        "",
        f"Qualification panel macro: evidence supported {pct(nested(qualification, 'panel_descriptive_macro_mean', 'mutation_evidence_supported_rate'))}, critical recall {pct(nested(qualification, 'panel_descriptive_macro_mean', 'critical_severity_recall_union'))}, clean FP {pct(nested(qualification, 'panel_descriptive_macro_mean', 'clean_false_positive_rate'))}; these are owner diagnostics with hidden key, not translation correctness. All five combined bundles report 24/24 scalar-valid variants; this does not make the protocol a formal 30/40-item panel.",
        "",
        "| Omitted family | Ranking after omission | Top-1 changed |",
        "|---|---|---|",
    ])
    for row in lofo_rows:
        lines.append(f"| `{row['family_omitted']}` | `{' > '.join(row['ranking'])}` | {row['top1_changed']} |")
    lines.extend([
        "",
        "Pairwise scope is 3 fixed pairs × AB/BA from one source cluster. The combined graph has 4 connected components and does not cover cand-07 in a pair, so selected component outcomes only are valid. Raw context flags remain in `pairwise_judge_diagnostics.fivejudges.owner.csv`; known operational evidence is separate from an absent raw boolean.",
        "",
        "## Exact evidence and output files",
        "",
        f"- Standard panel CSVs: [`panel_combined/translation_comparison.fivejudges.owner.csv`](panel_combined/translation_comparison.fivejudges.owner.csv), [`panel_combined/pointwise_results_long.fivejudges.owner.csv`](panel_combined/pointwise_results_long.fivejudges.owner.csv), [`panel_combined/pointwise_results_wide.fivejudges.owner.csv`](panel_combined/pointwise_results_wide.fivejudges.owner.csv), [`panel_combined/judge_summary.fivejudges.owner.csv`](panel_combined/judge_summary.fivejudges.owner.csv), [`panel_combined/judge_validity.fivejudges.owner.csv`](panel_combined/judge_validity.fivejudges.owner.csv), [`panel_combined/error_annotations.fivejudges.owner.csv`](panel_combined/error_annotations.fivejudges.owner.csv), [`panel_combined/pairwise_judge_diagnostics.fivejudges.owner.csv`](panel_combined/pairwise_judge_diagnostics.fivejudges.owner.csv).",
        f"- Aggregate: [`panel_combined/aggregate.fivejudges.json`](panel_combined/aggregate.fivejudges.json), ICC and Kendall values above are read from this aggregate. No ICC confidence interval was produced.",
        f"- Provenance: [`combined_metrics.json`](combined_metrics.json) and [`combined_diagnostic_manifest.json`](combined_diagnostic_manifest.json). Recovery composition provenance remains [`../nemotron_format_recovery_20261022/recovery_composed_metadata.json`](../nemotron_format_recovery_20261022/recovery_composed_metadata.json).",
        "- Existing baseline, Qwen single-swap, Mistral single-swap, Nemotron original-on, and Nemotron recovery outputs were not overwritten. Mistral targeted stability checkpoint is pending and excluded.",
        "- No new judge/API/LiteLLM call was made while building this diagnostic.",
        "",
        "Method reference: ICC interpretation should not be treated as a binary pass gate; see Koo & Li (2016), [Guideline for selecting and reporting intraclass correlation coefficients](https://pubmed.ncbi.nlm.nih.gov/27330520/). Local values above are from the frozen pilot aggregate, not from that paper.",
        "",
        "## Limitations",
        "",
        "- Three source clusters are insufficient for stable source-cluster uncertainty. CI is unavailable.",
        "- Agreement/reliability is not judge correctness. Do not select a mode solely because ICC, W, or score variance is higher.",
        "- Nemotron composition is mixed budget/context and includes one post-hoc format-only recovery at the frozen key. It is diagnostic only.",
        "- Different local settings and fixed cloud baselines remain a context/operation comparison, not a controlled thinking-only experiment.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["report_sha256_after_write"] = sha256_file(report_path)
    write_json(manifest_path, manifest)
    print(json.dumps({"report": str(report_path), "metrics": str(metrics_path), "manifest": str(manifest_path), "panel": str(panel)}, ensure_ascii=False, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("translation_benchmark_20261022_judge_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_thinking_combined_diagnostic_20261023"))
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
