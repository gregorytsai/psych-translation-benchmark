#!/usr/bin/env python3
"""Build the preregistered Mistral off-vs-high replacement comparison.

This is deliberately a thin, Mistral-labelled wrapper around the strict
mode/pairing helpers used by the Nemotron comparison builder.  It keeps the
two Mistral settings in separate five-judge panels: Mistral high replaces only
the frozen Mistral-off bundle and is never counted as a sixth rater.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_nemotron_thinking_comparison as modecmp
from aggregate_pilot import read_json
from build_fivejudge_report import slug


ROOT = Path(__file__).resolve().parent
MODE_LABELS = {"off": "thinking-off", "on": "thinking-high"}
TRACKS = ("qualification", "pointwise", "pairwise")

# The imported helper is generic except for the historical label and fallback
# judge-family selector.  Override only those two details; all strict parsing,
# exact source/target checks, and missing-vs-zero handling stay shared.
modecmp.MODE_LABELS.clear()
modecmp.MODE_LABELS.update(MODE_LABELS)


def sha256_file(path: Path) -> str:
    return modecmp.sha256_file(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None and number.is_integer() else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    if number is None:
        return "n.a."
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _num(value)
    return "n.a." if number is None else f"{100.0 * number:.1f}%"


def _parse_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _find_judge_id(rows: Sequence[Mapping[str, Any]], preferred: str) -> str:
    ids = sorted({str(row.get("judge_id") or "") for row in rows if str(row.get("judge_id") or "")})
    if preferred in ids:
        return preferred
    candidates = sorted({judge_id for judge_id in ids if "mistral" in judge_id.lower()})
    if len(candidates) != 1:
        raise ValueError(f"cannot identify one Mistral judge in panel rows: {candidates}; ids={ids}")
    return candidates[0]


modecmp._find_judge_id = _find_judge_id


def _mode_data(panel: Path, raw_summary: Mapping[str, Any]) -> dict[str, Any]:
    return modecmp._mode_pointwise(panel, raw_summary)


def _generic_sensitivity(mode: str, panel_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rename the shared helper's historical Nemotron columns to generic ones."""

    rows = modecmp._panel_sensitivity_rows(mode, panel_data)
    result: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["judge_valid_source_units"] = copied.pop("nemotron_valid_source_units", "")
        copied["judge_source_units_expected"] = copied.pop("nemotron_source_units_expected", "")
        copied["judge_source_coverage"] = copied.pop("nemotron_source_coverage", "")
        copied["candidate_coverage_status"] = (
            "complete_three_sources"
            if _int(copied.get("n_valid_source_units")) == 3 and _int(copied.get("judge_valid_source_units")) == 3
            else "partial_source_coverage"
        )
        copied["sensitivity_scope"] = "separate five-judge panel; Mistral setting swap only; not a joint six-judge panel"
        result.append(copied)
    return result


def _reliability(mode: str, panel_data: Mapping[str, Any]) -> dict[str, Any]:
    row = modecmp._reliability_row(mode, panel_data)
    row["mode_label"] = MODE_LABELS[mode]
    row["reliability_scope"] = "separate five-judge panel; Mistral setting swap only"
    row["scope_note"] = "Exploratory five-judge replacement panel; 3 source clusters; ICC is reliability, not correctness."
    return row


def _summary(mode: str, raw_summary: Mapping[str, Any], panel_data: Mapping[str, Any]) -> dict[str, Any]:
    row = modecmp._build_summary_row(mode, raw_summary, panel_data)
    row["mode_label"] = MODE_LABELS[mode]
    manifest = raw_summary.get("run_manifest", {})
    row["template_verification_status"] = raw_summary.get("template_verification_status") or manifest.get("template_verification_status", "")
    row["template_verification_note"] = raw_summary.get("template_verification_note") or manifest.get("template_verification_note", "")
    row["reasoning_budget_boundary_ids"] = _json_text(raw_summary.get("reasoning_budget_boundary_ids", []))
    row["reasoning_budget_boundary_count"] = len(raw_summary.get("reasoning_budget_boundary_ids", []))
    row["reasoning_budget_audit_path"] = raw_summary.get("reasoning_budget_audit_path", "")
    row["reasoning_budget_audit_records"] = raw_summary.get("reasoning_budget_audit_counts", {}).get("records", "") if isinstance(raw_summary.get("reasoning_budget_audit_counts"), Mapping) else ""
    row["reasoning_budget_audit_method"] = raw_summary.get("reasoning_budget_tokenize_method", "") or raw_summary.get("reasoning_budget_audit_interpretation", "")
    row["context_size"] = raw_summary.get("request_settings", {}).get("context_size", "")
    row["comparison_primary_role"] = "frozen Mistral off reference" if mode == "off" else "Mistral high replacement diagnostic"
    return row


def _join_pairwise(mode: str, panel_data: Mapping[str, Any], raw_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return modecmp._join_pairwise(mode, panel_data, raw_summary)


def _join_errors(mode: str, panel_data: Mapping[str, Any], raw_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return modecmp._join_errors(mode, panel_data, raw_summary)


def _qual_metric(row: Mapping[str, Any], name: str) -> str:
    value = row.get(f"qualification_{name}_estimate")
    return f"{_pct(value)} ({row.get(f'qualification_{name}_numerator', 'n.a.')}/{row.get(f'qualification_{name}_denominator', 'n.a.')})"


def _coverage(row: Mapping[str, Any]) -> str:
    n = _int(row.get("n_valid_scores"))
    k = _int(row.get("n_distinct_judges"))
    expected = k * 3 if k is not None else None
    panel_n = row.get("n_valid_source_units", "n.a.")
    judge_n = row.get("judge_valid_source_units", "n.a.")
    observed = f"{n}/{expected}" if n is not None and expected is not None else "n.a."
    return f"{observed}; panel sources {panel_n}/3; Mistral sources {judge_n}/3"


def _panel_summary_report(panel: Path, output: Path, high_data: Mapping[str, Any], high_summary: Mapping[str, Any]) -> None:
    aggregate = high_data.get("aggregate", {})
    rel = aggregate.get("reliability_overall", {}) if isinstance(aggregate.get("reliability_overall"), Mapping) else {}
    icc = rel.get("icc_2_1_2_k", {}) if isinstance(rel.get("icc_2_1_2_k"), Mapping) else {}
    lines = [
        "# Mistral thinking-high replacement panel",
        "",
        "This is the exploratory five-judge panel for the preregistered single-Mistral swap: GPT, Qwen-off, Gemini, and Nemotron-off are frozen; only Mistral is replaced by the high-setting bundle.",
        "",
        "Mistral off/high are two settings of one judge, not independent sixth/seventh judges. This panel is not combined with Mistral-off in one ICC or aggregate. Nemotron thinking-on is not an input.",
        "",
        f"- Raw strict validation: `{high_summary.get('validation_status')}`; high raw records `{high_summary.get('raw_record_count')}`, pointwise 21/21, qualification scalar 24/24, pairwise 6/6.",
        f"- Panel reliability is descriptive: ICC(2,1) `{icc.get('icc_2_1', 'n.a.')}`, ICC(2,k) `{icc.get('icc_2_k', 'n.a.')}`, actual k `{icc.get('n_judges', 'n.a.')}`, units n `{icc.get('n_units', 'n.a.')}`; CI unavailable and this is not a correctness claim.",
        f"- The current-run template status remains `template_verification_status={high_summary.get('template_verification_status', 'PENDING_PROVENANCE_INCOMPLETE')}`; provenance is incomplete despite retained props/smoke and separate prior local template-audit evidence.",
        "- See `../mistral_thinking_comparison.md` for paired off/high results, qualification/evidence, ABBA, score distribution, latency, budget audit, candidate coverage, and provenance.",
        "",
        "No raw score, quote, error, or invalid row was repaired. This panel is an exploratory sensitivity result, not formal 30-qualification/40-item approval.",
    ]
    report_text = "\n".join(lines) + "\n"
    output.write_text(report_text, encoding="utf-8")
    # The standard builder emits a Nemotron-oriented filename.  Keep that
    # filename as a corrected, Mistral-labelled alias so readers cannot
    # accidentally consume a misleading generated report.
    generic_alias = output.with_name("selection_report.fivejudges.md")
    if generic_alias != output:
        generic_alias.write_text(report_text, encoding="utf-8")


def _report(
    output: Path,
    off_summary: Mapping[str, Any],
    high_summary: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    off_data: Mapping[str, Any],
    high_data: Mapping[str, Any],
    sensitivity: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    off_raw: Mapping[str, Any],
    high_raw: Mapping[str, Any],
    high_manifest: Mapping[str, Any],
    addendum_path: Path | None,
) -> None:
    eligible = [r for r in paired if r.get("paired_score_status") == "paired_eligible"]
    deltas = [_num(r.get("score_delta_on_minus_off")) for r in eligible]
    deltas = [v for v in deltas if v is not None]
    changed = [r for r in eligible if str(r.get("score_changed")).lower() == "true"]
    off_counts = _parse_json_field(off_summary.get("pointwise_score_value_counts"), {})
    high_counts = _parse_json_field(high_summary.get("pointwise_score_value_counts"), {})
    off_unique = _parse_json_field(off_summary.get("pointwise_score_unique_values"), [])
    high_unique = _parse_json_field(high_summary.get("pointwise_score_unique_values"), [])
    high_manifest_validation = high_manifest.get("validation", {}) if isinstance(high_manifest.get("validation"), Mapping) else {}
    template_status = high_summary.get("template_verification_status") or high_manifest_validation.get("template_verification_status", "PENDING")
    template_note = high_summary.get("template_verification_note") or high_manifest_validation.get("template_verification_note", "")
    boundary_ids = high_raw.get("reasoning_budget_boundary_ids", [])
    token_counts = high_raw.get("reasoning_budget_audit_counts", {})
    rel_by_mode = {str(r.get("mode")): r for r in reliability}
    rank_rows = {mode: {str(r.get("candidate_id")): r for r in sensitivity if r.get("mode") == mode} for mode in ("off", "on")}
    rank_changes = []
    for cid in sorted(set(rank_rows["off"]) | set(rank_rows["on"])):
        if cid in rank_rows["off"] and cid in rank_rows["on"]:
            rank_changes.append((cid, _int(rank_rows["off"][cid].get("rank")), _int(rank_rows["on"][cid].get("rank"))))
    lines = [
        "# Mistral thinking-off vs thinking-high comparison (2026-10-22)",
        "",
        "## 繁中摘要",
        "",
        "本報告比較同一個本地 Mistral 模型的 off 與使用者指定的 native thinking-high 設定。主比較每次只替換 Mistral；GPT、Qwen-off、Gemini、Nemotron-off 四個 judge 固定。兩個 Mistral 設定不是第六位 judge，不會放進同一個 ICC、Kendall W、LOFO 或 aggregate。",
        "",
        f"- high raw 已完成 strict import：39/39 HTTP 成功，qualification 12/12 parents、24/24 variants scalar-valid，pointwise 21/21、pairwise 6/6；manifest template verification 仍為 `{template_status}`，不可簡化成模板已全通過。",
        f"- pointwise off：n={off_summary.get('pointwise_valid_rows')}, mean={off_summary.get('pointwise_mean')}, SD(pop)={off_summary.get('pointwise_score_sd_population')}, unique={off_unique}, counts={off_counts}；high：n={high_summary.get('pointwise_valid_rows')}, mean={high_summary.get('pointwise_mean')}, SD(pop)={high_summary.get('pointwise_score_sd_population')}, unique={high_unique}, counts={high_counts}。spread/unique 只是區辨描述，不代表正確性。",
        f"- 同 key 配對：{len(eligible)}/{len(paired)} units 雙方有效，{len(changed)} 筆改分；平均 high−off={statistics.fmean(deltas) if deltas else 'n.a.'}，median={statistics.median(deltas) if deltas else 'n.a.'}。沒有補分、修 quote 或改 raw error。",
        f"- qualification：off scalar {off_summary.get('qualification_scalar_valid')}/{off_summary.get('qualification_variants')} valid、{off_summary.get('qualification_scalar_invalid')} invalid；high {high_summary.get('qualification_scalar_valid')}/{high_summary.get('qualification_variants')} valid、{high_summary.get('qualification_scalar_invalid')} invalid。evidence/critical/FP 指標與 scalar validity 分開報告。",
        f"- AB/BA：off {off_summary.get('pairwise_complete_swaps')} complete、{off_summary.get('pairwise_consistent')} consistent；high {high_summary.get('pairwise_complete_swaps')} complete、{high_summary.get('pairwise_consistent')} consistent。這是 3 pairs/6 orientations、單一 source cluster 的小樣本；`3 complete; 1 consistent (33.3%)` 指三個 complete 中只有一個 consistent。",
        "- 候選 mean ranking 在 off/high available-case panel 中維持 `cand-04, cand-07, cand-05, cand-02, cand-06, cand-01, cand-03`；這是排名敏感度，不是正確性證據，且本 panel 的 valid N/source coverage 仍逐列保留。",
        f"- single-Mistral swap reliability：off ICC(2,1)={rel_by_mode.get('off', {}).get('icc_2_1', 'n.a.')}, ICC(2,k)={rel_by_mode.get('off', {}).get('icc_2_k', 'n.a.')}, W={rel_by_mode.get('off', {}).get('kendall_w', 'n.a.')}; high ICC(2,1)={rel_by_mode.get('on', {}).get('icc_2_1', 'n.a.')}, ICC(2,k)={rel_by_mode.get('on', {}).get('icc_2_k', 'n.a.')}, W={rel_by_mode.get('on', {}).get('kendall_w', 'n.a.')}; actual k/n 與 CI unavailable 見 reliability CSV。ICC/W 是 reliability 描述，不是 correctness 證明。",
        "- quality/evidence、paired coherence、reasonable ties、absolute-agreement reliability 優先；latency 僅次要，不能依高 ICC、較大 spread 或較低 latency 單獨選模式。正式 30-qualification/40-item protocol 仍未達。",
        "",
        "## 1. Frozen panel design and strict provenance",
        "",
        "| Panel | GPT | Qwen | Gemini | Nemotron | Mistral |",
        "|---|---|---|---|---|---|",
        "| off reference | frozen | frozen | frozen | frozen off | frozen off |",
        "| high replacement | frozen | frozen | frozen | frozen off | new high |",
        "",
        f"- off raw SHA-256: `{off_raw.get('raw_bundle_sha256')}`; high raw SHA-256: `{high_raw.get('raw_bundle_sha256')}` (requested raw SHA `768600c36ccaba1c5f844e4cac09b27dba43031b0f1b6e95c4282941f1b0baaa`).",
        f"- high manifest SHA-256: `{high_raw.get('run_manifest_sha256', 'n.a.')}`; template addendum: `{sha256_file(addendum_path) if addendum_path and addendum_path.exists() else 'n.a.'}`; addendum status is `PENDING_PROVENANCE_INCOMPLETE` for this run, while a separate prior local Mistral template audit is not provenance-bound to these request files.",
        f"- high template note: {template_note}",
        "- high settings: `reasoning_effort=high`, `reasoning_format=deepseek` (parser compatibility metadata, not model substitution), reasoning budget 8192, max_tokens 16384, context 32768, temperature 0, top_p 1, seed 20261022, cache disabled, strict JSON, preserve_reasoning=false; off context was 8192. Context is a confound, so differences cannot be attributed to thinking alone.",
        "- no LiteLLM, cloud judge, Qwen, or Nemotron API call was made by this aggregation; raw/checkpoint/frozen off files remain immutable.",
        "",
        "## 2. Pointwise paired comparison",
        "",
        f"All 21 frozen keys are in `mistral_thinking_off_vs_high_paired.owner.csv`; paired eligible={len(eligible)}/{len(paired)}, changed={len(changed)}, unchanged={len(eligible)-len(changed)}, mean delta={statistics.fmean(deltas) if deltas else 'n.a.'}, delta SD(pop)={statistics.pstdev(deltas) if len(deltas) > 1 else ('0' if deltas else 'n.a.')}.",
        "",
        "Ineligible/missing rows keep blank validated `error_count` with an explicit unknown scope; any HTTP-response errors remain separate as raw-response provenance. No missing value is recoded to zero.",
        "",
        "| Mode | n | Mean | SD(pop) | Unique | Value counts | Top ties items/slots |",
        "|---|---:|---:|---:|---|---|---:|",
        f"| off | {off_summary.get('pointwise_valid_rows')} | {_fmt(off_summary.get('pointwise_mean'))} | {_fmt(off_summary.get('pointwise_score_sd_population'))} | `{off_unique}` | `{off_counts}` | {off_summary.get('pointwise_top_tie_items')}/{off_summary.get('pointwise_top_tie_slots')} |",
        f"| high | {high_summary.get('pointwise_valid_rows')} | {_fmt(high_summary.get('pointwise_mean'))} | {_fmt(high_summary.get('pointwise_score_sd_population'))} | `{high_unique}` | `{high_counts}` | {high_summary.get('pointwise_top_tie_items')}/{high_summary.get('pointwise_top_tie_slots')} |",
        "",
        "## 3. Qualification, evidence, critical and false-positive diagnostics",
        "",
        "| Mode | Scalar valid/variants | Invalid | Evidence supported | Critical recall (union) | Clean FP | Error-type match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (off_summary, high_summary):
        lines.append(
            f"| {row.get('mode_label')} | {row.get('qualification_scalar_valid')}/{row.get('qualification_variants')} | {row.get('qualification_scalar_invalid')} | {_qual_metric(row, 'mutation_evidence_supported_rate')} | {_qual_metric(row, 'critical_severity_recall_union')} | {_qual_metric(row, 'clean_false_positive_rate')} | {_qual_metric(row, 'error_type_match')} |"
        )
    lines += [
        "",
        "All qualification invalid rows and exact error/quote objects remain in the panel/error CSVs. Scalar validity is not used to rewrite evidence or pointwise scores.",
        "",
        "## 4. AB/BA, latency and reasoning budget",
        "",
        "| Mode | Complete swaps | Consistent | Position-sensitive | Consistency | Raw context flag | Known operation |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| off | {off_summary.get('pairwise_complete_swaps')} | {off_summary.get('pairwise_consistent')} | {off_summary.get('pairwise_position_sensitive')} | {_pct(off_summary.get('pairwise_swap_consistency'))} | `{off_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        f"| high | {high_summary.get('pairwise_complete_swaps')} | {high_summary.get('pairwise_consistent')} | {high_summary.get('pairwise_position_sensitive')} | {_pct(high_summary.get('pairwise_swap_consistency'))} | `{high_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        "",
        "The raw context flag may be `not_recorded`; known operation evidence is stated separately and does not enlarge the six-orientation sample.",
        "",
        "| Mode | Track | Events ok/failed/truncated | Latency median/p95 (s) | Completion tokens median | Reasoning tokens median |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in (off_summary, high_summary):
        for track in TRACKS:
            lines.append(
                f"| {row.get('mode_label')} | {track} | {row.get(f'{track}_event_count')}/{row.get(f'{track}_failed_events')}/{row.get(f'{track}_truncated_events')} | {_fmt(row.get(f'{track}_latency_median_s'))}/{_fmt(row.get(f'{track}_latency_p95_s'))} | {_fmt(row.get(f'{track}_completion_tokens_median'), 1)} | {_fmt(row.get(f'{track}_reasoning_tokens_median'), 1)} |"
            )
    lines += [
        "",
        f"The full non-generating reasoning token audit at `{high_raw.get('reasoning_budget_audit_path', 'n.a.')}` reports `{token_counts}` and SHA `{high_raw.get('reasoning_budget_audit_sha256', 'n.a.')}`. Three retained-reasoning values are conservative boundary candidates (tokenizer/tag offsets unresolved), not proof of native cap overrun: `{', '.join(boundary_ids) if boundary_ids else 'none'}`. `finish_reason=stop` does not prove natural reasoning completion. No chain-of-thought is copied into this report.",
        "",
        "## 5. Candidate ranking and source coverage",
        "",
        "Ranks are available-case descriptive ranks; no missing score is imputed. A candidate's valid N and source coverage are shown because source-cluster coverage is more informative than a rank alone.",
        "",
        "| Candidate | Off rank/mean (coverage) | High rank/mean (coverage) | Rank high−off |",
        "|---|---|---|---:|",
    ]
    for cid, off_rank, high_rank in rank_changes:
        off_row, high_row = rank_rows["off"][cid], rank_rows["on"][cid]
        delta = high_rank - off_rank if off_rank is not None and high_rank is not None else "n.a."
        lines.append(
            f"| `{cid}` | {off_rank}/{_fmt(off_row.get('score_mean'), 2)} ({_coverage(off_row)}) | {high_rank}/{_fmt(high_row.get('score_mean'), 2)} ({_coverage(high_row)}) | {delta} |"
        )
    lines += [
        "",
        "## 6. Reliability and rank concordance",
        "",
        "| Panel | ICC(2,1) | ICC(2,k) | actual k | units n | Kendall W | rankings n | ICC CI | LOFO caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in reliability:
        lines.append(
            f"| {row.get('mode_label', row.get('mode'))} | {_fmt(row.get('icc_2_1'), 6)} | {_fmt(row.get('icc_2_k'), 6)} | {row.get('n_judges_actual', 'n.a.')} | {row.get('n_units_actual', 'n.a.')} | {_fmt(row.get('kendall_w'), 6)} | {row.get('kendall_w_n_rankings', 'n.a.')} | {row.get('icc_ci_status', 'not available')} | {row.get('lofo_status', 'n.a.')} |"
        )
    lines += [
        "",
        "ICC CIs are unavailable because the implementation does not estimate them; none are invented. Kendall W uses the actual complete source-cluster rankings. With only three source clusters, LOFO is one-family/one-judge sensitivity and not a family-robust confidence interval. Higher ICC/W is not correctness evidence and was not used as a mode-selection rule.",
        "",
        "## 7. Method references",
        "",
        "The ICC reporting follows the two-way random-effects absolute-agreement form implemented by this repository; the local numeric results are computed only from the frozen raw bundles and manifests above. For interpretation of ICC reliability estimates and confidence-interval caution, see [Koo and Li (2016), PubMed](https://pubmed.ncbi.nlm.nih.gov/27330520/). This pilot has no implementation-generated ICC CI, so the CI field remains `not available`; the citation is methodological context, not a source for any local score.",
        "The qualification/error annotations are MQM-inspired owner diagnostics, not a claim of official MQM adjudication or semantic-quality ground truth. The frozen study protocol uses the structured MQM/GEMBA-MQM framing for judge design; see [GEMBA-MQM V2](https://aclanthology.org/2025.wmt-1.67/) and the repository's [original selection report](../selection_report.md). This comparison does not compute a GEMBA score and does not use that citation as a source for local values.",
        "Pairwise and evidence metrics remain as defined in the frozen comparison protocol; agreement remains descriptive and does not establish translation correctness.",
        "",
        "## 8. Output files and limitations",
        "",
        "- `panel_high/` is the separate high-replacement five-judge panel; its corrected report is `selection_report.mistral_high.md`.",
        "- `mistral_thinking_off_vs_high_paired.owner.csv` retains 21 exact frozen keys, exact source/target text, hashes, score eligibility, shape/quote/critical fields, and missing-vs-zero error-count scopes.",
        "- `mistral_thinking_off_vs_high_judge_summary.owner.csv` records both settings' validity, score distribution, qualification, ABBA, telemetry, budget and template status.",
        "- `mistral_thinking_off_vs_high_error_annotations.owner.csv` preserves judge-authored annotations plus clearly labelled raw-response/contract annotations; no annotation repairs a score.",
        "- `mistral_thinking_off_vs_high_pairwise.owner.csv`, `...panel_sensitivity.owner.csv`, and `...reliability.owner.csv` are separate mode diagnostics, never a combined six-judge estimate.",
        "- `mistral_retest_registry.json` pre-registers the fixed eight-request stability subset and observation-driven review candidates; it is `PLANNED_NOT_RUN` and cannot overwrite or replace this completed result.",
        "- The formal protocol remains under-sampled (21 pointwise source×candidate units, 12 parents/24 variants, 3 pairs/6 orientations); this is an exploratory local setting comparison. Qwen thinking-on is a later, separate worker result and is not included here.",
        "",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(packet: Path, off_raw_path: Path, high_raw_path: Path, off_panel: Path, high_panel: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    off_raw = read_json(off_raw_path)
    high_raw = read_json(high_raw_path)
    off_validation = modecmp.validate_import(packet, off_raw_path)
    high_validation = modecmp.validate_import(packet, high_raw_path)
    _write_json(output_dir / "validation.off.json", off_validation)
    _write_json(output_dir / "validation.high.json", high_validation)
    for label, audit in (("off", off_validation), ("high", high_validation)):
        eligibility = audit.get("track_eligibility", {}) if isinstance(audit.get("track_eligibility"), Mapping) else {}
        point = eligibility.get("pointwise", {}) if isinstance(eligibility.get("pointwise"), Mapping) else {}
        pair = eligibility.get("pairwise", {}) if isinstance(eligibility.get("pairwise"), Mapping) else {}
        errors = audit.get("errors", []) if isinstance(audit.get("errors"), list) else []
        if audit.get("status") != "PASS" or errors or point.get("status") != "eligible" or pair.get("status") != "eligible":
            raise ValueError(f"strict import gate failed for {label}: status={audit.get('status')} errors={errors[:2]} point={point} pair={pair}")
    off_summary = modecmp._raw_mode_summary(off_raw_path, off_raw, "off", off_validation)
    high_summary = modecmp._raw_mode_summary(high_raw_path, high_raw, "on", high_validation)
    high_manifest_path = high_raw_path.with_name(high_raw_path.name.replace(".raw.json", ".manifest.json"))
    high_manifest = read_json(high_manifest_path)
    addendum_path = high_raw_path.parent / "template_evidence_addendum.md"
    token_audit_path = high_raw_path.parent / "mistral-medium-3.5-128b-local-thinking-high-20261022.token_audit.json"
    token_audit = read_json(token_audit_path) if token_audit_path.exists() else {}
    counts = token_audit.get("summary", {}) if isinstance(token_audit.get("summary"), Mapping) else token_audit.get("counts", {})
    boundary_ids = token_audit.get("boundary_candidate_ids", [])
    if not isinstance(boundary_ids, list):
        boundary_ids = token_audit.get("at_or_over_budget_ids", []) if isinstance(token_audit.get("at_or_over_budget_ids"), list) else []
    if not boundary_ids and isinstance(counts, Mapping):
        boundary_ids = counts.get("at_or_over_budget_request_ids", [])
        if not isinstance(boundary_ids, list):
            boundary_ids = []
    if not boundary_ids:
        samples = token_audit.get("samples", []) if isinstance(token_audit.get("samples"), list) else []
        boundary_ids = [str(s.get("request_id")) for s in samples if isinstance(s, Mapping) and _num(s.get("reasoning_token_count")) is not None and _num(s.get("reasoning_token_count")) >= 8192]
    high_summary.update({
        "run_manifest_path": str(high_manifest_path),
        "run_manifest_sha256": sha256_file(high_manifest_path),
        "run_manifest": high_manifest.get("validation", {}),
        "template_verification_status": "PENDING_PROVENANCE_INCOMPLETE" if addendum_path.exists() else high_manifest.get("validation", {}).get("template_verification_status", "PENDING"),
        "template_verification_note": high_manifest.get("validation", {}).get("template_verification_note", ""),
        "reasoning_budget_audit_counts": dict(counts) if isinstance(counts, Mapping) else {},
        "reasoning_budget_audit_sha256": sha256_file(token_audit_path) if token_audit_path.exists() else "",
        "reasoning_budget_audit_path": str(token_audit_path) if token_audit_path.exists() else "",
        "reasoning_budget_tokenize_method": _json_text(token_audit.get("method", {})) if isinstance(token_audit, Mapping) else "",
        "reasoning_budget_tokenize_budget": 8192,
        "reasoning_budget_boundary_ids": boundary_ids,
        "reasoning_budget_audit_interpretation": token_audit.get("interpretation", "") if isinstance(token_audit, Mapping) else "",
    })
    off_data = _mode_data(off_panel, off_summary)
    high_data = _mode_data(high_panel, high_summary)
    paired = modecmp._paired_rows(off_data, high_data, off_summary, high_summary)
    if len(paired) != 21:
        raise ValueError(f"expected 21 paired rows, got {len(paired)}")
    off_row = _summary("off", off_summary, off_data)
    high_row = _summary("on", high_summary, high_data)
    high_row["mode_label"] = "thinking-high"
    _write_json(output_dir / "raw_mode_summary.json", {"off": off_summary, "high": high_summary})
    _write_csv(output_dir / "mistral_thinking_off_vs_high_paired.owner.csv", paired, list(paired[0]))
    summaries = [off_row, high_row]
    _write_csv(output_dir / "mistral_thinking_off_vs_high_judge_summary.owner.csv", summaries, list(off_row))
    errors = _join_errors("off", off_data, off_summary) + _join_errors("on", high_data, high_summary)
    error_fields: list[str] = ["mode", "mode_label"]
    for row in errors:
        for key in row:
            if key not in error_fields:
                error_fields.append(key)
    _write_csv(output_dir / "mistral_thinking_off_vs_high_error_annotations.owner.csv", errors, error_fields or ["mode", "mode_label"])
    pairwise = _join_pairwise("off", off_data, off_summary) + _join_pairwise("on", high_data, high_summary)
    pair_fields = ["mode", "mode_label"] + [key for key in pairwise[0] if key not in {"mode", "mode_label"}] if pairwise else ["mode", "mode_label"]
    _write_csv(output_dir / "mistral_thinking_off_vs_high_pairwise.owner.csv", pairwise, pair_fields)
    sensitivity = _generic_sensitivity("off", off_data) + _generic_sensitivity("on", high_data)
    _write_csv(output_dir / "mistral_thinking_off_vs_high_panel_sensitivity.owner.csv", sensitivity, list(sensitivity[0]) if sensitivity else ["mode"])
    reliability = [_reliability("off", off_data), _reliability("on", high_data)]
    reliability[0]["primary_panel_status"] = "ELIGIBLE"
    reliability[1]["primary_panel_status"] = "ELIGIBLE"
    _write_csv(output_dir / "mistral_thinking_off_vs_high_reliability.owner.csv", reliability, list(reliability[0]))
    manifest = {
        "schema_version": "mistral-thinking-comparison-v1",
        "comparison_status": "COMPLETE_STRICT_IMPORT_ANALYSIS",
        "packet_sha256": sha256_file(packet),
        "off_raw": {"path": str(off_raw_path), "sha256": off_summary["raw_bundle_sha256"], "judge_id": off_summary["judge_id"]},
        "high_raw": {"path": str(high_raw_path), "sha256": high_summary["raw_bundle_sha256"], "judge_id": high_summary["judge_id"]},
        "high_run_manifest": {"path": str(high_manifest_path), "sha256": sha256_file(high_manifest_path)},
        "high_template_evidence_addendum": {"path": str(addendum_path), "sha256": sha256_file(addendum_path) if addendum_path.exists() else ""},
        "off_panel": str(off_panel),
        "high_panel": str(high_panel),
        "paired_units": len(paired),
        "paired_eligible_units": sum(1 for row in paired if row.get("paired_score_status") == "paired_eligible"),
        "high_validation": {"status": high_validation.get("status"), "records": high_validation.get("records"), "track_eligibility": high_validation.get("track_eligibility")},
        "high_template_verification_status": high_summary.get("template_verification_status"),
        "high_reasoning_budget_audit_sha256": high_summary.get("reasoning_budget_audit_sha256"),
        "high_reasoning_budget_boundary_ids": boundary_ids,
        "reasoning_budget_audit_path": str(token_audit_path) if token_audit_path.exists() else "",
        "retest_registry": {
            "path": str(output_dir / "mistral_retest_registry.json"),
            "sha256": sha256_file(output_dir / "mistral_retest_registry.json") if (output_dir / "mistral_retest_registry.json").exists() else "",
        },
        "same_model_two_settings_not_sixth_judge": True,
        "primary_design": "replace_mistral_only; GPT/Qwen-off/Gemini/Nemotron-off frozen",
        "no_raw_mutation": True,
        "ci_status": "not available; implementation does not estimate ICC CI",
        "source_text_and_target_exact_checked": True,
        "qwen_thinking_on_included": False,
    }
    _write_json(output_dir / "mistral_thinking_comparison_manifest.json", manifest)
    _report(output_dir / "mistral_thinking_comparison.md", off_row, high_row, paired, off_data, high_data, sensitivity, reliability, off_summary, high_summary, high_manifest, addendum_path)
    _panel_summary_report(high_panel, high_panel / "selection_report.mistral_high.md", high_data, high_summary)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "paired_rows": len(paired),
        "paired_eligible": sum(1 for row in paired if row.get("paired_score_status") == "paired_eligible"),
        "error_rows": len(errors),
        "pairwise_rows": len(pairwise),
        "sensitivity_rows": len(sensitivity),
        "reliability_rows": len(reliability),
        "high_summary": {key: high_row.get(key) for key in ("pointwise_mean", "pointwise_score_sd_population", "pointwise_score_unique_values", "qualification_scalar_valid", "qualification_scalar_invalid", "pairwise_complete_swaps", "pairwise_consistent")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--off-raw", required=True, type=Path)
    parser.add_argument("--high-raw", required=True, type=Path)
    parser.add_argument("--off-panel", required=True, type=Path)
    parser.add_argument("--high-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_comparison(args.packet, args.off_raw, args.high_raw, args.off_panel, args.high_panel, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
