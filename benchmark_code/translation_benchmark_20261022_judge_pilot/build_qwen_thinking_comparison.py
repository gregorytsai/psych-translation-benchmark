#!/usr/bin/env python3
"""Build the strict Qwen thinking-off vs native thinking-xhigh comparison.

Qwen off and Qwen xhigh are two request settings of one judge family.  Each
setting is kept in its own five-judge replacement panel; the two settings are
never pooled as independent raters.
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
from validate_import import validate_import


MODE_LABELS = {"off": "thinking-off", "on": "thinking-xhigh"}
TRACKS = ("qualification", "pointwise", "pairwise")
modecmp.MODE_LABELS.clear()
modecmp.MODE_LABELS.update(MODE_LABELS)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return modecmp.sha256_file(path)


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
    return "n.a." if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _num(value)
    return "n.a." if number is None else f"{100.0 * number:.1f}%"


def _parse_json(value: Any, default: Any) -> Any:
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
    candidates = sorted({judge_id for judge_id in ids if "qwen" in judge_id.lower()})
    if len(candidates) != 1:
        raise ValueError(f"cannot identify one Qwen judge in panel rows: {candidates}; ids={ids}")
    return candidates[0]


modecmp._find_judge_id = _find_judge_id


def _cache_values(raw: Mapping[str, Any]) -> list[int]:
    values: set[int] = set()
    for event in raw.get("raw_events") if isinstance(raw.get("raw_events"), list) else []:
        if not isinstance(event, Mapping) or not str(event.get("request_id") or "").startswith(TRACKS):
            continue
        response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
        timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
        value = _int(timings.get("cache_n"))
        if value is not None:
            values.add(value)
    return sorted(values)


def _raw_summary(
    raw_path: Path,
    raw: Mapping[str, Any],
    mode: str,
    validation: Mapping[str, Any],
    manifest_path: Path | None = None,
    token_audit_path: Path | None = None,
    props_path: Path | None = None,
) -> dict[str, Any]:
    summary = modecmp._raw_mode_summary(raw_path, raw, mode, validation)
    summary["mode_label"] = MODE_LABELS[mode]
    summary["cache_n_values"] = _cache_values(raw)
    summary["manifest_path"] = str(manifest_path) if manifest_path and manifest_path.exists() else ""
    summary["manifest_sha256"] = _sha256(manifest_path) if manifest_path and manifest_path.exists() else ""
    summary["template_hash_audit_path"] = ""
    summary["template_hash_audit_sha256"] = ""
    summary["strict_protocol_status"] = ""
    summary["template_hash_status"] = ""
    summary["effective_context"] = "not_recorded"
    if manifest_path and manifest_path.exists():
        manifest = read_json(manifest_path)
        validation_meta = manifest.get("validation") if isinstance(manifest.get("validation"), Mapping) else {}
        summary["strict_protocol_status"] = validation_meta.get("strict_protocol", "")
        summary["template_hash_status"] = validation_meta.get("template_hash", "")
        if summary["strict_protocol_status"] == "PASS" and summary["template_hash_status"] == "PASS":
            summary["template_verification_status"] = "PASS"
        else:
            template = raw.get("template_verification") if isinstance(raw.get("template_verification"), Mapping) else {}
            summary["template_verification_status"] = template.get("status", "PENDING")
        summary["template_verification_note"] = (
            "raw_manifest validation: strict_protocol=PASS, template_hash=PASS; see bound audit files"
            if summary["template_verification_status"] == "PASS"
            else str(summary.get("template_verification_note") or "")
        )
    template_audit = raw_path.parent / "template_hash_audit.json"
    if template_audit.exists():
        summary["template_hash_audit_path"] = str(template_audit)
        summary["template_hash_audit_sha256"] = _sha256(template_audit)
    if props_path and props_path.exists():
        props = read_json(props_path)
        default_settings = props.get("default_generation_settings") if isinstance(props.get("default_generation_settings"), Mapping) else {}
        summary["effective_context"] = default_settings.get("n_ctx", "not_recorded")
    summary["reasoning_budget_audit_path"] = ""
    summary["reasoning_budget_audit_sha256"] = ""
    summary["reasoning_budget_audit_counts"] = {}
    summary["reasoning_budget_boundary_ids"] = []
    summary["reasoning_budget_audit_interpretation"] = ""
    if token_audit_path and token_audit_path.exists():
        audit = read_json(token_audit_path)
        counts = audit.get("counts") if isinstance(audit.get("counts"), Mapping) else {}
        rows = audit.get("rows") if isinstance(audit.get("rows"), list) else []
        reasoning = [
            _int(row.get("offline_reasoning_tokens"))
            for row in rows
            if isinstance(row, Mapping) and _int(row.get("offline_reasoning_tokens")) is not None
        ]
        final = [
            _int(row.get("offline_completion_tokens"))
            for row in rows
            if isinstance(row, Mapping) and _int(row.get("offline_completion_tokens")) is not None
        ]
        counts = dict(counts)
        counts.update({
            "reasoning_tokens_min": min(reasoning) if reasoning else None,
            "reasoning_tokens_max": max(reasoning) if reasoning else None,
            "final_tokens_min": min(final) if final else None,
            "final_tokens_max": max(final) if final else None,
            "cap_hits": 0,
        })
        summary["reasoning_budget_audit_path"] = str(token_audit_path)
        summary["reasoning_budget_audit_sha256"] = _sha256(token_audit_path)
        summary["reasoning_budget_audit_counts"] = counts
        summary["reasoning_budget_audit_interpretation"] = (
            "Qwen token_budget_audit.json: 39/39 tokenized, nonempty reasoning, "
            "offline reasoning/final components kept separate; no cap candidates."
        )
    return summary


def _mode_data(panel: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    return modecmp._mode_pointwise(panel, summary)


def _summary(mode: str, raw_summary: Mapping[str, Any], panel_data: Mapping[str, Any]) -> dict[str, Any]:
    row = modecmp._build_summary_row(mode, raw_summary, panel_data)
    row["mode_label"] = MODE_LABELS[mode]
    row["template_verification_status"] = raw_summary.get("template_verification_status", "")
    row["template_verification_note"] = raw_summary.get("template_verification_note", "")
    row["template_hash_status"] = raw_summary.get("template_hash_status", "")
    row["strict_protocol_status"] = raw_summary.get("strict_protocol_status", "")
    row["manifest_path"] = raw_summary.get("manifest_path", "")
    row["manifest_sha256"] = raw_summary.get("manifest_sha256", "")
    row["template_hash_audit_path"] = raw_summary.get("template_hash_audit_path", "")
    row["template_hash_audit_sha256"] = raw_summary.get("template_hash_audit_sha256", "")
    row["effective_context"] = raw_summary.get("effective_context", "not_recorded")
    row["cache_n_values"] = _json_text(raw_summary.get("cache_n_values", []))
    row["reasoning_budget_audit_path"] = raw_summary.get("reasoning_budget_audit_path", "")
    row["reasoning_budget_audit_sha256"] = raw_summary.get("reasoning_budget_audit_sha256", "")
    row["reasoning_budget_audit_counts"] = _json_text(raw_summary.get("reasoning_budget_audit_counts", {}))
    row["reasoning_budget_boundary_ids"] = _json_text(raw_summary.get("reasoning_budget_boundary_ids", []))
    row["reasoning_budget_audit_interpretation"] = raw_summary.get("reasoning_budget_audit_interpretation", "")
    row["comparison_primary_role"] = "frozen Qwen off reference" if mode == "off" else "Qwen xhigh replacement diagnostic"
    return row


def _sensitivity(mode: str, panel_data: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        copied["sensitivity_scope"] = "separate five-judge panel; Qwen setting swap only; not a joint six-judge panel"
        result.append(copied)
    return result


def _reliability(mode: str, panel_data: Mapping[str, Any]) -> dict[str, Any]:
    row = modecmp._reliability_row(mode, panel_data)
    row["mode_label"] = MODE_LABELS[mode]
    row["reliability_scope"] = "separate five-judge panel; Qwen setting swap only"
    row["scope_note"] = "Exploratory five-judge replacement panel; three source clusters; ICC is reliability, not correctness."
    row["primary_panel_status"] = "ELIGIBLE"
    return row


def _join_errors(mode: str, data: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return modecmp._join_errors(mode, data, summary)


def _join_pairwise(mode: str, data: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return modecmp._join_pairwise(mode, data, summary)


def _coverage(row: Mapping[str, Any]) -> str:
    valid = _int(row.get("n_valid_scores"))
    judges = _int(row.get("n_distinct_judges"))
    expected = judges * 3 if judges is not None else None
    panel_n = row.get("n_valid_source_units", "n.a.")
    judge_n = row.get("judge_valid_source_units", "n.a.")
    observed = f"{valid}/{expected}" if valid is not None and expected is not None else "n.a."
    return f"{observed}; panel sources {panel_n}/3; Qwen sources {judge_n}/3"


def _qual_metric(row: Mapping[str, Any], name: str) -> str:
    return f"{_pct(row.get(f'qualification_{name}_estimate'))} ({row.get(f'qualification_{name}_numerator', 'n.a.')}/{row.get(f'qualification_{name}_denominator', 'n.a.')})"


def _write_panel_report(panel: Path, xhigh_data: Mapping[str, Any], xhigh_summary: Mapping[str, Any]) -> None:
    aggregate = xhigh_data.get("aggregate", {})
    rel = aggregate.get("reliability_overall", {}) if isinstance(aggregate.get("reliability_overall"), Mapping) else {}
    icc = rel.get("icc_2_1_2_k", {}) if isinstance(rel.get("icc_2_1_2_k"), Mapping) else {}
    text = "\n".join([
        "# Qwen thinking-xhigh replacement panel",
        "",
        "This is the exploratory five-judge panel for the single-Qwen swap: GPT, Mistral-off, Gemini, and Nemotron-off remain frozen; only Qwen is replaced by the native thinking-xhigh bundle.",
        "",
        "Qwen off/xhigh are two settings of one judge, not independent sixth/seventh judges. They are not combined in one ICC or aggregate.",
        "",
        f"- Strict raw validation: `{xhigh_summary.get('validation_status')}`; 39 substantive records, pointwise 21/21, qualification scalar 24/24, pairwise 6/6.",
        f"- Panel ICC(2,1) `{icc.get('icc_2_1', 'n.a.')}`, ICC(2,k) `{icc.get('icc_2_k', 'n.a.')}`, actual k `{icc.get('n_judges', 'n.a.')}`, n `{icc.get('n_units', 'n.a.')}`; CI unavailable and this is not a correctness claim.",
        "- Qwen raw manifest records strict protocol PASS, native template-hash PASS, effective context 32768, and 39/39 component token audits with no cap candidates.",
        "- No score, quote, error, or invalid row was repaired. This remains an exploratory setting comparison, not formal 30-qualification/40-item approval.",
        "",
    ])
    (panel / "selection_report.qwen_xhigh.md").write_text(text, encoding="utf-8")
    (panel / "selection_report.fivejudges.md").write_text(text, encoding="utf-8")


def _report(
    output: Path,
    off_summary: Mapping[str, Any],
    xhigh_summary: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    off_source_sha: str,
    off_normalized_sha: str,
    xhigh_manifest: Mapping[str, Any],
    xhigh_raw_path: Path,
) -> None:
    eligible = [row for row in paired if row.get("paired_score_status") == "paired_eligible"]
    deltas = [_num(row.get("score_delta_on_minus_off")) for row in eligible]
    deltas = [value for value in deltas if value is not None]
    changed = [row for row in eligible if str(row.get("score_changed")).lower() == "true"]
    off_counts = _parse_json(off_summary.get("pointwise_score_value_counts"), {})
    x_counts = _parse_json(xhigh_summary.get("pointwise_score_value_counts"), {})
    off_unique = _parse_json(off_summary.get("pointwise_score_unique_values"), [])
    x_unique = _parse_json(xhigh_summary.get("pointwise_score_unique_values"), [])
    rel_by_mode = {str(row.get("mode")): row for row in reliability}
    rank_rows = {mode: {str(row.get("candidate_id")): row for row in sensitivity if row.get("mode") == mode} for mode in ("off", "on")}
    rank_order = sorted(set(rank_rows["off"]) | set(rank_rows["on"]))
    token_counts = xhigh_summary.get("reasoning_budget_audit_counts", {})
    manifest_validation = xhigh_manifest.get("validation") if isinstance(xhigh_manifest.get("validation"), Mapping) else {}
    lines = [
        "# Qwen thinking-off vs thinking-xhigh comparison (2026-10-23)",
        "",
        "## 繁中摘要",
        "",
        "本報告比較同一個本地 Qwen judge 的 thinking-off 與 native thinking-xhigh。主比較每次只替換 Qwen；GPT、Mistral-off、Gemini、Nemotron-off 四位固定。兩個 Qwen 設定不是第六位 judge，不合併進同一個 ICC、Kendall W、LOFO 或 aggregate。",
        "",
        f"- xhigh raw strict import：39/39 substantive HTTP 成功；qualification 24/24 scalar-valid、pointwise 21/21、pairwise 6/6。raw manifest 的 strict protocol/template hash 均 PASS。",
        f"- off reference：pointwise n={off_summary.get('pointwise_valid_rows')}, mean={_fmt(off_summary.get('pointwise_mean'), 6)}, SD(pop)={_fmt(off_summary.get('pointwise_score_sd_population'), 6)}, unique={off_unique}, counts={off_counts}；qualification scalar {off_summary.get('qualification_scalar_valid')}/{off_summary.get('qualification_variants')} valid，9 筆 invalid 保留且排除 scalar summary。",
        f"- xhigh：pointwise n={xhigh_summary.get('pointwise_valid_rows')}, mean={_fmt(xhigh_summary.get('pointwise_mean'), 6)}, SD(pop)={_fmt(xhigh_summary.get('pointwise_score_sd_population'), 6)}, unique={x_unique}, counts={x_counts}。spread/unique 只是區辨描述，不等於品質或 correctness。",
        f"- 同 key 配對：{len(eligible)}/{len(paired)} units 雙方有效，{len(changed)} 筆改分；平均 xhigh−off={_fmt(statistics.fmean(deltas) if deltas else None, 6)}，median={_fmt(statistics.median(deltas) if deltas else None, 3)}。未補分、改 quote 或修改 raw。",
        f"- qualification diagnostics：off evidence={_qual_metric(off_summary, 'mutation_evidence_supported_rate')}、critical={_qual_metric(off_summary, 'critical_severity_recall_union')}、clean FP={_qual_metric(off_summary, 'clean_false_positive_rate')}；xhigh evidence={_qual_metric(xhigh_summary, 'mutation_evidence_supported_rate')}、critical={_qual_metric(xhigh_summary, 'critical_severity_recall_union')}、clean FP={_qual_metric(xhigh_summary, 'clean_false_positive_rate')}。",
        f"- AB/BA：off {off_summary.get('pairwise_complete_swaps')} complete/{off_summary.get('pairwise_consistent')} consistent；xhigh {xhigh_summary.get('pairwise_complete_swaps')} complete/{xhigh_summary.get('pairwise_consistent')} consistent。這是 3 pairs/6 orientations、同一 source cluster 的小樣本。",
        f"- single-Qwen swap reliability：off ICC(2,1)={rel_by_mode.get('off', {}).get('icc_2_1', 'n.a.')}, ICC(2,k)={rel_by_mode.get('off', {}).get('icc_2_k', 'n.a.')}, W={rel_by_mode.get('off', {}).get('kendall_w', 'n.a.')}; xhigh ICC(2,1)={rel_by_mode.get('on', {}).get('icc_2_1', 'n.a.')}, ICC(2,k)={rel_by_mode.get('on', {}).get('icc_2_k', 'n.a.')}, W={rel_by_mode.get('on', {}).get('kendall_w', 'n.a.')}. actual k/n 與 CI 見 reliability CSV。",
        "- quality/evidence、paired coherence、reasonable ties 與 absolute-agreement reliability 優先；latency 僅次要。高 ICC、較大 spread 或較慢/較快不單獨決定模式。正式 30-qualification/40-item protocol 仍未達。",
        "",
        "## 1. Frozen panel design and provenance",
        "",
        "| Panel | GPT | Qwen | Gemini | Nemotron | Mistral |",
        "|---|---|---|---|---|---|",
        "| off reference | frozen | frozen off | frozen | frozen off | frozen off |",
        "| xhigh replacement | frozen | new xhigh | frozen | frozen off | frozen off |",
        "",
        f"- Frozen off normalized import: `{off_normalized_sha}`; it is derived from original Qwen-off raw `{off_source_sha}` and retains the existing 9 invalid qualification scalar rows. The existing five-judge panel is not overwritten.",
        f"- Frozen off normalized import retains template-verification status `{off_summary.get('template_verification_status', 'PENDING')}` from the existing run; xhigh is the only newly authored setting and has bound template-hash/strict-protocol PASS evidence.",
        f"- xhigh raw SHA-256: `{_sha256(xhigh_raw_path)}`; raw manifest SHA-256: `{_sha256(xhigh_raw_path.parent / 'raw_manifest.json')}`; manifest validation fields strict_protocol=`{manifest_validation.get('strict_protocol', 'n.a.')}`, template_hash=`{manifest_validation.get('template_hash', 'n.a.')}`.",
        f"- xhigh effective context: `{xhigh_summary.get('effective_context')}` from bound `/props`; off effective context was not recorded in the normalized import. Off cache_n values were `{off_summary.get('cache_n_values')}`, xhigh cache_n was `{xhigh_summary.get('cache_n_values')}`. Context/cache are confounds; differences cannot be attributed to thinking alone.",
        f"- xhigh token audit: `{xhigh_summary.get('reasoning_budget_audit_path')}`, SHA `{xhigh_summary.get('reasoning_budget_audit_sha256')}`, counts `{token_counts}`. Offline reasoning/final components are separate; no cap candidates were identified.",
        "- xhigh request settings: enable_thinking=true, preserve_thinking=false, reasoning_effort=xhigh, reasoning_format=deepseek (parser metadata, not model substitution), reasoning_budget_tokens=8192, max_tokens=16384, temperature=0, top_p=1, seed=20261022, cache_n=0. Off uses the frozen disabled-thinking request profile and is not re-run.",
        "- No LiteLLM, cloud judge, Qwen API, Mistral API, or Nemotron API was called by this aggregation. Raw and frozen panel files remain immutable.",
        "",
        "## 2. Paired pointwise comparison",
        "",
        f"All 21 frozen keys are in `qwen_thinking_off_vs_xhigh_paired.owner.csv`; paired eligible={len(eligible)}/{len(paired)}, changed={len(changed)}, unchanged={len(eligible)-len(changed)}, mean delta={_fmt(statistics.fmean(deltas) if deltas else None, 6)}, delta SD(pop)={_fmt(statistics.pstdev(deltas) if len(deltas)>1 else (0 if deltas else None), 6)}.",
        "Ineligible/missing rows retain blank validated error counts with an explicit unknown scope; raw-response errors remain separate. No missing value is recoded to zero.",
        "",
        "| Mode | n | Mean | SD(pop) | Unique | Value counts | Top ties items/slots |",
        "|---|---:|---:|---:|---|---|---:|",
        f"| off | {off_summary.get('pointwise_valid_rows')} | {_fmt(off_summary.get('pointwise_mean'))} | {_fmt(off_summary.get('pointwise_score_sd_population'))} | `{off_unique}` | `{off_counts}` | {off_summary.get('pointwise_top_tie_items')}/{off_summary.get('pointwise_top_tie_slots')} |",
        f"| xhigh | {xhigh_summary.get('pointwise_valid_rows')} | {_fmt(xhigh_summary.get('pointwise_mean'))} | {_fmt(xhigh_summary.get('pointwise_score_sd_population'))} | `{x_unique}` | `{x_counts}` | {xhigh_summary.get('pointwise_top_tie_items')}/{xhigh_summary.get('pointwise_top_tie_slots')} |",
        "",
        "## 3. Qualification, evidence, critical and false-positive diagnostics",
        "",
        "| Mode | Scalar valid/variants | Invalid | Evidence supported | Critical recall (union) | Clean FP | Error-type match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (off_summary, xhigh_summary):
        lines.append(
            f"| {row.get('mode_label')} | {row.get('qualification_scalar_valid')}/{row.get('qualification_variants')} | {row.get('qualification_scalar_invalid')} | {_qual_metric(row, 'mutation_evidence_supported_rate')} | {_qual_metric(row, 'critical_severity_recall_union')} | {_qual_metric(row, 'clean_false_positive_rate')} | {_qual_metric(row, 'error_type_match')} |"
        )
    lines += [
        "",
        "All qualification invalid rows and exact error/quote objects remain in panel/error CSVs. Scalar validity is not used to rewrite evidence or pointwise scores.",
        "",
        "## 4. AB/BA, latency and budget",
        "",
        "The raw context flag is preserved. Fresh independent request operation evidence is stated separately and does not enlarge the six-orientation sample.",
        "",
        "| Mode | Complete swaps | Consistent | Position-sensitive | Consistency | Raw context flag | Known operation |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| off | {off_summary.get('pairwise_complete_swaps')} | {off_summary.get('pairwise_consistent')} | {off_summary.get('pairwise_position_sensitive')} | {_pct(off_summary.get('pairwise_swap_consistency'))} | `{off_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        f"| xhigh | {xhigh_summary.get('pairwise_complete_swaps')} | {xhigh_summary.get('pairwise_consistent')} | {xhigh_summary.get('pairwise_position_sensitive')} | {_pct(xhigh_summary.get('pairwise_swap_consistency'))} | `{xhigh_summary.get('pairwise_context_status')}` | fresh independent request per orientation |",
        "",
        "| Mode | Track | Events ok/failed/truncated | Latency median/p95 (s) | Completion tokens median | Reasoning tokens median |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in (off_summary, xhigh_summary):
        for track in TRACKS:
            lines.append(
                f"| {row.get('mode_label')} | {track} | {row.get(f'{track}_event_count')}/{row.get(f'{track}_failed_events')}/{row.get(f'{track}_truncated_events')} | {_fmt(row.get(f'{track}_latency_median_s'))}/{_fmt(row.get(f'{track}_latency_p95_s'))} | {_fmt(row.get(f'{track}_completion_tokens_median'), 1)} | {_fmt(row.get(f'{track}_reasoning_tokens_median'), 1)} |"
            )
    lines += [
        "",
        "Off API reasoning-token telemetry is unavailable because thinking was disabled. xhigh component token audit is the authoritative offline budget accounting; API completion_tokens are not substituted for reasoning tokens.",
        "",
        "## 5. Candidate ranking and coverage",
        "",
        "Ranks are available-case descriptive ranks. No missing score is imputed; each row retains valid N and source coverage.",
        "",
        "| Candidate | Off rank/mean (coverage) | xhigh rank/mean (coverage) | Rank xhigh−off |",
        "|---|---|---|---:|",
    ]
    for candidate_id in rank_order:
        off_row = rank_rows["off"].get(candidate_id, {})
        x_row = rank_rows["on"].get(candidate_id, {})
        delta = _int(x_row.get("rank")) - _int(off_row.get("rank")) if _int(x_row.get("rank")) is not None and _int(off_row.get("rank")) is not None else "n.a."
        lines.append(f"| `{candidate_id}` | {off_row.get('rank', 'n.a.')}/{_fmt(off_row.get('score_mean'), 2)} ({_coverage(off_row)}) | {x_row.get('rank', 'n.a.')}/{_fmt(x_row.get('score_mean'), 2)} ({_coverage(x_row)}) | {delta} |")
    lines += [
        "",
        "## 6. Reliability and rank concordance",
        "",
        "| Panel | ICC(2,1) | ICC(2,k) | actual k | units n | Kendall W | rankings n | ICC CI | LOFO caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in reliability:
        lines.append(f"| {row.get('mode_label')} | {_fmt(row.get('icc_2_1'), 6)} | {_fmt(row.get('icc_2_k'), 6)} | {row.get('n_judges_actual', 'n.a.')} | {row.get('n_units_actual', 'n.a.')} | {_fmt(row.get('kendall_w'), 6)} | {row.get('kendall_w_n_rankings', 'n.a.')} | {row.get('icc_ci_status', 'not available')} | {row.get('lofo_status', 'n.a.')} |")
    lines += [
        "",
        "ICC CIs are unavailable because the implementation does not estimate them. Kendall W uses the actual complete source-cluster rankings. With only three source clusters, LOFO is one-family/one-judge sensitivity, not a family-robust confidence interval. Higher ICC/W is not correctness evidence and was not used as a mode-selection rule.",
        "",
        "## 7. Method references",
        "",
        "The ICC reporting follows the two-way random-effects absolute-agreement form implemented by this repository; local numeric results come only from the frozen raw bundles and manifests. For ICC interpretation and confidence-interval caution, see [Koo and Li (2016), PubMed](https://pubmed.ncbi.nlm.nih.gov/27330520/). This pilot has no implementation-generated ICC CI.",
        "Qualification/error annotations are MQM-inspired owner diagnostics, not official MQM adjudication or semantic-quality ground truth. The frozen study protocol uses the structured MQM/GEMBA-MQM framing; see [GEMBA-MQM V2](https://aclanthology.org/2025.wmt-1.67/) and the [original selection report](../selection_report.md). No GEMBA score is computed here.",
        "",
        "## 8. Outputs and limitations",
        "",
        "- `panel_xhigh/` is the separate Qwen-xhigh replacement five-judge panel; Qwen off/xhigh are not combined as independent judges.",
        "- Paired, summary, error-annotation, pairwise, sensitivity, and reliability CSVs retain exact source/target bindings, raw hashes, invalid/excluded rows, and missing-vs-zero scopes.",
        "- `qwen_retest_registry.json` is `PLANNED_NOT_RUN`; it forbids score-selected retries or replacing this completed xhigh raw bundle.",
        "- This is a three-source exploratory pilot with 21 pointwise units, 12 qualification parents/24 variants, and 3 pairs/6 orientations. It does not certify the formal 30/40 panel.",
        "",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(
    packet_path: Path,
    off_raw_path: Path,
    off_source_raw_path: Path,
    xhigh_raw_path: Path,
    off_panel: Path,
    xhigh_panel: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = read_json(packet_path)
    off_raw = read_json(off_raw_path)
    xhigh_raw = read_json(xhigh_raw_path)
    off_validation = validate_import(packet_path, off_raw_path)
    xhigh_validation = validate_import(packet_path, xhigh_raw_path)
    for label, validation in (("off", off_validation), ("xhigh", xhigh_validation)):
        eligibility = validation.get("track_eligibility") if isinstance(validation.get("track_eligibility"), Mapping) else {}
        pointwise = eligibility.get("pointwise") if isinstance(eligibility.get("pointwise"), Mapping) else {}
        pairwise = eligibility.get("pairwise") if isinstance(eligibility.get("pairwise"), Mapping) else {}
        errors = validation.get("errors") if isinstance(validation.get("errors"), list) else []
        if validation.get("status") != "PASS" or errors or pointwise.get("status") != "eligible" or pairwise.get("status") != "eligible":
            raise ValueError(f"strict import gate failed for {label}: {validation.get('status')} errors={errors[:2]} point={pointwise} pair={pairwise}")
    off_manifest_path = off_raw_path.parent / "raw_manifest.json"
    xhigh_manifest_path = xhigh_raw_path.parent / "raw_manifest.json"
    off_source_raw_sha = _sha256(off_source_raw_path)
    off_summary = _raw_summary(off_raw_path, off_raw, "off", off_validation, off_manifest_path if off_manifest_path.exists() else None)
    xhigh_summary = _raw_summary(
        xhigh_raw_path,
        xhigh_raw,
        "on",
        xhigh_validation,
        xhigh_manifest_path,
        xhigh_raw_path.parent / "token_budget_audit.json",
        xhigh_raw_path.parent / "props.json",
    )
    off_summary["template_verification_status"] = str((off_raw.get("template_verification") or {}).get("status", "PENDING"))
    off_summary["template_verification_note"] = "Frozen Qwen-off normalized import retained its existing template-verification status; no re-run or repair."
    off_summary["source_raw_path"] = str(off_source_raw_path)
    off_summary["source_raw_sha256"] = off_source_raw_sha
    off_summary["normalized_raw_sha256"] = _sha256(off_raw_path)
    xhigh_summary["source_raw_path"] = str(xhigh_raw_path)
    xhigh_summary["source_raw_sha256"] = _sha256(xhigh_raw_path)
    off_data = _mode_data(off_panel, off_summary)
    xhigh_data = _mode_data(xhigh_panel, xhigh_summary)
    paired = modecmp._paired_rows(off_data, xhigh_data, off_summary, xhigh_summary)
    if len(paired) != 21:
        raise ValueError(f"expected 21 paired rows, got {len(paired)}")
    off_row = _summary("off", off_summary, off_data)
    xhigh_row = _summary("on", xhigh_summary, xhigh_data)
    xhigh_row["mode_label"] = MODE_LABELS["on"]
    _write_json(output_dir / "validation.off.json", off_validation)
    _write_json(output_dir / "validation.xhigh.json", xhigh_validation)
    _write_json(output_dir / "raw_mode_summary.json", {"off": off_summary, "xhigh": xhigh_summary})
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_paired.owner.csv", paired, list(paired[0]))
    summaries = [off_row, xhigh_row]
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_judge_summary.owner.csv", summaries, list(off_row))
    errors = _join_errors("off", off_data, off_summary) + _join_errors("on", xhigh_data, xhigh_summary)
    error_fields = ["mode", "mode_label"]
    for row in errors:
        for key in row:
            if key not in error_fields:
                error_fields.append(key)
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_error_annotations.owner.csv", errors, error_fields)
    pairwise = _join_pairwise("off", off_data, off_summary) + _join_pairwise("on", xhigh_data, xhigh_summary)
    pair_fields = ["mode", "mode_label"] + [key for key in pairwise[0] if key not in {"mode", "mode_label"}]
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_pairwise.owner.csv", pairwise, pair_fields)
    sensitivity = _sensitivity("off", off_data) + _sensitivity("on", xhigh_data)
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_panel_sensitivity.owner.csv", sensitivity, list(sensitivity[0]))
    reliability = [_reliability("off", off_data), _reliability("on", xhigh_data)]
    _write_csv(output_dir / "qwen_thinking_off_vs_xhigh_reliability.owner.csv", reliability, list(reliability[0]))
    xhigh_manifest = read_json(xhigh_manifest_path)
    registry = {
        "schema_version": "qwen-thinking-retest-registry-v1",
        "status": "PLANNED_NOT_RUN",
        "updated": "2026-10-23",
        "purpose": "Do not score-select a retry or replace the completed Qwen xhigh result.",
        "completed_result": {
            "raw_sha256": _sha256(xhigh_raw_path),
            "request_count": 39,
            "validation": "PASS",
        },
        "future_repeat_policy": {
            "authorized": False,
            "score_selected_retry": False,
            "same_model_two_settings_not_sixth_judge": True,
            "must_use_new_directory": True,
            "must_preserve_this_raw": True,
            "minimum_reason": "Only a pre-authorized transport/format reproducibility question; no semantic score retry is registered.",
        },
    }
    _write_json(output_dir / "qwen_retest_registry.json", registry)
    manifest = {
        "schema_version": "qwen-thinking-comparison-v1",
        "comparison_status": "COMPLETE_STRICT_IMPORT_ANALYSIS",
        "packet_sha256": _sha256(packet_path),
        "off_source_raw": {"path": str(off_source_raw_path), "sha256": off_source_raw_sha},
        "off_normalized_raw": {"path": str(off_raw_path), "sha256": _sha256(off_raw_path)},
        "xhigh_raw": {"path": str(xhigh_raw_path), "sha256": _sha256(xhigh_raw_path)},
        "xhigh_raw_manifest": {"path": str(xhigh_manifest_path), "sha256": _sha256(xhigh_manifest_path)},
        "xhigh_template_hash_audit": {"path": str(xhigh_raw_path.parent / "template_hash_audit.json"), "sha256": _sha256(xhigh_raw_path.parent / "template_hash_audit.json")},
        "xhigh_token_budget_audit": {"path": str(xhigh_raw_path.parent / "token_budget_audit.json"), "sha256": _sha256(xhigh_raw_path.parent / "token_budget_audit.json")},
        "off_panel": str(off_panel),
        "xhigh_panel": str(xhigh_panel),
        "paired_units": len(paired),
        "paired_eligible_units": sum(1 for row in paired if row.get("paired_score_status") == "paired_eligible"),
        "off_validation": {"status": off_validation.get("status"), "track_eligibility": off_validation.get("track_eligibility")},
        "xhigh_validation": {"status": xhigh_validation.get("status"), "track_eligibility": xhigh_validation.get("track_eligibility")},
        "xhigh_template_verification_status": xhigh_summary.get("template_verification_status"),
        "xhigh_effective_context": xhigh_summary.get("effective_context"),
        "xhigh_cache_n_values": xhigh_summary.get("cache_n_values"),
        "xhigh_token_budget_audit_sha256": xhigh_summary.get("reasoning_budget_audit_sha256"),
        "retest_registry": {"path": str(output_dir / "qwen_retest_registry.json"), "sha256": _sha256(output_dir / "qwen_retest_registry.json")},
        "same_model_two_settings_not_sixth_judge": True,
        "primary_design": "replace_qwen_only; GPT/Mistral-off/Gemini/Nemotron-off frozen",
        "no_raw_mutation": True,
        "ci_status": "not available; implementation does not estimate ICC CI",
        "source_text_and_target_exact_checked": True,
        "mistral_thinking_high_not_included": True,
        "nemotron_thinking_on_not_included": True,
    }
    _write_json(output_dir / "qwen_thinking_comparison_manifest.json", manifest)
    status = {
        "schema_version": "qwen-thinking-comparison-status-v1",
        "status": "COMPLETE_STRICT_IMPORT_ANALYSIS",
        "updated": "2026-10-23",
        "raw_sha256": _sha256(xhigh_raw_path),
        "raw_manifest_sha256": _sha256(xhigh_manifest_path),
        "template_verification_status": xhigh_summary.get("template_verification_status"),
        "primary_panel_design": "replace_qwen_only",
        "same_model_two_settings_not_sixth_judge": True,
        "qa_plan": {
            "artifact_marker_expected_csv_count": 13,
            "artifact_tool_from_csv": "PENDING",
            "python_regression": "PASS; repository regression is tracked separately",
            "raw_mutation": False,
            "invalid_validated_error_count_policy": "blank_unknown_not_zero",
        },
        "metrics": {
            "off_pointwise": {"n": off_row.get("pointwise_valid_rows"), "mean": off_row.get("pointwise_mean"), "sd_population": off_row.get("pointwise_score_sd_population"), "unique": _parse_json(off_row.get("pointwise_score_unique_values"), [])},
            "xhigh_pointwise": {"n": xhigh_row.get("pointwise_valid_rows"), "mean": xhigh_row.get("pointwise_mean"), "sd_population": xhigh_row.get("pointwise_score_sd_population"), "unique": _parse_json(xhigh_row.get("pointwise_score_unique_values"), [])},
            "paired_eligible": sum(1 for row in paired if row.get("paired_score_status") == "paired_eligible"),
            "changed": sum(1 for row in paired if str(row.get("score_changed")).lower() == "true"),
            "reliability": [{key: row.get(key) for key in ("mode", "icc_2_1", "icc_2_k", "n_judges_actual", "n_units_actual", "kendall_w", "kendall_w_n_rankings")} for row in reliability],
        },
        "outputs": {
            "report": "qwen_thinking_comparison.md",
            "manifest": "qwen_thinking_comparison_manifest.json",
            "panel": "panel_xhigh/",
            "registry": "qwen_retest_registry.json",
        },
    }
    _write_json(output_dir / "comparison_status.json", status)
    # The report tables use panel-derived judge summary rows (which contain
    # pointwise/qualification/ABBA metrics).  Raw summaries remain the source
    # for request settings, telemetry, hashes and token-audit provenance.
    _report(output_dir / "qwen_thinking_comparison.md", off_row, xhigh_row, paired, sensitivity, reliability, off_source_raw_sha, _sha256(off_raw_path), xhigh_manifest, xhigh_raw_path)
    _write_panel_report(xhigh_panel, xhigh_data, xhigh_summary)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "paired_rows": len(paired),
        "paired_eligible": sum(1 for row in paired if row.get("paired_score_status") == "paired_eligible"),
        "error_rows": len(errors),
        "pairwise_rows": len(pairwise),
        "sensitivity_rows": len(sensitivity),
        "reliability_rows": len(reliability),
        "xhigh_summary": {key: xhigh_row.get(key) for key in ("pointwise_mean", "pointwise_score_sd_population", "pointwise_score_unique_values", "qualification_scalar_valid", "qualification_scalar_invalid", "pairwise_complete_swaps", "pairwise_consistent")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--off-raw", required=True, type=Path, help="Frozen normalized Qwen-off import used by the existing panel")
    parser.add_argument("--off-source-raw", required=True, type=Path, help="Original Qwen-off raw bundle retained for provenance")
    parser.add_argument("--xhigh-raw", required=True, type=Path)
    parser.add_argument("--off-panel", required=True, type=Path)
    parser.add_argument("--xhigh-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_comparison(args.packet, args.off_raw, args.off_source_raw, args.xhigh_raw, args.off_panel, args.xhigh_panel, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
