#!/usr/bin/env node

/*
 * Independent evidence QA for the current partial judge reports.
 *
 * This is intentionally not a scorer and never starts a worker, model,
 * server, endpoint, or refresh.  It reads the already-produced report
 * snapshot and writes only report_qa_results.json/report_qa_report.md next to
 * this file.  Every CSV is opened through the bundled ArtifactTool API.
 */

import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const { Workbook } = await import(
  process.env.CODEX_ARTIFACT_TOOL_MODULE ||
    "/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
);

const QA_ROOT = path.dirname(fileURLToPath(import.meta.url));
const JUDGE_ROOT = path.resolve(QA_ROOT, "..");
const BATCH_ROOT = path.resolve(JUDGE_ROOT, "..");
const REPORT_ROOT = path.join(JUDGE_ROOT, "reports");
const MATERIAL_ROOT = path.join(BATCH_ROOT, "materials");
const RESULTS_ROOT = path.join(BATCH_ROOT, "results");
const ANALYSIS_ROOT = path.join(JUDGE_ROOT, "analysis_runner");
const REPORT_MANIFEST_PATH = path.join(REPORT_ROOT, "analysis_manifest.json");
const SOURCE_CSV_PATH = path.join(RESULTS_ROOT, "translation_results.csv");
const PACKET_MANIFEST_PATH = path.join(JUDGE_ROOT, "manifest.json");

const JUDGES = ["judge-01", "judge-02", "judge-03", "judge-04", "judge-05"];
const IMAGE_PANEL = ["judge-01", "judge-02", "judge-04", "judge-05"];
const MODELS = ["gemma4-12b", "gemma4-26b", "gemma4-31b-qat", "gemma4-e4b", "translategemma-12b", "translategemma-27b", "translategemma-4b"];
const EXPECTED_REPORT_ROWS = {
  "translation_comparison.csv": 238,
  "pointwise_results_long.csv": 1190,
  "pairwise_raw.csv": 360,
  "pairwise_canonical.csv": 180,
  "judge_summary.csv": 5,
  "model_summary.csv": 14,
  "reliability_summary.csv": 2,
  "bootstrap_summary.csv": 14,
  "translation_generation_summary.csv": 14,
  "lofo_summary.csv": 9,
  "settings_provenance.csv": 3,
  "snapshot_diagnostics.csv": 0,
  "error_annotations.csv": 34,
  "invalid_annotations.csv": 9,
};
const REPORT_FILES = Object.keys(EXPECTED_REPORT_ROWS);
const REQUIRED_REPORT_EXTRAS = ["operational_retry_addendum.md", "analysis_report.md", "analysis_manifest.json"];

function sha256Bytes(bytes) { return crypto.createHash("sha256").update(bytes).digest("hex"); }
function sha256Text(text) { return sha256Bytes(Buffer.from(String(text), "utf8")); }
function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function boolCell(value) { return String(value).toLowerCase() === "true"; }
function isBlank(value) { return value === null || value === undefined || value === ""; }
function stableJson(value) {
  if (Array.isArray(value)) return JSON.stringify(value.map((x) => JSON.parse(stableJson(x))));
  if (value && typeof value === "object") return JSON.stringify(Object.fromEntries(Object.keys(value).sort().map((k) => [k, JSON.parse(stableJson(value[k]))])));
  return JSON.stringify(value);
}
function add(issues, code, detail = undefined, severity = "block") {
  const item = { code, severity };
  if (detail !== undefined) item.detail = detail;
  issues.push(item);
}
function key(...parts) { return parts.join("::"); }
function uniq(values) { return [...new Set(values)]; }

async function stableRead(filePath) {
  const firstStat = await fs.stat(filePath);
  const first = await fs.readFile(filePath);
  const firstHash = sha256Bytes(first);
  const second = await fs.readFile(filePath);
  const secondStat = await fs.stat(filePath);
  const secondHash = sha256Bytes(second);
  const stable = firstHash === secondHash && firstStat.size === secondStat.size && firstStat.mtimeMs === secondStat.mtimeMs;
  return { file: filePath, text: first.toString("utf8"), bytes: first, sha256: firstHash, size: first.length, stable, mtimeMs: firstStat.mtimeMs };
}
async function readJson(filePath) { return JSON.parse((await stableRead(filePath)).text); }
function parseJsonl(text, filePath) {
  const rows = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); }
    catch (error) { throw new Error(`invalid JSONL ${filePath}:${index + 1}: ${error.message}`); }
  }
  return rows;
}
async function readJsonl(filePath) { return parseJsonl((await stableRead(filePath)).text, filePath); }

function artifactRows(values, filePath, issues) {
  if (!Array.isArray(values) || values.length < 1) { add(issues, "artifact_csv_empty", filePath); return { headers: [], rows: [] }; }
  const headers = values[0].map((x) => String(x ?? ""));
  if (!headers.length || headers.some((x) => !x)) add(issues, "artifact_csv_empty_header", filePath);
  if (new Set(headers).size !== headers.length) add(issues, "artifact_csv_duplicate_header", { filePath, headers });
  const rows = values.slice(1).map((valuesRow) => Object.fromEntries(headers.map((h, i) => [h, valuesRow?.[i] === null || valuesRow?.[i] === undefined ? "" : String(valuesRow[i])] )));
  return { headers, rows };
}

async function loadCsv(name, issues) {
  const filePath = path.join(REPORT_ROOT, name);
  const stable = await stableRead(filePath);
  if (!stable.stable) add(issues, "report_csv_unstable_read", name);
  const sheetName = name.replace(/\.csv$/, "");
  const workbook = await Workbook.fromCSV(stable.text, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  const parsed = artifactRows(used?.values || [], filePath, issues);
  const expected = EXPECTED_REPORT_ROWS[name];
  if (expected !== null && parsed.rows.length !== expected) add(issues, "report_csv_row_count", { name, expected, actual: parsed.rows.length });
  return { name, filePath, sha256: stable.sha256, stable: stable.stable, headers: parsed.headers, rows: parsed.rows, usedRange: used?.address || null };
}

async function loadArtifactCsv(filePath, sheetName, issues) {
  const stable = await stableRead(filePath);
  if (!stable.stable) add(issues, "source_csv_unstable_read", filePath);
  const workbook = await Workbook.fromCSV(stable.text, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const parsed = artifactRows(sheet.getUsedRange()?.values || [], filePath, issues);
  return { ...parsed, sha256: stable.sha256, stable: stable.stable, usedRange: sheet.getUsedRange()?.address || null };
}

function mapRows(rows, fields, issues, label) {
  const map = new Map();
  for (const row of rows) {
    const id = key(...fields.map((field) => row[field]));
    if (map.has(id)) add(issues, "duplicate_report_key", { label, id });
    map.set(id, row);
  }
  return map;
}

function expectedMaterialCases(textRows, imageSelection) {
  const cases = new Map();
  for (const row of textRows) {
    cases.set(row.item_id, {
      material_id: row.item_id, source_kind: "text", source_id: row.source_id, source_en: row.source_en,
      source_sha256: sha256Text(row.source_en), image_sha256: "", image_ground_truth_en: "", image_ground_truth_status: "",
      image_reference_text_kind: "", image_role: "", image_layout_stratum: "", source_record: row,
    });
  }
  for (const image of imageSelection.selected_images || []) {
    const gt = Array.isArray(image.ground_truth_text) ? image.ground_truth_text.join("\n") : String(image.provisional_reference_text || "");
    cases.set(image.image_id, {
      material_id: image.image_id, source_kind: "image", source_id: image.image_id, source_en: "", source_sha256: image.sha256,
      image_sha256: image.sha256, image_ground_truth_en: gt, image_ground_truth_status: image.ground_truth_status,
      image_reference_text_kind: Array.isArray(image.ground_truth_text) ? "ground_truth_text" : "provisional_reference_text",
      image_role: image.role, image_layout_stratum: image.layout_stratum, source_record: image,
    });
  }
  return cases;
}

async function verifyTranslationComparison(reportRows, sourceRows, textRows, imageSelection, issues) {
  const cases = expectedMaterialCases(textRows, imageSelection);
  const sourceMap = mapRows(sourceRows, ["model_id", "case_id"], issues, "translation_results");
  const reportMap = mapRows(reportRows, ["model_id", "material_id"], issues, "translation_comparison");
  if (sourceRows.length !== 238 || reportRows.length !== 238 || reportMap.size !== 238) add(issues, "primary_translation_comparison_count", { source: sourceRows.length, report: reportRows.length, unique: reportMap.size });
  let sourceExact = 0; let targetExact = 0; let rawExact = 0; let rawMissing = 0; let rawMismatch = 0;
  const rawCache = new Map();
  for (const source of sourceRows) {
    const material = cases.get(source.case_id);
    const row = reportMap.get(key(source.model_id, source.case_id));
    if (!material || !row) { add(issues, "primary_translation_row_missing", { model_id: source.model_id, case_id: source.case_id }); continue; }
    const checks = {
      source_id: [row.source_id, source.source_id], source_kind: [row.source_kind, source.input_type], source_en: [row.source_en, material.source_en],
      source_sha256: [row.source_sha256, material.source_sha256], image_sha256: [row.image_sha256, material.image_sha256],
      target_zh_tw: [row.target_zh_tw, source.translation_zh_tw], model_id: [row.model_id, source.model_id], model_name: [row.model_name, source.model_name],
      model_alias: [row.model_alias, source.model_alias], translation_status: [row.translation_status, source.status],
    };
    if (material.source_kind === "image") {
      checks.image_ground_truth_en = [row.image_ground_truth_en, material.image_ground_truth_en];
      checks.image_ground_truth_status = [row.image_ground_truth_status, material.image_ground_truth_status];
      checks.image_reference_text_kind = [row.image_reference_text_kind, material.image_reference_text_kind];
      checks.image_role = [row.image_role, material.image_role];
      checks.image_layout_stratum = [row.image_layout_stratum, material.image_layout_stratum];
    }
    let exact = true;
    for (const [field, [actual, expected]] of Object.entries(checks)) {
      if (actual !== expected) { exact = false; add(issues, "primary_translation_field_mismatch", { model_id: source.model_id, case_id: source.case_id, field, actual, expected }); }
    }
    if (exact) sourceExact += 1;
    if (row.target_zh_tw === source.translation_zh_tw) targetExact += 1;
    const rawPath = source.case_record_path;
    try {
      if (!rawCache.has(rawPath)) rawCache.set(rawPath, parseJsonl((await stableRead(rawPath)).text, rawPath));
      const matches = rawCache.get(rawPath).filter((wrapper) => wrapper?.selected?.model_id === source.model_id && wrapper?.selected?.case_id === source.case_id);
      const selected = matches[matches.length - 1]?.selected;
      if (!selected) { rawMissing += 1; add(issues, "translation_raw_selected_missing", { model_id: source.model_id, case_id: source.case_id, raw_path: rawPath }); }
      else {
        const rawFields = { content: selected.content, source_sha256: selected.source_sha256, image_sha256: selected.image_sha256 || "", status: selected.status };
        if (rawFields.content !== source.translation_zh_tw || rawFields.source_sha256 !== source.source_sha256 || rawFields.image_sha256 !== source.image_sha256 || rawFields.status !== source.status) {
          rawMismatch += 1; add(issues, "translation_raw_selected_mismatch", { model_id: source.model_id, case_id: source.case_id, raw_path: rawPath });
        } else rawExact += 1;
      }
    } catch (error) { rawMissing += 1; add(issues, "translation_raw_unreadable", { raw_path: rawPath, error: String(error.message || error) }); }
  }
  const statuses = [...new Set((imageSelection.selected_images || []).map((x) => x.ground_truth_status))];
  const synthetic = (imageSelection.selected_images || []).filter((x) => String(x.ground_truth_status).startsWith("authored_rendered_text;")).length;
  const who = (imageSelection.selected_images || []).filter((x) => String(x.ground_truth_status).includes("not_human_gold")).length;
  const hasActualHumanGold = statuses.some((x) => /human_gold/i.test(String(x)) && !/not_human_gold/i.test(String(x)));
  if (synthetic !== 2 || who !== 2 || hasActualHumanGold) add(issues, "image_ground_truth_status_policy", { synthetic, who, statuses });
  const reportImageRows = reportRows.filter((x) => x.source_kind === "image");
  const reportStatuses = new Map(reportImageRows.map((x) => [x.material_id, x.image_ground_truth_status]));
  for (const image of imageSelection.selected_images || []) if (reportStatuses.get(image.image_id) !== image.ground_truth_status) add(issues, "report_image_ground_truth_status_mismatch", image.image_id);
  return { report_rows: reportRows.length, source_rows: sourceRows.length, source_exact: sourceExact, target_exact: targetExact, raw_exact: rawExact, raw_missing: rawMissing, raw_mismatch: rawMismatch, image_materials: { synthetic, who, statuses, has_actual_human_gold: hasActualHumanGold } };
}

function checkNoZeroImputation(pointwise, pairwise, canonical, comparison, reliability, bootstrap, lofo, issues) {
  const result = { pointwise_nonvalid_scores_blank: true, pairwise_nonvalid_fields_blank: true, comparison_nonvalid_scores_blank: true, partial_reliability_blank: true, partial_bootstrap_blank: true, lofo_no_partial_rank: true };
  const pointScoreFields = ["score_0_100", "completeness_format", "fluency", "semantic_safety", "terminology_locale"];
  for (const row of pointwise.rows) {
    const invalid = row.scalar_valid !== "true";
    if (invalid && pointScoreFields.some((field) => !isBlank(row[field]))) { result.pointwise_nonvalid_scores_blank = false; add(issues, "nonvalid_pointwise_score_not_blank", { judge_slot: row.judge_slot, job_id: row.job_id }); }
    if (row.scalar_valid === "true") {
      const score = numberOrNull(row.score_0_100);
      const subs = ["completeness_format", "fluency", "semantic_safety", "terminology_locale"].map((x) => numberOrNull(row[x]));
      if (score === null || subs.some((x) => x === null) || Math.round(subs[0] + subs[1] + subs[2] + subs[3]) !== score || score < 0 || score > 100) add(issues, "valid_pointwise_score_contract", { judge_slot: row.judge_slot, job_id: row.job_id });
    }
  }
  const pairFields = ["winner", "confidence", "critical_override", "reason_codes"];
  for (const row of pairwise.rows) if (row.valid !== "true" && pairFields.some((field) => !isBlank(row[field]))) { result.pairwise_nonvalid_fields_blank = false; add(issues, "nonvalid_pairwise_field_not_blank", { judge_slot: row.judge_slot, pair_job_id: row.pair_job_id }); }
  for (const row of canonical.rows) if (row.canonical_status !== "CONSISTENT" && row.canonical_status !== "CONSISTENT_TIE" && !isBlank(row.canonical_winner)) add(issues, "nonvalid_canonical_winner_not_blank", { judge_slot: row.judge_slot, pair_key: row.pair_key });
  for (const row of comparison.rows) {
    for (const judge of JUDGES) if (row[`${judge}_valid`] !== "true" && !isBlank(row[`${judge}_score`])) { result.comparison_nonvalid_scores_blank = false; add(issues, "comparison_nonvalid_score_not_blank", { material_id: row.material_id, model_id: row.model_id, judge }); }
    // A partial panel may expose an observed mean for its valid scores.  It
    // must not expose a fabricated value when no judge is valid.
    if (row.panel_n_valid === "0" && !isBlank(row.panel_mean)) { result.comparison_nonvalid_scores_blank = false; add(issues, "empty_panel_mean_not_blank", { material_id: row.material_id, model_id: row.model_id }); }
    if (row.panel_n_valid === "0" && !isBlank(row.panel_median)) { result.comparison_nonvalid_scores_blank = false; add(issues, "empty_panel_median_not_blank", { material_id: row.material_id, model_id: row.model_id }); }
    const validCount = JUDGES.filter((judge) => row[`${judge}_valid`] === "true").length;
    if (Number(row.panel_n_valid) !== validCount) add(issues, "comparison_panel_n_valid_mismatch", { material_id: row.material_id, model_id: row.model_id, panel_n_valid: row.panel_n_valid, validCount });
  }
  for (const row of reliability.rows) {
    for (const field of ["icc_2_1_absolute_agreement", "icc_2_k_absolute_agreement", "kendall_w", "kendall_w_unit_level", "kendall_w_system_level", "kendall_w_per_item_mean", "kendall_w_per_item_median", "kendall_w_per_item_min", "kendall_w_per_item_max", "ci_low", "ci_high"]) if (!isBlank(row[field])) { result.partial_reliability_blank = false; add(issues, "partial_reliability_metric_nonblank", { track: row.track, field, value: row[field] }); }
  }
  for (const row of bootstrap.rows) if (!isBlank(row.ci_low) || !isBlank(row.ci_high) || row.status !== "NOT_AVAILABLE_GATE_PARTIAL") { if (!isBlank(row.ci_low) || !isBlank(row.ci_high)) { result.partial_bootstrap_blank = false; add(issues, "partial_bootstrap_ci_nonblank", row); } }
  for (const row of lofo.rows) if (row.status !== "NOT_AVAILABLE_GATE_PARTIAL" || row.candidate_order_if_full !== "[]") { result.lofo_no_partial_rank = false; add(issues, "partial_lofo_rank_output", { track: row.track, omitted: row.omitted_judge_slot }); }
  return result;
}

function checkPointwisePairwiseAccounting(pointwise, pairwise, canonical, judgeSummary, manifest, progress, analysisReport, issues) {
  const result = { counts: {}, per_judge: {} };
  const pByJudge = new Map(JUDGES.map((j) => [j, pointwise.rows.filter((r) => r.judge_slot === j)]));
  const qByJudge = new Map(JUDGES.map((j) => [j, pairwise.rows.filter((r) => r.judge_slot === j)]));
  const pStatus = Object.fromEntries([...new Set(pointwise.rows.map((r) => r.status))].map((s) => [s, pointwise.rows.filter((r) => r.status === s).length]));
  const qStatus = Object.fromEntries([...new Set(pairwise.rows.map((r) => r.status))].map((s) => [s, pairwise.rows.filter((r) => r.status === s).length]));
  const pValid = pointwise.rows.filter((r) => r.scalar_valid === "true").length;
  const pTechnicalInvalid = pointwise.rows.filter((r) => r.technical_status === "fail").length;
  const qValid = pairwise.rows.filter((r) => r.valid === "true").length;
  const unsupportedPoint = pointwise.rows.filter((r) => r.status === "unsupported_by_design").length;
  const unsupportedPair = pairwise.rows.filter((r) => r.status === "unsupported_by_design").length;
  const terminalPoint = pointwise.rows.filter((r) => !["missing", "unsupported_by_design"].includes(r.status)).length;
  const terminalPair = pairwise.rows.filter((r) => !["missing", "unsupported_by_design"].includes(r.status)).length;
  result.counts = { pointwise_rows: pointwise.rows.length, pointwise_valid: pValid, pointwise_technical_invalid: pTechnicalInvalid, pointwise_status: pStatus, pointwise_unsupported: unsupportedPoint, pointwise_terminal: terminalPoint, pointwise_terminal_expected: 1190 - 28, pairwise_rows: pairwise.rows.length, pairwise_valid: qValid, pairwise_status: qStatus, pairwise_unsupported: unsupportedPair, pairwise_terminal: terminalPair, pairwise_terminal_expected: 360 - 24, canonical_rows: canonical.rows.length };
  if (pointwise.rows.length !== 1190 || pValid !== 39 || pTechnicalInvalid !== 8 || unsupportedPoint !== 28 || terminalPoint !== 47 || unsupportedPair !== 24 || pairwise.rows.length !== 360 || qValid !== 0 || terminalPair !== 0 || canonical.rows.length !== 180) add(issues, "global_partial_denominator_mismatch", result.counts);
  for (const judge of JUDGES) {
    const p = pByJudge.get(judge); const q = qByJudge.get(judge); const summary = judgeSummary.rows.find((r) => r.judge_slot === judge);
    if (!summary || p.length !== 238 || q.length !== 72 || new Set(p.map((r) => r.job_id)).size !== 238 || new Set(q.map((r) => r.pair_job_id)).size !== 72) add(issues, "per_judge_unique_count_mismatch", { judge, pointwise: p.length, pointwise_unique: new Set(p.map((r) => r.job_id)).size, pairwise: q.length, pairwise_unique: new Set(q.map((r) => r.pair_job_id)).size });
    if (summary) {
      const expected = { pointwise_expected: 238, pointwise_scalar_valid: p.filter((r) => r.scalar_valid === "true").length, pointwise_invalid_or_excluded: p.filter((r) => ["invalid", "technical_invalid"].includes(r.status)).length, pointwise_technical_invalid: p.filter((r) => r.technical_status === "fail").length, pointwise_blocked_or_missing: p.filter((r) => ["blocked", "missing"].includes(r.status)).length, pointwise_unsupported_by_design: p.filter((r) => r.status === "unsupported_by_design").length, pairwise_expected: 72, pairwise_valid_orientations: q.filter((r) => r.valid === "true").length, pairwise_invalid_or_missing: q.filter((r) => r.valid !== "true").length, pairwise_canonical_groups: canonical.rows.filter((r) => r.judge_slot === judge).length, pairwise_consistent: canonical.rows.filter((r) => r.judge_slot === judge && ["CONSISTENT", "CONSISTENT_TIE"].includes(r.canonical_status)).length, pairwise_inconsistent_abstain: canonical.rows.filter((r) => r.judge_slot === judge && r.canonical_status === "ABSTAIN_INCONSISTENT").length };
      for (const [field, value] of Object.entries(expected)) if (Number(summary[field]) !== value) add(issues, "judge_summary_mismatch", { judge, field, report: summary[field], expected: value });
      result.per_judge[judge] = expected;
    }
  }
  const manifestStatus = manifest.status || {};
  const expectedStatus = { pointwise_planned_rows: 1190, pointwise_scalar_valid_rows: 39, pointwise_technical_invalid_rows: 8, pairwise_planned_orientations: 360, pairwise_valid_orientations: 0, text_complete_units: 0, text_expected_units: 210, image_complete_units: 0, image_expected_units: 28, image_fixed_panel_k: 4 };
  for (const [field, value] of Object.entries(expectedStatus)) if (Number(manifestStatus[field]) !== value) add(issues, "analysis_manifest_status_mismatch", { field, report: manifestStatus[field], expected: value });
  const terminal = manifestStatus.terminal_coverage || {};
  const terminalExpected = { pointwise_terminal_jobs: 47, pointwise_terminal_expected_jobs: 1162, pairwise_terminal_jobs: 0, pairwise_terminal_expected_jobs: 336, pointwise_design_unsupported_jobs: 28, pairwise_design_unsupported_jobs: 24, all_primary_terminal: false };
  for (const [field, value] of Object.entries(terminalExpected)) if (terminal[field] !== value) add(issues, "terminal_coverage_mismatch", { field, report: terminal[field], expected: value });
  if (!String(analysisReport).includes("PARTIAL_PROGRESS_NO_RANKING_OR_ICC") || !String(progress).includes("PARTIAL_PROGRESS_NO_RANKING_OR_ICC")) add(issues, "partial_status_not_explicit_in_reports");
  if (!String(progress).includes("47/1162") || !String(progress).includes("0/336")) add(issues, "partial_terminal_denominator_not_explicit");
  return result;
}

function checkPartialNoRank(modelSummary, reliability, bootstrap, lofo, manifest, issues) {
  const result = { analysis_status: manifest.status?.analysis_status, all_rank_ineligible: true, reliability_blank: true, bootstrap_gated: true, lofo_gated: true };
  if (manifest.status?.analysis_status !== "PARTIAL_PROGRESS_NO_RANKING_OR_ICC") add(issues, "analysis_status_not_partial_gate", manifest.status?.analysis_status);
  for (const row of modelSummary.rows) {
    if (row.rank_eligible !== "false" || row.ranking_status !== "PROVISIONAL_NO_RANKING_OR_MEAN_COMPARISON") { result.all_rank_ineligible = false; add(issues, "model_rank_published_while_partial", { track: row.track, model_id: row.model_id, rank_eligible: row.rank_eligible, ranking_status: row.ranking_status }); }
  }
  for (const row of reliability.rows) if (row.analysis_status !== "NOT_AVAILABLE_GATE_PARTIAL") { result.reliability_blank = false; add(issues, "reliability_not_gated", row.track); }
  for (const row of bootstrap.rows) if (row.status !== "NOT_AVAILABLE_GATE_PARTIAL") { result.bootstrap_gated = false; add(issues, "bootstrap_not_gated", { track: row.track, model_id: row.model_id }); }
  for (const row of lofo.rows) if (row.status !== "NOT_AVAILABLE_GATE_PARTIAL") { result.lofo_gated = false; add(issues, "lofo_not_gated", { track: row.track, omitted: row.omitted_judge_slot }); }
  return result;
}

function checkRetryAndSelection(pointwise, invalidAnnotations, issues) {
  const result = { unique_rows: new Set(pointwise.rows.map((r) => key(r.judge_slot, r.job_id))).size, retry_rows: [], first_valid_retry_ok: true, technical_invalid_not_scored: true };
  for (const row of pointwise.rows) {
    if (Number(row.attempt_count) > 1) result.retry_rows.push({ judge_slot: row.judge_slot, job_id: row.job_id, attempt_count: Number(row.attempt_count), selected_attempt: row.selected_attempt, attempt_kind: row.attempt_kind, all_attempt_statuses: row.all_attempt_statuses });
    if (row.technical_status === "fail" && row.scalar_valid === "true") { result.technical_invalid_not_scored = false; add(issues, "technical_invalid_row_scored", { judge_slot: row.judge_slot, job_id: row.job_id }); }
  }
  if (result.retry_rows.length !== 1) { result.first_valid_retry_ok = false; add(issues, "unexpected_retry_row_count", result.retry_rows.length); }
  else {
    const retry = result.retry_rows[0];
    if (retry.judge_slot !== "judge-04" || retry.attempt_count !== 2 || retry.selected_attempt !== "1" || retry.attempt_kind !== "infrastructure_recovery" || !String(retry.all_attempt_statuses).includes("invalid") || !String(retry.all_attempt_statuses).includes("ok")) { result.first_valid_retry_ok = false; add(issues, "retry_selection_not_first_valid_infrastructure", retry); }
  }
  if (new Set(pointwise.rows.map((r) => key(r.judge_slot, r.job_id))).size !== pointwise.rows.length) add(issues, "pointwise_duplicate_after_resume");
  if (new Set(invalidAnnotations.rows.map((r) => key(r.judge_slot, r.request_id, r.attempt))).size !== invalidAnnotations.rows.length) add(issues, "invalid_annotation_duplicate");
  const preservedExcluded = invalidAnnotations.rows.filter((r) => r.preserved_excluded === "true").length;
  if (preservedExcluded !== invalidAnnotations.rows.length) add(issues, "invalid_annotation_not_preserved", { preservedExcluded, total: invalidAnnotations.rows.length });
  return result;
}

async function inspectRawSnapshot(reportManifest, issues) {
  const records = reportManifest.raw_snapshot_files || [];
  const result = { manifest_count: records.length, valid_hash_records: 0, missing: [], changed_since_report: [], extra_live_files: [], diagnostics: reportManifest.raw_snapshot_diagnostics || [] };
  const snapSet = new Set();
  for (const record of records) {
    const filePath = record.path; snapSet.add(filePath);
    if (!/^[0-9a-f]{64}$/i.test(String(record.sha256 || ""))) { add(issues, "raw_snapshot_hash_invalid", filePath); continue; }
    result.valid_hash_records += 1;
    if (!fsSync.existsSync(filePath)) { result.missing.push(filePath); add(issues, "raw_snapshot_file_missing", filePath); continue; }
    const actual = sha256Bytes(fsSync.readFileSync(filePath));
    if (actual !== record.sha256) result.changed_since_report.push(filePath);
  }
  const current = [];
  for (const root of [path.join(JUDGE_ROOT, "raw"), path.join(JUDGE_ROOT, "local_runner", "results")]) {
    if (!fsSync.existsSync(root)) continue;
    const walk = (dir) => {
      for (const entry of fsSync.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(p);
        else if (entry.isFile() && [".json", ".jsonl"].includes(path.extname(entry.name).toLowerCase())) {
          const lower = entry.name.toLowerCase();
          if (lower.endsWith(".summary.json") || p.split(path.sep).includes("technical_smoke") || lower.includes("server_cleanup") || lower.includes("preflight") || lower.includes("template_probe") || p.includes(`${path.sep}transport_schemas${path.sep}`)) return;
          current.push(p);
        }
      }
    };
    walk(root);
  }
  result.extra_live_files = current.filter((p) => !snapSet.has(p));
  // New raw arrivals and in-place checkpoint/append-only updates are expected
  // after a historical report snapshot; they do not mutate report hashes.
  result.historical_snapshot_manifest_preserved = result.missing.length === 0 && result.valid_hash_records === records.length && result.diagnostics.length === 0;
  return result;
}

function checkReportHashes(reportManifest, loadedReports, analysisReport, issues) {
  const result = { expected_report_list_match: false, report_hash_matches: 0, report_hash_total: 0, reports: {} };
  const expected = new Set([...REPORT_FILES, ...REQUIRED_REPORT_EXTRAS]);
  const listed = new Set(reportManifest.reports || []);
  result.expected_report_list_match = [...expected].every((x) => listed.has(x)) && listed.size === expected.size;
  if (!result.expected_report_list_match) add(issues, "analysis_manifest_report_list_mismatch", { listed: [...listed], expected: [...expected] });
  for (const [name, expectedHash] of Object.entries(reportManifest.report_sha256 || {})) {
    result.report_hash_total += 1;
    const report = loadedReports[name];
    const actual = report?.sha256 || (name === "analysis_report.md" ? sha256Text(analysisReport) : null);
    const matches = actual === expectedHash;
    result.reports[name] = { expected: expectedHash, actual, matches };
    if (matches) result.report_hash_matches += 1; else add(issues, "report_hash_mismatch", { name, expected: expectedHash, actual });
  }
  if (result.report_hash_total !== 16 || result.report_hash_matches !== result.report_hash_total) add(issues, "report_hash_coverage_mismatch", result);
  return result;
}

function checkLockSafety(analysisSource, lockPath, issues) {
  const result = { lock_file_exists: fsSync.existsSync(lockPath), flock_nonblocking: /fcntl\.flock/.test(analysisSource) && /LOCK_EX/.test(analysisSource) && /LOCK_NB/.test(analysisSource), atomic_write_present: /def atomic_write/.test(analysisSource), stable_read_present: /def read_json_stable/.test(analysisSource) };
  if (!result.lock_file_exists || !result.flock_nonblocking || !result.atomic_write_present || !result.stable_read_present) add(issues, "refresh_lock_or_stable_read_contract", result);
  return result;
}

function freshOfflineTests(issues) {
  const python = "/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
  const proc = spawnSync(python, ["-m", "unittest", "discover", "-s", path.join(ANALYSIS_ROOT, "tests"), "-p", "test_aggregate.py", "-v"], {
    cwd: BATCH_ROOT,
    encoding: "utf8",
    stdio: "pipe",
    timeout: 60000,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
  });
  const text = `${proc.stdout || ""}\n${proc.stderr || ""}`;
  const match = text.match(/Ran\s+(\d+)\s+tests?/);
  const passed = proc.status === 0 && !proc.error && Boolean(match && Number(match[1]) === 16 && /OK/.test(text));
  if (!passed) add(issues, proc.error ? "fresh_offline_tests_failed" : "fresh_offline_tests_count_or_status", text.slice(-1000));
  return { attempted: true, passed, tests: match ? Number(match[1]) : null, exit_status: proc.status, output_tail: text.slice(-1500) };
}

async function main() {
  const issues = [];
  const reportManifest = await readJson(REPORT_MANIFEST_PATH);
  const packetManifest = await readJson(PACKET_MANIFEST_PATH);
  const textRows = parseJsonl((await stableRead(path.join(MATERIAL_ROOT, "frozen_text_materials.jsonl"))).text, "frozen_text_materials.jsonl");
  const imageSelection = await readJson(path.join(MATERIAL_ROOT, "image_selection.json"));
  const loaded = {};
  for (const name of REPORT_FILES) loaded[name] = await loadCsv(name, issues);
  const analysisStable = await stableRead(path.join(REPORT_ROOT, "analysis_report.md"));
  const progressStable = await stableRead(path.join(JUDGE_ROOT, "progress_analysis.md"));
  const sourceCsv = await loadArtifactCsv(SOURCE_CSV_PATH, "translation_results", issues);
  const sourceRows = sourceCsv.rows;
  const reports = Object.fromEntries(Object.entries(loaded).map(([name, report]) => [name, report.rows]));
  const sourceAudit = await verifyTranslationComparison(reports["translation_comparison.csv"], sourceRows, textRows, imageSelection, issues);
  const noZero = checkNoZeroImputation(loaded["pointwise_results_long.csv"], loaded["pairwise_raw.csv"], loaded["pairwise_canonical.csv"], loaded["translation_comparison.csv"], loaded["reliability_summary.csv"], loaded["bootstrap_summary.csv"], loaded["lofo_summary.csv"], issues);
  const accounting = checkPointwisePairwiseAccounting(loaded["pointwise_results_long.csv"], loaded["pairwise_raw.csv"], loaded["pairwise_canonical.csv"], loaded["judge_summary.csv"], reportManifest, progressStable.text, analysisStable.text, issues);
  const partialGate = checkPartialNoRank(loaded["model_summary.csv"], loaded["reliability_summary.csv"], loaded["bootstrap_summary.csv"], loaded["lofo_summary.csv"], reportManifest, issues);
  const retries = checkRetryAndSelection(loaded["pointwise_results_long.csv"], loaded["invalid_annotations.csv"], issues);
  const rawSnapshot = await inspectRawSnapshot(reportManifest, issues);
  const loadedForHash = Object.fromEntries(Object.entries(loaded).map(([name, value]) => [name, value]));
  loadedForHash["analysis_report.md"] = analysisStable;
  loadedForHash["operational_retry_addendum.md"] = await stableRead(path.join(REPORT_ROOT, "operational_retry_addendum.md"));
  const reportHashes = checkReportHashes(reportManifest, loadedForHash, analysisStable.text, issues);
  const analysisSource = await stableRead(path.join(ANALYSIS_ROOT, "aggregate_new_panel.py"));
  const lockSafety = checkLockSafety(analysisSource.text, path.join(ANALYSIS_ROOT, ".refresh.lock"), issues);
  const freshTests = freshOfflineTests(issues);

  const frozenHashChecks = {
    frozen_text_sha256: sha256Bytes(await fs.readFile(path.join(MATERIAL_ROOT, "frozen_text_materials.jsonl"))),
    image_selection_sha256: sha256Bytes(await fs.readFile(path.join(MATERIAL_ROOT, "image_selection.json"))),
    material_manifest_sha256: sha256Bytes(await fs.readFile(path.join(MATERIAL_ROOT, "metadata_manifest.json"))),
    packet_manifest_sha256: sha256Bytes(await fs.readFile(PACKET_MANIFEST_PATH)),
    source_csv_sha256: sourceCsv.sha256,
  };
  for (const [field, actual] of Object.entries(frozenHashChecks)) if (reportManifest.frozen_inputs?.[field] !== actual) add(issues, "frozen_input_hash_mismatch", { field, report: reportManifest.frozen_inputs?.[field], actual });
  if (reportManifest.frozen_inputs?.root_approved !== true || !fsSync.existsSync(path.join(JUDGE_ROOT, "ROOT_APPROVED"))) add(issues, "root_approval_snapshot_missing");
  if (packetManifest.packet_id !== reportManifest.frozen_inputs?.packet_manifest_sha256 && false) add(issues, "unreachable_packet_check");
  const pointRows = loaded["pointwise_results_long.csv"].rows;
  const pairRows = loaded["pairwise_raw.csv"].rows;
  const comparisonRows = loaded["translation_comparison.csv"].rows;
  const imageFlagRows = comparisonRows.filter((r) => r.translation_untranslated_visible_english === "true");
  const modelFlag = imageFlagRows.map((r) => `${r.model_id}::${r.material_id}`);
  if (imageFlagRows.length !== 1 || modelFlag[0] !== "gemma4-31b-qat::img-09") add(issues, "image_quality_flag_semantic_mismatch", { count: imageFlagRows.length, rows: modelFlag });
  const result = {
    qa_name: "independent_report_evidence_qa",
    generated_at: new Date().toISOString(),
    read_only: true,
    judge_calls_made: false,
    scoring_or_regrade_started: false,
    status: issues.length ? "FAIL" : "PASS",
    report_snapshot_status: reportManifest.status,
    counts: { report_csv_count: REPORT_FILES.length, source_csv_rows: sourceRows.length, primary_comparison_rows: comparisonRows.length, pointwise_rows: pointRows.length, pairwise_rows: pairRows.length, canonical_rows: loaded["pairwise_canonical.csv"].rows.length, raw_snapshot_manifest_files: rawSnapshot.manifest_count, raw_live_extra_files: rawSnapshot.extra_live_files.length, raw_changed_since_report: rawSnapshot.changed_since_report.length },
    checks: {
      all_14_csv_artifacttool_pass: REPORT_FILES.every((name) => loaded[name].rows.length === EXPECTED_REPORT_ROWS[name] && loaded[name].stable),
      primary_comparison_238_exact_source_target_raw: sourceAudit.source_exact === 238 && sourceAudit.target_exact === 238 && sourceAudit.raw_exact === 238 && sourceAudit.raw_missing === 0 && sourceAudit.raw_mismatch === 0,
      image_ground_truth_status_preserved: sourceAudit.image_materials.synthetic === 2 && sourceAudit.image_materials.who === 2 && !sourceAudit.image_materials.has_actual_human_gold,
      no_zero_imputation: Object.values(noZero).every(Boolean),
      no_partial_icc_or_rank: partialGate.all_rank_ineligible && partialGate.reliability_blank && partialGate.bootstrap_gated && partialGate.lofo_gated,
      unique_deduplicated_judge_rows: retries.unique_rows === 1190 && retries.first_valid_retry_ok,
      first_valid_and_technical_invalid_separation: retries.technical_invalid_not_scored,
      denominators_1190_1162_360_336: accounting.counts.pointwise_rows === 1190 && accounting.counts.pointwise_terminal_expected === 1162 && accounting.counts.pairwise_rows === 360 && accounting.counts.pairwise_terminal_expected === 336,
      report_hashes_consistent: reportHashes.expected_report_list_match && reportHashes.report_hash_matches === reportHashes.report_hash_total,
      refresh_lock_stable_read_contract: lockSafety.lock_file_exists && lockSafety.flock_nonblocking && lockSafety.atomic_write_present && lockSafety.stable_read_present,
      historical_raw_snapshot_preserved: rawSnapshot.historical_snapshot_manifest_preserved,
      fresh_16_offline_tests_pass: freshTests.passed && freshTests.tests === 16,
      image_english_flag_is_diagnostic_only: imageFlagRows.length === 1 && imageFlagRows[0].model_id === "gemma4-31b-qat" && imageFlagRows[0].material_id === "img-09",
    },
    frozen_hash_checks: frozenHashChecks,
    artifact_csvs: Object.fromEntries(Object.entries(loaded).map(([name, value]) => [name, { rows: value.rows.length, columns: value.headers.length, used_range: value.usedRange, sha256: value.sha256, stable: value.stable }])),
    source_audit: sourceAudit,
    no_zero_imputation: noZero,
    accounting,
    partial_gate: partialGate,
    retry_selection: retries,
    raw_snapshot: rawSnapshot,
    report_hashes: reportHashes,
    lock_safety: lockSafety,
    fresh_offline_tests: freshTests,
    image_quality_flag_summary: { rows: imageFlagRows.length, rows_detail: imageFlagRows.map((r) => ({ model_id: r.model_id, material_id: r.material_id, span: r.translation_untranslated_visible_english_span })) },
    issue_count: issues.length,
    issue_codes: uniq(issues.map((x) => x.code)),
    issues: issues.slice(0, 200),
  };
  await fs.writeFile(path.join(QA_ROOT, "report_qa_results.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
  const lines = [
    "# Independent report evidence QA",
    "",
    `- Status: **${result.status}**; report snapshot status: **${reportManifest.status?.analysis_status || "unknown"}**.`,
    "- Read-only: true; no judge/model/API/server/LiteLLM calls; no scoring or regrade performed.",
    `- ArtifactTool CSVs: ${REPORT_FILES.length}/${REPORT_FILES.length} loaded; primary comparison ${sourceAudit.target_exact}/238 target cells exact and ${sourceAudit.raw_exact}/238 selected raw cells exact.`,
    `- Snapshot: ${rawSnapshot.manifest_count} historical raw entries; ${rawSnapshot.extra_live_files.length} live files arrived after the report snapshot; these are recorded but do not invalidate the historical manifest.`,
    "",
    "## Checks",
    "",
    ...Object.entries(result.checks).map(([name, pass]) => `- ${name}: **${pass ? "PASS" : "FAIL"}**`),
    "",
    "## Evidence notes",
    "",
    `- Partial denominators are explicit: pointwise ${accounting.counts.pointwise_terminal}/${accounting.counts.pointwise_terminal_expected} terminal primary jobs and pairwise ${accounting.counts.pairwise_terminal}/${accounting.counts.pairwise_terminal_expected}; planned totals remain 1190/360, with Mistral design-unsupported 28/24 excluded only from terminal denominators.`,
    `- The one GPT retry is retained as an infrastructure recovery: first attempt invalid, selected attempt 1 valid; no quality-preferred retry or zero imputation.`,
    `- The ${imageFlagRows.length} visible-English image flag is diagnostic only (Gemma 4 31B QAT / img-09); candidate text remains immutable and is not regraded here.`,
    rawSnapshot.changed_since_report.length ? `- ${rawSnapshot.changed_since_report.length} historical raw paths have changed on disk since the snapshot; recorded manifest hashes remain intact, so this is live-worker delta rather than a report-hash failure.` : "- No historical raw path changed on disk after the snapshot.",
    issues.length ? `- Issues: ${uniq(issues.map((x) => x.code)).join(", ")}` : "- No QA issues.",
  ];
  await fs.writeFile(path.join(QA_ROOT, "report_qa_report.md"), lines.join("\n") + "\n", "utf8");
  process.stdout.write(JSON.stringify({ status: result.status, issue_count: result.issue_count, issue_codes: result.issue_codes, checks: result.checks }, null, 2) + "\n");
}

await main();

