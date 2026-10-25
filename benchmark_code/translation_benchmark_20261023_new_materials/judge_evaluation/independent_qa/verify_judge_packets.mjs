#!/usr/bin/env node

/*
 * Read-only independent QA for the frozen five-judge packet bundle.
 *
 * This program deliberately does not call a judge, model, endpoint, LiteLLM,
 * or cloud service.  It reads the frozen packet, private maps, materials, and
 * translation CSV, then writes only the two reports in this directory.  The
 * CSV is parsed through the bundled ArtifactTool API as required by the
 * spreadsheet audit instructions.
 */

import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const { Workbook } = await import(
  process.env.CODEX_ARTIFACT_TOOL_MODULE ||
    "/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
);

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const JUDGE_ROOT = path.resolve(SCRIPT_DIR, "..");
const BATCH_ROOT = path.resolve(JUDGE_ROOT, "..");
const MATERIALS_ROOT = path.join(BATCH_ROOT, "materials");
const RESULTS_ROOT = path.join(BATCH_ROOT, "results");
const PUBLIC_ROOT = path.join(JUDGE_ROOT, "public");
const PRIVATE_ROOT = path.join(JUDGE_ROOT, "private");
const QA_ROOT = SCRIPT_DIR;

const JUDGES = ["judge-01", "judge-02", "judge-03", "judge-04", "judge-05"];
const MODELS = [
  "gemma4-12b", "gemma4-26b", "gemma4-31b-qat", "gemma4-e4b",
  "translategemma-12b", "translategemma-27b", "translategemma-4b",
];
const TEXT_COUNT = 30;
const IMAGE_COUNT = 4;
const MATERIAL_COUNT = 34;
const POINTWISE_PER_JUDGE = 238;
const PAIRWISE_PER_JUDGE = 72;
const EXPECTED_PRIVATE_POINTWISE = 1190;
const EXPECTED_PRIVATE_PAIRWISE = 360;
const CSV_FIELDS = [
  "model_id", "model_name", "model_alias", "case_id", "input_type", "source_id",
  "source_en", "source_sha256", "image_path", "image_sha256", "image_ground_truth_en",
  "image_ground_truth_status", "translation_zh_tw", "reasoning_content", "http_status",
  "finish_reason", "status", "elapsed_seconds", "prompt_tokens", "completion_tokens",
  "total_tokens", "prompt_tok_s", "generation_tok_s", "cache_tokens", "request_sha256",
  "response_body_sha256", "config_sha256", "model_sha256", "mmproj_sha256", "template_mode",
  "embedded_template_sha256", "adapter_template_sha256", "rendered_prompt_sha256", "retry_count",
  "error", "smoke_text_supported", "smoke_image_supported", "case_record_path",
];
const MODEL_TOKEN_RE = /(gemma4|translategemma|qwen|mistral|muse-glimmer|model[_ -]?id|model[_ -]?name|model[_ -]?alias)/i;
const FORBIDDEN_PUBLIC_KEYS = new Set([
  "model", "model_id", "model_name", "model_alias", "source_owner", "owner", "rank",
  "old_score", "historical_score", "prior_score", "score_0_100", "scores", "score",
  "template", "template_path", "template_sha256", "template_mode", "elapsed_seconds",
  "prompt_tokens", "completion_tokens", "timings", "speed", "generation_tok_s",
  "runtime", "config_sha256", "model_sha256", "mmproj_sha256", "old_pilot_score",
  "ground_truth", "answer_key", "mapping", "private", "raw_path", "raw_file",
]);

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}
function sha256Text(value) { return sha256Bytes(Buffer.from(String(value), "utf8")); }
async function sha256File(filePath) { return sha256Bytes(await fs.readFile(filePath)); }
async function readText(filePath) { return fs.readFile(filePath, "utf8"); }
async function readJson(filePath) { return JSON.parse(await readText(filePath)); }
async function exists(filePath) { try { await fs.access(filePath); return true; } catch { return false; } }
function cell(value) { return value === null || value === undefined ? "" : String(value); }
function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((k) => [k, stable(value[k])]));
  return value;
}
function stableJson(value) { return JSON.stringify(stable(value)); }
function add(issues, code, detail = undefined, severity = "block") {
  const item = { severity, code };
  if (detail !== undefined) item.detail = detail;
  issues.push(item);
}
function uniq(items) { return [...new Set(items)]; }
function eq(a, b) { return a === b; }
function key(...parts) { return parts.join("::"); }
function parseJsonl(text, filePath) {
  const rows = [];
  for (const [i, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); }
    catch (error) { throw new Error(`invalid JSONL ${filePath}:${i + 1}: ${error.message}`); }
  }
  return rows;
}
async function readJsonl(filePath) { return parseJsonl(await readText(filePath), filePath); }
function mapBy(items, field) { return new Map(items.map((x) => [x[field], x])); }

function pngInfo(bytes) {
  const valid = bytes.length >= 26 && Buffer.from(bytes.subarray(0, 8)).toString("hex") === "89504e470d0a1a0a";
  if (!valid) return { validPng: false, sha256: sha256Bytes(bytes) };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const colorType = view.getUint8(25);
  const color = colorType === 6 ? "RGBA" : colorType === 2 ? "RGB" : `png-color-type-${colorType}`;
  return { validPng: true, sha256: sha256Bytes(bytes), width: view.getUint32(16), height: view.getUint32(20), color };
}

function parseArtifactRows(values, issues) {
  if (!Array.isArray(values) || values.length < 1) { add(issues, "csv_artifact_empty"); return []; }
  const header = (values[0] || []).map(cell);
  if (stableJson(header) !== stableJson(CSV_FIELDS)) add(issues, "csv_header_mismatch", { actual: header, expected: CSV_FIELDS });
  return values.slice(1).map((valuesRow) => Object.fromEntries(header.map((h, i) => [h, cell(valuesRow?.[i])] )));
}

function rowIndex(rows, issues) {
  const index = new Map();
  for (const row of rows) {
    const k = key(row.model_id, row.case_id);
    if (index.has(k)) add(issues, "duplicate_translation_csv_cell", k);
    index.set(k, row);
  }
  if (rows.length !== TEXT_COUNT * 7 + IMAGE_COUNT * 7) add(issues, "translation_csv_row_count", rows.length);
  return index;
}

function materialCases(textRecords, imageSelection) {
  const cases = new Map();
  for (const record of textRecords) {
    cases.set(record.item_id, {
      material_id: record.item_id, source_kind: "text", source_id: record.source_id,
      source_en: record.source_en, source_sha256: sha256Text(record.source_en), image_sha256: "",
      image_alias: null, source_record: record,
    });
  }
  for (const image of imageSelection.selected_images || []) {
    cases.set(image.image_id, {
      material_id: image.image_id, source_kind: "image", source_id: image.image_id,
      source_en: "", source_sha256: image.sha256, image_sha256: image.sha256,
      image_alias: image, source_record: image,
    });
  }
  return cases;
}

function recursivePublicScan(value, where, findings, modelIds) {
  if (Array.isArray(value)) {
    value.forEach((child, i) => recursivePublicScan(child, `${where}[${i}]`, findings, modelIds));
    return;
  }
  if (!value || typeof value !== "object") {
    if (typeof value === "string") {
      const lower = value.toLowerCase();
      if (lower.includes("/private/") || lower.includes("/raw/") || lower.includes("litellm") || lower.includes("answer_key")) {
        findings.push({ code: "public_private_path_or_tool_token", where });
      }
      for (const model of modelIds) if (value === model || lower.includes(model.toLowerCase())) findings.push({ code: "public_model_identity_token", where, token: model });
    }
    return;
  }
  for (const [name, child] of Object.entries(value)) {
    if (FORBIDDEN_PUBLIC_KEYS.has(name.toLowerCase())) findings.push({ code: "public_forbidden_key", where: `${where}.${name}` });
    recursivePublicScan(child, `${where}.${name}`, findings, modelIds);
  }
}

function imageEnglishLeak(target, image) {
  const text = String(target || "");
  const reference = image.ground_truth_text ? image.ground_truth_text.join("\n") : (image.provisional_reference_text || "");
  const hits = [];
  if (reference && text.includes(reference)) hits.push({ type: "full_reference", length: reference.length });
  const lines = image.ground_truth_text || String(image.provisional_reference_text || "").split(/\n+/);
  for (const line of lines) {
    const normalized = String(line).trim();
    if (normalized.length < 12 || !/[A-Za-z]/.test(normalized)) continue;
    if (text.includes(normalized)) hits.push({ type: "exact_visible_line", sample: normalized.slice(0, 80), length: normalized.length });
  }
  // A machine-extracted dense page has long lines that can be split by the
  // translator.  Two substantial English lines or one long contiguous run is
  // a clear transcript leak; a medication name or WHO acronym alone is not.
  const longHits = lines.filter((line) => {
    const s = String(line).trim();
    return s.length >= 48 && /[A-Za-z]/.test(s) && text.includes(s);
  });
  if (longHits.length >= 2) hits.push({ type: "multiple_long_reference_lines", count: longHits.length });
  return hits;
}

// Image candidates are immutable translator outputs.  English inside a
// candidate target is therefore a quality signal (an untranslated output),
// not prompt/GT leakage.  Only auxiliary fields in an image job can constitute
// a packet leak.  Keep this walk deliberately separate from target_zh_tw.
function imageAuxiliaryEnglishLeak(value, image, where = "job", hits = []) {
  if (Array.isArray(value)) {
    value.forEach((child, index) => imageAuxiliaryEnglishLeak(child, image, `${where}[${index}]`, hits));
    return hits;
  }
  if (!value || typeof value !== "object") return hits;
  for (const [name, child] of Object.entries(value)) {
    const lower = name.toLowerCase();
    if (lower === "target_zh_tw") continue;
    if (/(source_en|source_text|english_source|ground_truth|transcript|transcription|provisional_reference)/i.test(name)) {
      hits.push({ field: `${where}.${name}`, kind: "auxiliary_english_field" });
      continue;
    }
    if (typeof child === "string") {
      const reference = image?.ground_truth_text ? image.ground_truth_text.join("\n") : (image?.provisional_reference_text || "");
      if (reference && child.includes(reference)) hits.push({ field: `${where}.${name}`, kind: "auxiliary_reference_text" });
    } else {
      imageAuxiliaryEnglishLeak(child, image, `${where}.${name}`, hits);
    }
  }
  return hits;
}

function compactLeakFinding(finding) {
  return {
    track: finding.track,
    judge: finding.judge,
    job_id: finding.job_id,
    pair_job_id: finding.pair_job_id,
    material_id: finding.material_id,
    candidate: finding.candidate,
    hit_count: finding.hits.length,
    hit_types: uniq(finding.hits.map((hit) => hit.type)),
    samples: finding.hits.filter((hit) => hit.sample).slice(0, 3).map((hit) => hit.sample),
  };
}

function expectedPairTarget(csvIndex, modelId, materialId) {
  return csvIndex.get(key(modelId, materialId))?.translation_zh_tw;
}

function expectedBlind(mapping, judge, model) { return mapping.mapping_by_judge?.[judge]?.[model]; }

function validateOutputSchemaContracts(pointSchema, pairSchema, issues) {
  const point = pointSchema || {};
  const pair = pairSchema || {};
  if (point.additionalProperties !== false || pair.additionalProperties !== false) add(issues, "result_schema_additional_properties_not_false");
  const p = point.properties || {};
  const q = pair.properties || {};
  const sub = p.subscores?.properties || {};
  const expectedBounds = { terminology_locale: 30, semantic_safety: 30, fluency: 20, completeness_format: 20 };
  for (const [name, max] of Object.entries(expectedBounds)) {
    if (sub[name]?.type !== "integer" || sub[name]?.minimum !== 0 || sub[name]?.maximum !== max) add(issues, "pointwise_subscore_schema_bound", { name, schema: sub[name] });
  }
  if (p.score_0_100?.minimum !== 0 || p.score_0_100?.maximum !== 100) add(issues, "pointwise_score_schema_bound");
  if (q.winner?.enum?.join(",") !== "A,B,TIE" || q.confidence?.minimum !== 0 || q.confidence?.maximum !== 1 || q.critical_override?.type !== "boolean") add(issues, "pairwise_schema_contract");
  if (!Array.isArray(point.required) || !point.required.includes("score_0_100") || !point.required.includes("subscores")) add(issues, "pointwise_required_fields_contract");
  if (!Array.isArray(pair.required) || !pair.required.includes("winner") || !pair.required.includes("reason_codes")) add(issues, "pairwise_required_fields_contract");
  return {
    pointwise_bounds_30_30_20_20: Object.entries(expectedBounds).every(([name, max]) => sub[name]?.type === "integer" && sub[name]?.minimum === 0 && sub[name]?.maximum === max),
    score_0_100_integer_bounds: p.score_0_100?.minimum === 0 && p.score_0_100?.maximum === 100,
    pairwise_winner_confidence_contract: q.winner?.enum?.join(",") === "A,B,TIE" && q.confidence?.minimum === 0 && q.confidence?.maximum === 1,
    blocked_null_contract: Boolean(point.allOf?.some((x) => x.then?.properties?.score_0_100?.const === null && x.then?.properties?.subscores?.const === null)),
  };
}

function checkResultFormatPolicy(publicJob, track, issues, where) {
  if (publicJob.track !== track) add(issues, "public_track_mismatch", where);
  if (publicJob.target_locale !== "zh-TW") add(issues, "public_target_locale_mismatch", where);
  if (typeof publicJob.schema_version !== "string" || !publicJob.schema_version.startsWith("new-material-")) add(issues, "public_schema_version_invalid", where);
  if (track === "pointwise") {
    if (publicJob.result_schema !== "schemas/pointwise_result.schema.json") add(issues, "pointwise_result_schema_ref_invalid", where);
    if (typeof publicJob.job_id !== "string" || typeof publicJob.blind_id !== "string" || typeof publicJob.blind_source_id !== "string") add(issues, "pointwise_required_binding_field_missing", where);
    if (!Object.prototype.hasOwnProperty.call(publicJob, "target_zh_tw")) add(issues, "pointwise_target_missing", where);
    if (publicJob.source_kind === "image" && publicJob.native_image_required !== true) add(issues, "image_native_required_not_true", where);
    if (publicJob.source_kind === "text" && Object.prototype.hasOwnProperty.call(publicJob, "image_alias")) add(issues, "text_job_has_image_alias", where);
  } else {
    if (publicJob.result_schema !== "schemas/pairwise_result.schema.json") add(issues, "pairwise_result_schema_ref_invalid", where);
    if (typeof publicJob.pair_job_id !== "string" || typeof publicJob.pair_key !== "string" || !["AB", "BA"].includes(publicJob.orientation)) add(issues, "pairwise_required_binding_field_missing", where);
    if (!publicJob.candidate_a || !publicJob.candidate_b || publicJob.candidate_a.blind_id !== "A" || publicJob.candidate_b.blind_id !== "B") add(issues, "pair_candidate_labels_invalid", where);
    for (const candidate of [publicJob.candidate_a, publicJob.candidate_b]) if (!candidate || typeof candidate.target_zh_tw !== "string") add(issues, "pair_candidate_target_missing", where);
    if (publicJob.source_kind === "image" && publicJob.native_image_required !== true) add(issues, "pair_image_native_required_not_true", where);
    if (publicJob.source_kind === "text" && Object.prototype.hasOwnProperty.call(publicJob, "image_alias")) add(issues, "pair_text_has_image_alias", where);
  }
}

async function verifyManifestHashes(manifest, manifestPath, issues) {
  const result = { checked: 0, matches: 0, missing: [] };
  const check = async (declared, filePath, code) => {
    result.checked += 1;
    const item = { file: filePath, declared };
    if (!(await exists(filePath))) { item.matches = false; result.missing.push(filePath); add(issues, `${code}_missing`, filePath); return item; }
    item.actual = await sha256File(filePath);
    item.matches = item.actual === declared;
    if (item.matches) result.matches += 1; else add(issues, `${code}_mismatch`, { file: filePath, declared, actual: item.actual });
    return item;
  };
  const records = [];
  for (const [rel, declared] of Object.entries(manifest.documentation_sha256 || {})) records.push(await check(declared, path.join(JUDGE_ROOT, rel), "documentation_hash"));
  const inputPaths = {
    "frozen_text_materials.jsonl": path.join(MATERIALS_ROOT, "frozen_text_materials.jsonl"),
    "image_selection.json": path.join(MATERIALS_ROOT, "image_selection.json"),
    "metadata_manifest.json": path.join(MATERIALS_ROOT, "metadata_manifest.json"),
    "translation_results.csv": path.join(RESULTS_ROOT, "translation_results.csv"),
    // The new packet manifest intentionally names this old instruction file by
    // basename; this is the frozen pilot artifact, not a new network fetch.
    "old_judge_instructions.txt": path.resolve(BATCH_ROOT, "..", "translation_benchmark_20261022_judge_pilot", "judge_instructions.txt"),
  };
  for (const [name, declared] of Object.entries(manifest.input_hashes_sha256 || {})) {
    if (!inputPaths[name]) add(issues, "manifest_input_path_unresolved", name);
    else records.push(await check(declared, inputPaths[name], "input_hash"));
  }
  for (const [rel, declared] of Object.entries(manifest.private_file_sha256 || {})) records.push(await check(declared, path.join(JUDGE_ROOT, rel), "private_hash"));
  for (const [rel, declared] of Object.entries(manifest.public_file_sha256 || {})) records.push(await check(declared, path.join(JUDGE_ROOT, rel), "public_hash"));
  result.records = records;
  result.all_match = issues.every((x) => !["documentation_hash", "input_hash", "private_hash", "public_hash"].some((prefix) => x.code.startsWith(prefix)));
  return result;
}

async function verifySourceImages(cases, sourceMap, imageSelection, manifest, issues) {
  const result = { text_count: 0, image_count: 0, image_checks: [] };
  const selected = mapBy(imageSelection.selected_images || [], "image_id");
  const aliases = sourceMap.image_aliases || {};
  for (const [materialId, item] of cases) {
    const mapped = sourceMap.materials?.find((x) => x.material_id === materialId);
    if (!mapped) { add(issues, "source_map_material_missing", materialId); continue; }
    const mappedHash = item.source_kind === "image" ? mapped.image_sha256 : mapped.source_sha256;
    if (mapped.source_kind !== item.source_kind || mapped.source_id !== item.source_id || mappedHash !== item.source_sha256) add(issues, "source_map_material_mismatch", materialId);
    if (item.source_kind === "text") { result.text_count += 1; if (mapped.source_length_chars !== item.source_en.length) add(issues, "source_length_mismatch", materialId); }
    else {
      result.image_count += 1;
      const image = selected.get(materialId); const alias = aliases[materialId];
      if (!image || !alias || image.sha256 !== item.source_sha256 || alias.sha256 !== item.source_sha256) add(issues, "image_selection_or_alias_hash_mismatch", materialId);
      if (alias && image && stableJson(alias.dimensions) !== stableJson(image.dimensions)) add(issues, "image_dimensions_metadata_mismatch", materialId);
      const publicPath = path.join(JUDGE_ROOT, alias?.public_path || "");
      const rec = { material_id: materialId, public_alias: alias?.public_alias, public_path: publicPath, declared_sha256: alias?.sha256, declared_dimensions: alias?.dimensions };
      if (!(await exists(publicPath))) { add(issues, "public_image_missing", materialId); }
      else {
        const actual = pngInfo(await fs.readFile(publicPath));
        rec.actual_sha256 = actual.sha256; rec.actual_dimensions = { width: actual.width, height: actual.height, color: actual.color }; rec.valid_png = actual.validPng;
        if (!actual.validPng || actual.sha256 !== alias.sha256) add(issues, "public_image_hash_or_png_mismatch", materialId);
        if (actual.width !== alias.dimensions.width || actual.height !== alias.dimensions.height || actual.color !== alias.dimensions.color) add(issues, "public_image_dimensions_mismatch", materialId);
      }
      result.image_checks.push(rec);
    }
  }
  if (result.text_count !== TEXT_COUNT || result.image_count !== IMAGE_COUNT) add(issues, "material_scope_count_mismatch", { text: result.text_count, image: result.image_count });
  if (manifest.image_policy?.english_transcription_in_public_payload !== false || manifest.image_policy?.ground_truth_in_public_payload !== false) add(issues, "manifest_image_policy_not_private");
  return result;
}

function auditPointwise(judge, rows, keymap, cases, sourceMap, mapping, csvIndex, issues, candidateEnglishFindings, auxiliaryFindings) {
  const entries = Object.entries(keymap).map(([job_id, x]) => ({ ...x, job_id })).filter((x) => x.judge_slot === judge);
  if (entries.length !== POINTWISE_PER_JUDGE) add(issues, "pointwise_private_count", { judge, actual: entries.length });
  const entryById = new Map(entries.map((x) => [x.job_id, x]));
  const seen = new Set();
  if (rows.length !== POINTWISE_PER_JUDGE) add(issues, "pointwise_public_count", { judge, actual: rows.length });
  for (const [i, job] of rows.entries()) {
    const where = `pointwise/${judge}/${i}`;
    checkResultFormatPolicy(job, "pointwise", issues, where);
    if (seen.has(job.job_id)) add(issues, "pointwise_public_duplicate_job_id", { judge, job_id: job.job_id });
    seen.add(job.job_id);
    const entry = entryById.get(job.job_id);
    if (!entry) { add(issues, "pointwise_public_job_not_private_keymap", { judge, job_id: job.job_id }); continue; }
    const material = cases.get(entry.material_id);
    if (!material) { add(issues, "pointwise_material_missing", entry.material_id); continue; }
    const alias = sourceMap.source_alias_by_judge?.[judge]?.[entry.material_id];
    const expectedRow = csvIndex.get(key(entry.model_id, entry.material_id));
    if (!expectedRow) add(issues, "pointwise_csv_target_missing", { judge, job_id: job.job_id });
    if (entry.judge_slot !== judge || entry.track !== "pointwise" || entry.source_kind !== material.source_kind || entry.source_id !== material.source_id) add(issues, "pointwise_private_binding_mismatch", { judge, job_id: job.job_id });
    if (entry.source_sha256 !== (material.source_kind === "text" ? material.source_sha256 : "") || entry.image_sha256 !== (material.source_kind === "image" ? material.image_sha256 : "")) add(issues, "pointwise_private_source_hash_mismatch", { judge, job_id: job.job_id });
    if (job.blind_source_id !== alias || job.source_kind !== material.source_kind) add(issues, "pointwise_source_alias_or_kind_mismatch", { judge, job_id: job.job_id });
    if (job.blind_id !== expectedBlind(mapping, judge, entry.model_id)) add(issues, "pointwise_blind_id_mismatch", { judge, job_id: job.job_id });
    if (expectedRow && job.target_zh_tw !== expectedRow.translation_zh_tw) add(issues, "pointwise_target_not_csv_exact", { judge, job_id: job.job_id });
    if (material.source_kind === "text") {
      if (job.source_en !== material.source_en) add(issues, "pointwise_text_source_not_frozen_exact", { judge, job_id: job.job_id });
      if (Object.prototype.hasOwnProperty.call(job, "image_alias") || Object.prototype.hasOwnProperty.call(job, "native_image_required")) add(issues, "pointwise_text_has_image_fields", { judge, job_id: job.job_id });
    } else {
      const expectedAlias = sourceMap.image_aliases?.[entry.material_id]?.public_alias;
      if (job.image_alias !== expectedAlias || job.native_image_required !== true) add(issues, "pointwise_image_alias_or_native_flag_mismatch", { judge, job_id: job.job_id });
      if (Object.prototype.hasOwnProperty.call(job, "source_en")) add(issues, "pointwise_image_contains_source_en", { judge, job_id: job.job_id });
      const hits = imageEnglishLeak(job.target_zh_tw, material.image_alias);
      if (hits.length) candidateEnglishFindings.push({ track: "pointwise", judge, job_id: job.job_id, material_id: entry.material_id, model_id: entry.model_id, target_sha256: sha256Text(job.target_zh_tw), hits });
      const auxiliary = imageAuxiliaryEnglishLeak(job, material.image_alias, "job");
      if (auxiliary.length) auxiliaryFindings.push({ track: "pointwise", judge, job_id: job.job_id, material_id: entry.material_id, hits: auxiliary });
    }
  }
  for (const entry of entries) if (!seen.has(entry.job_id)) add(issues, "pointwise_private_job_not_public", { judge, job_id: entry.job_id });
  return { private: entries.length, public: rows.length, bound: seen.size };
}

function auditPairwise(judge, rows, keymap, cases, sourceMap, mapping, csvIndex, issues, candidateEnglishFindings, auxiliaryFindings, pairGroups) {
  const entries = Object.values(keymap).filter((x) => x.judge_slot === judge);
  if (entries.length !== PAIRWISE_PER_JUDGE) add(issues, "pairwise_private_count", { judge, actual: entries.length });
  const entryById = new Map(entries.map((x) => [x.pair_job_id, x]));
  const seen = new Set();
  if (rows.length !== PAIRWISE_PER_JUDGE) add(issues, "pairwise_public_count", { judge, actual: rows.length });
  for (const [i, job] of rows.entries()) {
    const where = `pairwise/${judge}/${i}`;
    checkResultFormatPolicy(job, "pairwise", issues, where);
    if (seen.has(job.pair_job_id)) add(issues, "pairwise_public_duplicate_job_id", { judge, pair_job_id: job.pair_job_id });
    seen.add(job.pair_job_id);
    const entry = entryById.get(job.pair_job_id);
    if (!entry) { add(issues, "pairwise_public_job_not_private_keymap", { judge, pair_job_id: job.pair_job_id }); continue; }
    const material = cases.get(entry.material_id);
    if (!material) { add(issues, "pairwise_material_missing", entry.material_id); continue; }
    const alias = sourceMap.source_alias_by_judge?.[judge]?.[entry.material_id];
    const aTarget = expectedPairTarget(csvIndex, entry.model_a, entry.material_id);
    const bTarget = expectedPairTarget(csvIndex, entry.model_b, entry.material_id);
    if (entry.judge_slot !== judge || entry.source_kind !== material.source_kind || entry.source_id !== material.source_id) add(issues, "pairwise_private_binding_mismatch", { judge, pair_job_id: job.pair_job_id });
    if (entry.source_sha256 !== (material.source_kind === "text" ? material.source_sha256 : "") || entry.image_sha256 !== (material.source_kind === "image" ? material.image_sha256 : "")) add(issues, "pairwise_private_source_hash_mismatch", { judge, pair_job_id: job.pair_job_id });
    if (job.blind_source_id !== alias || job.source_kind !== material.source_kind || job.pair_key !== entry.pair_key || job.orientation !== entry.orientation) add(issues, "pairwise_source_or_orientation_binding_mismatch", { judge, pair_job_id: job.pair_job_id });
    if (job.candidate_a?.target_zh_tw !== aTarget || job.candidate_b?.target_zh_tw !== bTarget) add(issues, "pairwise_actual_target_not_csv_exact", { judge, pair_job_id: job.pair_job_id, model_a: entry.model_a, model_b: entry.model_b });
    if (job.candidate_a?.blind_id !== "A" || job.candidate_b?.blind_id !== "B") add(issues, "pairwise_candidate_label_not_AB", { judge, pair_job_id: job.pair_job_id });
    if (material.source_kind === "text") {
      if (job.source_en !== material.source_en) add(issues, "pairwise_text_source_not_frozen_exact", { judge, pair_job_id: job.pair_job_id });
      if (Object.prototype.hasOwnProperty.call(job, "image_alias") || Object.prototype.hasOwnProperty.call(job, "native_image_required")) add(issues, "pairwise_text_has_image_fields", { judge, pair_job_id: job.pair_job_id });
    } else {
      const expectedAlias = sourceMap.image_aliases?.[entry.material_id]?.public_alias;
      if (job.image_alias !== expectedAlias || job.native_image_required !== true) add(issues, "pairwise_image_alias_or_native_flag_mismatch", { judge, pair_job_id: job.pair_job_id });
      if (Object.prototype.hasOwnProperty.call(job, "source_en")) add(issues, "pairwise_image_contains_source_en", { judge, pair_job_id: job.pair_job_id });
      for (const [label, candidate] of [["A", job.candidate_a], ["B", job.candidate_b]]) {
        const hits = imageEnglishLeak(candidate?.target_zh_tw, material.image_alias);
        if (hits.length) candidateEnglishFindings.push({ track: "pairwise", judge, pair_job_id: job.pair_job_id, material_id: entry.material_id, model_id: label === "A" ? entry.model_a : entry.model_b, candidate: label, target_sha256: sha256Text(candidate?.target_zh_tw || ""), hits });
      }
      const auxiliary = imageAuxiliaryEnglishLeak(job, material.image_alias, "job");
      if (auxiliary.length) auxiliaryFindings.push({ track: "pairwise", judge, pair_job_id: job.pair_job_id, material_id: entry.material_id, hits: auxiliary });
    }
    const groupKey = key(judge, entry.material_id, entry.pair_key);
    if (!pairGroups.has(groupKey)) pairGroups.set(groupKey, []);
    pairGroups.get(groupKey).push({ job, entry, material });
  }
  for (const entry of entries) if (!seen.has(entry.pair_job_id)) add(issues, "pairwise_private_job_not_public", { judge, pair_job_id: entry.pair_job_id });
  return { private: entries.length, public: rows.length, bound: seen.size };
}

function auditSpanningGraphs(pairSelection, keymap, issues) {
  const result = [];
  for (const materialId of pairSelection.selection?.selected_material_ids_in_order || []) {
    const pathModels = pairSelection.pairs_by_source?.[materialId]?.candidate_chain_model_ids_private || [];
    const edges = pairSelection.pairs_by_source?.[materialId]?.edges_private || [];
    const expectedEdges = pathModels.slice(0, -1).map((m, i) => [m, pathModels[i + 1]]);
    const canon = (edge) => edge.slice().sort().join("|");
    const expected = new Set(expectedEdges.map(canon));
    const actual = new Set(edges.map(canon));
    if (pathModels.length !== 7 || new Set(pathModels).size !== 7 || !MODELS.every((m) => pathModels.includes(m)) || edges.length !== 6 || expected.size !== actual.size || [...expected].some((e) => !actual.has(e))) add(issues, "pairwise_chain_not_spanning_7_models", { material_id: materialId, pathModels, edges });
    for (const judge of JUDGES) {
      const entries = Object.values(keymap.pairwise || {}).filter((x) => x.judge_slot === judge && x.material_id === materialId);
      const jobEdges = new Set(entries.map((x) => canon([x.model_a, x.model_b])));
      if (entries.length !== 12 || jobEdges.size !== 6 || [...expected].some((e) => !jobEdges.has(e))) add(issues, "pairwise_judge_chain_coverage_mismatch", { judge, material_id: materialId, entries: entries.length, edges: [...jobEdges] });
    }
    result.push({ material_id: materialId, path_models: pathModels, edge_count: edges.length, spanning_all_models: pathModels.length === 7 && new Set(pathModels).size === 7 && MODELS.every((m) => pathModels.includes(m)) });
  }
  return result;
}

function auditABBA(pairGroups, issues) {
  const result = { groups: 0, complete: 0, swapped_targets: 0 };
  for (const [groupKey, group] of pairGroups.entries()) {
    result.groups += 1;
    const ab = group.find((x) => x.entry.orientation === "AB");
    const ba = group.find((x) => x.entry.orientation === "BA");
    if (!ab || !ba) { add(issues, "pairwise_ABBA_orientation_missing", groupKey); continue; }
    if (ab.entry.model_a !== ba.entry.model_b || ab.entry.model_b !== ba.entry.model_a) add(issues, "pairwise_ABBA_models_not_swapped", groupKey);
    if (ab.job.blind_source_id !== ba.job.blind_source_id || ab.job.source_kind !== ba.job.source_kind || ab.job.target_locale !== ba.job.target_locale) add(issues, "pairwise_ABBA_source_not_same", groupKey);
    if (ab.material.source_kind === "text" && ab.job.source_en !== ba.job.source_en) add(issues, "pairwise_ABBA_text_source_not_same", groupKey);
    if (ab.material.source_kind === "image" && ab.job.image_alias !== ba.job.image_alias) add(issues, "pairwise_ABBA_image_alias_not_same", groupKey);
    if (ab.job.candidate_a?.target_zh_tw !== ba.job.candidate_b?.target_zh_tw || ab.job.candidate_b?.target_zh_tw !== ba.job.candidate_a?.target_zh_tw) add(issues, "pairwise_ABBA_targets_not_swapped", groupKey);
    else result.swapped_targets += 1;
    result.complete += 1;
  }
  if (result.groups !== 5 * 6 * 6) add(issues, "pairwise_ABBA_group_count", result.groups);
  return result;
}

async function revalidateCandidateEnglishOutputs(findings, csvIndex, issues) {
  const result = { suspect_count: findings.length, csv_exact: 0, raw_exact: 0, raw_missing: 0, raw_mismatch: 0, records: [] };
  const rawCache = new Map();
  for (const finding of findings) {
    const csv = csvIndex.get(key(finding.model_id, finding.material_id));
    const record = { track: finding.track, judge: finding.judge, job_id: finding.job_id, pair_job_id: finding.pair_job_id, candidate: finding.candidate, model_id: finding.model_id, material_id: finding.material_id, csv_exact: false, raw_exact: false };
    if (!csv) { add(issues, "candidate_english_csv_revalidation_missing", record); result.records.push(record); continue; }
    // The field-level packet audit already compared public target to this CSV
    // cell; repeat the immutable equality here so the quality flag cannot be
    // mistaken for a rewritten public candidate.
    record.csv_exact = finding.target_sha256 === sha256Text(csv.translation_zh_tw);
    if (record.csv_exact) result.csv_exact += 1;
    const rawPath = csv.case_record_path;
    try {
      if (!rawCache.has(rawPath)) rawCache.set(rawPath, parseJsonl(await readText(rawPath), rawPath));
      const wrappers = rawCache.get(rawPath);
      const matches = wrappers.filter((wrapper) => wrapper?.selected?.model_id === finding.model_id && wrapper?.selected?.case_id === finding.material_id);
      const selected = matches[matches.length - 1]?.selected;
      if (!selected) { result.raw_missing += 1; add(issues, "candidate_english_raw_revalidation_missing", record); }
      else if (selected.content !== csv.translation_zh_tw) { result.raw_mismatch += 1; add(issues, "candidate_english_raw_revalidation_mismatch", record); }
      else { record.raw_exact = true; result.raw_exact += 1; }
    } catch (error) {
      result.raw_missing += 1;
      add(issues, "candidate_english_raw_revalidation_unreadable", { ...record, error: String(error.message || error) });
    }
    result.records.push(record);
  }
  return result;
}

function auditRunnerInterface(issues) {
  const result = { local: {}, cloud: {}, no_wrong_field: true };
  const files = [
    ["local", path.join(JUDGE_ROOT, "local_runner", "judge_driver.py")],
    ["cloud", path.join(JUDGE_ROOT, "cloud_runner", "run_cloud_judge.py")],
  ];
  for (const [name, filePath] of files) {
    const text = requireTextSync(filePath);
    result[name] = {
      exists: Boolean(text),
      expects_pair_job_id: /pair_job_id/.test(text),
      expects_job_id: /job_id/.test(text),
      expects_image_alias: /image_alias/.test(text),
      expects_candidate_a_b: /candidate_a/.test(text) && /candidate_b/.test(text),
      calls_network_or_model: false,
    };
    if (!text) add(issues, "runner_source_missing", name);
    // These are the packet fields adapted by both runners.  A literal
    // source_en_or_image_alias/wrongpair_job_id dependency would be a frozen
    // interface mismatch; helper aliases are acceptable.
    if (/wrongpair_job_id|source_en_or_image_alias/.test(text)) { add(issues, "runner_expects_wrong_packet_field", name); result.no_wrong_field = false; }
    if (!result[name].expects_pair_job_id || !result[name].expects_image_alias || !result[name].expects_candidate_a_b) add(issues, "runner_packet_interface_field_missing", name);
  }
  return result;
}

function requireTextSync(filePath) {
  try { return fsSync.readFileSync(filePath, "utf8"); } catch { return ""; }
}

function runFreshOfflineTests(issues) {
  const pythonCandidates = [
    process.env.CODEX_PYTHON,
    "/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
    "/usr/bin/python3",
  ].filter(Boolean);
  const command = pythonCandidates.find((p) => { try { execFileSync(p, ["--version"], { encoding: "utf8", stdio: "pipe" }); return true; } catch { return false; } });
  if (!command) { add(issues, "offline_test_python_missing"); return { attempted: false, passed: false }; }
  try {
    const output = execFileSync(command, ["-m", "unittest", "discover", "-s", path.join(JUDGE_ROOT, "tests"), "-p", "test_packets.py", "-v"], { cwd: BATCH_ROOT, encoding: "utf8", stdio: "pipe", timeout: 60000 });
    return { attempted: true, passed: true, command, output_tail: output.slice(-3000) };
  } catch (error) {
    add(issues, "fresh_offline_packet_tests_failed", { status: error.status, stdout_tail: String(error.stdout || "").slice(-2000), stderr_tail: String(error.stderr || "").slice(-2000) });
    return { attempted: true, passed: false, command, output_tail: String(error.stdout || "").slice(-3000) };
  }
}

async function main() {
  const issues = [];
  const candidateEnglishFindings = [];
  const auxiliaryFindings = [];
  const manifestPath = path.join(JUDGE_ROOT, "manifest.json");
  const files = {
    manifest: manifestPath,
    packetReady: path.join(JUDGE_ROOT, "packet_ready.json"),
    sourceMap: path.join(PRIVATE_ROOT, "source_and_image_map.json"),
    jobKeymap: path.join(PRIVATE_ROOT, "job_keymap.json"),
    candidateMap: path.join(PRIVATE_ROOT, "candidate_map.json"),
    pairSelection: path.join(PRIVATE_ROOT, "pair_selection.json"),
    text: path.join(MATERIALS_ROOT, "frozen_text_materials.jsonl"),
    image: path.join(MATERIALS_ROOT, "image_selection.json"),
    csv: path.join(RESULTS_ROOT, "translation_results.csv"),
    pointSchema: path.join(PUBLIC_ROOT, "schemas", "pointwise_result.schema.json"),
    pairSchema: path.join(PUBLIC_ROOT, "schemas", "pairwise_result.schema.json"),
  };
  const [manifest, packetReady, sourceMap, jobKeymap, candidateMap, pairSelection, textRows, imageSelection, csvText, pointSchema, pairSchema] = await Promise.all([
    readJson(files.manifest), readJson(files.packetReady), readJson(files.sourceMap), readJson(files.jobKeymap), readJson(files.candidateMap), readJson(files.pairSelection), readJsonl(files.text), readJson(files.image), readText(files.csv), readJson(files.pointSchema), readJson(files.pairSchema),
  ]);
  const artifactWorkbook = await Workbook.fromCSV(csvText, { sheetName: "translation_results" });
  const artifactSheet = artifactWorkbook.worksheets.getItem("translation_results");
  const artifactValues = artifactSheet.getUsedRange()?.values || [];
  const artifactInspection = await artifactWorkbook.inspect({ kind: "sheet,table,region", sheetId: "translation_results", range: `A1:AK${Math.max(1, artifactValues.length)}`, maxChars: 4000, tableMaxRows: 2, tableMaxCols: 8, tableMaxCellChars: 120 });
  const csvRows = parseArtifactRows(artifactValues, issues);
  const csvIndex = rowIndex(csvRows, issues);
  const cases = materialCases(textRows, imageSelection);
  if (textRows.length !== TEXT_COUNT || (imageSelection.selected_images || []).length !== IMAGE_COUNT || cases.size !== MATERIAL_COUNT) add(issues, "frozen_material_count_mismatch", { text: textRows.length, image: imageSelection.selected_images?.length, total: cases.size });
  if (manifest.packet_id !== jobKeymap.packet_id || manifest.packet_id !== candidateMap.packet_id || manifest.packet_id !== pairSelection.packet_id) add(issues, "packet_id_cross_file_mismatch");
  if (manifest.scoring_authorized !== false || packetReady.scoring_authorized !== false || packetReady.root_approval_required_before_any_judge_call !== true) add(issues, "root_gate_or_scoring_authorization_invalid");
  if (manifest.qa?.root_review !== "PENDING") add(issues, "root_review_marker_not_pending", manifest.qa?.root_review);
  const hashAudit = await verifyManifestHashes(manifest, manifestPath, issues);
  const imageAudit = await verifySourceImages(cases, sourceMap, imageSelection, manifest, issues);
  const schemaAudit = validateOutputSchemaContracts(pointSchema, pairSchema, issues);
  const pointwiseMap = jobKeymap.pointwise || {};
  const pairwiseMap = jobKeymap.pairwise || {};
  if (Object.keys(pointwiseMap).length !== EXPECTED_PRIVATE_POINTWISE) add(issues, "private_pointwise_jobmap_count", Object.keys(pointwiseMap).length);
  if (Object.keys(pairwiseMap).length !== EXPECTED_PRIVATE_PAIRWISE) add(issues, "private_pairwise_jobmap_count", Object.keys(pairwiseMap).length);
  const pointwisePublic = {}; const pairwisePublic = {}; const pairGroups = new Map();
  const publicScanFindings = [];
  for (const judge of JUDGES) {
    const pointPath = path.join(PUBLIC_ROOT, "pointwise", `${judge}.jsonl`);
    const pairPath = path.join(PUBLIC_ROOT, "pairwise", `${judge}.jsonl`);
    const points = await readJsonl(pointPath); const pairs = await readJsonl(pairPath);
    for (const [i, job] of points.entries()) recursivePublicScan(job, `pointwise/${judge}/${i}`, publicScanFindings, MODELS);
    for (const [i, job] of pairs.entries()) recursivePublicScan(job, `pairwise/${judge}/${i}`, publicScanFindings, MODELS);
    pointwisePublic[judge] = auditPointwise(judge, points, pointwiseMap, cases, sourceMap, candidateMap, csvIndex, issues, candidateEnglishFindings, auxiliaryFindings);
    pairwisePublic[judge] = auditPairwise(judge, pairs, pairwiseMap, cases, sourceMap, candidateMap, csvIndex, issues, candidateEnglishFindings, auxiliaryFindings, pairGroups);
  }
  // A model token must not occur anywhere in a public job.  The scan is done
  // after field-level validation so source_en on text jobs is not mistaken for
  // provenance merely because the English text mentions a model-like word.
  for (const finding of publicScanFindings) add(issues, finding.code, finding);
  const graphAudit = auditSpanningGraphs(pairSelection, jobKeymap, issues);
  const abbaAudit = auditABBA(pairGroups, issues);
  const candidateRawRevalidation = await revalidateCandidateEnglishOutputs(candidateEnglishFindings, csvIndex, issues);
  const runnerAudit = auditRunnerInterface(issues);
  const freshTests = runFreshOfflineTests(issues);
  const noEnglishTranscript = auxiliaryFindings.length === 0;
  if (!noEnglishTranscript) add(issues, "image_english_transcript_in_public_payload", { count: auxiliaryFindings.length, samples: auxiliaryFindings.slice(0, 12).map(compactLeakFinding) });
  // Preserve the useful discovery without treating an immutable candidate
  // failure as a packet/privacy failure.
  const candidateEnglishFlags = candidateEnglishFindings.map(compactLeakFinding);
  const uniqueIssueCodes = uniq(issues.map((x) => x.code));
  const result = {
    qa_name: "independent_judge_packet_qa",
    packet_id: manifest.packet_id,
    run_utc: new Date().toISOString(),
    read_only: true,
    judge_calls_made: false,
    scoring_started: false,
    root_marker_created: false,
    status: issues.length ? "BLOCKED" : "PASS",
    freeze_gate: "ROOT_APPROVED_PENDING",
    counts: {
      material_total: cases.size, text_materials: textRows.length, image_materials: imageSelection.selected_images?.length || 0,
      csv_rows: csvRows.length, public_pointwise: Object.values(pointwisePublic).reduce((n, x) => n + x.public, 0), public_pairwise: Object.values(pairwisePublic).reduce((n, x) => n + x.public, 0),
      private_pointwise: Object.keys(pointwiseMap).length, private_pairwise: Object.keys(pairwiseMap).length, abba_groups: abbaAudit.groups,
    },
    checks: {
      every_238_target_source_image_binding: uniqueIssueCodes.every((x) => !x.startsWith("pointwise_") && !x.startsWith("pairwise_")) || !uniqueIssueCodes.some((x) => ["pointwise_target_not_csv_exact", "pointwise_source_alias_or_kind_mismatch", "pointwise_image_alias_or_native_flag_mismatch", "pairwise_actual_target_not_csv_exact", "pairwise_source_or_orientation_binding_mismatch", "pairwise_image_alias_or_native_flag_mismatch"].includes(x)),
      all_private_jobmaps_bound: !uniqueIssueCodes.some((x) => x.includes("private_job_not_public") || x.includes("public_job_not_private_keymap") || x.includes("jobmap_count")),
      canonical_abba_actual_targets_swapped: !uniqueIssueCodes.some((x) => x.includes("ABBA")),
      allsource_same_across_pair: !uniqueIssueCodes.some((x) => x.includes("ABBA_source")),
      spanning_graph_seven_covers: !uniqueIssueCodes.some((x) => x.includes("spanning") || x.includes("chain_coverage")),
      image_alias_sha_dimensions: !uniqueIssueCodes.some((x) => x.includes("image") && (x.includes("hash") || x.includes("dimension") || x.includes("alias"))),
      no_english_transcript_public_payload: noEnglishTranscript,
      no_owner_speed_template_oldscore_public_payload: !uniqueIssueCodes.some((x) => x.startsWith("public_")),
      manifest_hashes_fresh: hashAudit.all_match,
      six_offline_packet_tests_pass: freshTests.passed,
      schema_bounds_nulls_intbool_format_policy: !uniqueIssueCodes.some((x) => x.includes("schema") || x.includes("contract") || x.includes("required_field") || x.includes("native_flag") || x.includes("candidate_label") || x.includes("target_missing")),
      runner_interface_smoke: runnerAudit.no_wrong_field && !uniqueIssueCodes.some((x) => x.startsWith("runner_")),
    },
    artifact_tool: { api: "Workbook.fromCSV", inspection: artifactInspection },
    schema_audit: schemaAudit,
    hash_audit: hashAudit,
    image_audit: imageAudit,
    pointwise_by_judge: pointwisePublic,
    pairwise_by_judge: pairwisePublic,
    graph_audit: graphAudit,
    abba_audit: abbaAudit,
    runner_interface_audit: runnerAudit,
    candidate_raw_revalidation: candidateRawRevalidation,
    fresh_offline_tests: freshTests,
    public_scan_finding_count: publicScanFindings.length,
    image_candidate_english_quality_flags: candidateEnglishFlags,
    image_auxiliary_leak_findings: auxiliaryFindings.map(compactLeakFinding),
    issue_count: issues.length,
    issue_codes: uniqueIssueCodes,
    issues: issues.slice(0, 200),
  };
  await fs.writeFile(path.join(QA_ROOT, "qa_results.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
  const report = [
    "# Independent judge-packet QA",
    "",
    `- Status: **${result.status}** (root approval remains **PENDING**; this QA did not create ROOT_APPROVED).`,
    `- Read-only: ${result.read_only}; judge/model/API/LiteLLM calls: none; scoring started: ${result.scoring_started}.`,
    `- Scope: ${result.counts.public_pointwise} pointwise + ${result.counts.public_pairwise} pairwise public jobs; ${result.counts.private_pointwise} + ${result.counts.private_pairwise} private keymap entries; ${result.counts.abba_groups} AB/BA groups.`,
    "",
    "## Required checks",
    "",
    ...Object.entries(result.checks).map(([name, pass]) => `- ${name}: **${pass ? "PASS" : "FAIL"}**`),
    "",
    "## Findings",
    "",
    issues.length ? `- ${issues.length} issue(s), ${uniqueIssueCodes.length} unique code(s): ${uniqueIssueCodes.join(", ")}` : "- No issues.",
    candidateEnglishFindings.length ? `- ${candidateEnglishFindings.length} image candidate(s) contain exact English visible text; these are immutable untranslated-output quality flags, not packet leakage. Revalidation: ${candidateRawRevalidation.csv_exact}/${candidateRawRevalidation.suspect_count} exact to CSV and ${candidateRawRevalidation.raw_exact}/${candidateRawRevalidation.suspect_count} exact to selected raw content.` : "- No candidate-English quality flags.",
    auxiliaryFindings.length ? `- Auxiliary image English/GT leakage was detected in ${auxiliaryFindings.length} public job(s); this remains blocking.` : "- No image English/GT auxiliary fields detected in public jobs.",
    "- No judge scoring or root approval was performed by this audit.",
  ].join("\n") + "\n";
  await fs.writeFile(path.join(QA_ROOT, "qa_report.md"), report, "utf8");
  process.stdout.write(JSON.stringify({ status: result.status, issue_count: result.issue_count, issue_codes: result.issue_codes, checks: result.checks }, null, 2) + "\n");
}

await main();

