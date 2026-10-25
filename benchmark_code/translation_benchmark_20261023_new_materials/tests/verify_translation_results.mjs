#!/usr/bin/env node

/*
 * Independent, read-only QA for the new-material translation CSV.
 *
 * This deliberately runs only after the runner has written a complete
 * 7 x (30 text + 4 image) CSV and marked generation complete.  It imports
 * the CSV through the bundled ArtifactTool Workbook.fromCSV API, then joins
 * each CSV row back to its immutable raw JSONL selected record.  It never
 * starts a server, calls an endpoint, modifies raw/material/result files, or
 * computes a translation/quality score.
 */

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

// The spreadsheet skill requires the bundled runtime.  Keep this explicit so
// the harness cannot accidentally resolve a repository-local or global copy.
const { Workbook } = await import(
  process.env.CODEX_ARTIFACT_TOOL_MODULE ||
    "/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
);

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_BATCH_ROOT = path.resolve(SCRIPT_DIR, "..");
const EXPECTED_TEXT_COUNT = 30;
const EXPECTED_IMAGE_COUNT = 4;
const EXPECTED_MODEL_COUNT = 7;
const EXPECTED_ROW_COUNT = (EXPECTED_TEXT_COUNT + EXPECTED_IMAGE_COUNT) * EXPECTED_MODEL_COUNT;

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

const COMPLETED_STATUSES = new Set(["ok", "technical_ok"]);
const KNOWN_NONCOMPLETED_STATUSES = new Set([
  "N/A_not_supported", "batch_start_fail", "transport_error", "http_error", "format_fail",
  "generation_empty", "render_evidence_missing", "truncated", "finish_reason_invalid",
  "reasoning_off_violation", "template_leak", "technical_render_gate_fail", "render_gate_fail",
]);
const ALLOWED_EMPTY_RAW_STATUSES = new Set(["N/A_not_supported", "batch_start_fail"]);
const VALID_FINISH_REASONS = new Set(["stop", "length"]);
const CONTROL_TOKEN_RE = /<\|(?:end_of_turn|start_of_turn|image|image\||channel|end|start)\b|<channel\||<start_of_image>|<end_of_image>|<image>|<\/image>/;

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256Text(text) {
  return sha256Bytes(Buffer.from(text, "utf8"));
}

async function readText(filePath) {
  return fs.readFile(filePath, "utf8");
}

async function readJson(filePath) {
  return JSON.parse(await readText(filePath));
}

async function readJsonl(filePath) {
  const text = await readText(filePath);
  const records = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    try {
      records.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`invalid JSONL ${filePath}:${index + 1}: ${error.message}`);
    }
  }
  return records;
}

function asCell(value) {
  return value === null || value === undefined ? "" : String(value);
}

function asRawCell(value) {
  return value === null || value === undefined ? "" : String(value);
}

function sortedObject(value) {
  if (Array.isArray(value)) return value.map(sortedObject);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortedObject(value[key])]));
  }
  return value;
}

function stableJson(value) {
  return JSON.stringify(sortedObject(value));
}

function numericOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : String(value);
}

function metricEqual(csvValue, rawValue) {
  const csv = numericOrNull(csvValue);
  const raw = numericOrNull(rawValue);
  if (csv === null && raw === null) return true;
  // The runner's fallback fields intentionally allow a missing metric to be
  // represented by 0 in usage/timings.  Do not apply this tolerance to text,
  // hashes, status, or finish reason fields.
  if ((csv === null && raw === 0) || (raw === null && csv === 0)) return true;
  if (typeof csv === "number" && typeof raw === "number") return Object.is(csv, raw) || Math.abs(csv - raw) <= 1e-9;
  return String(csv) === String(raw);
}

function exactCellEqual(csvValue, rawValue) {
  return asCell(csvValue) === asRawCell(rawValue);
}

function addIssue(issues, code, detail = undefined) {
  issues.push(detail === undefined ? code : `${code}:${detail}`);
}

function isBlank(value) {
  return value === null || value === undefined || value === "";
}

function countOccurrences(text, needle) {
  if (typeof text !== "string" || !needle) return 0;
  return text.split(needle).length - 1;
}

function resolvePath(batchRoot, candidate) {
  if (!candidate) return null;
  return path.isAbsolute(candidate) ? candidate : path.resolve(batchRoot, candidate);
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function pngInfo(filePath) {
  const bytes = new Uint8Array(await fs.readFile(filePath));
  const signature = "89504e470d0a1a0a";
  if (bytes.length < 26 || Buffer.from(bytes.slice(0, 8)).toString("hex") !== signature) {
    return { sha256: sha256Bytes(bytes), validPng: false };
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const colorType = view.getUint8(25);
  return {
    sha256: sha256Bytes(bytes),
    validPng: true,
    width: view.getUint32(16),
    height: view.getUint32(20),
    color: colorType === 6 ? "RGBA" : colorType === 2 ? "RGB" : `png-color-type-${colorType}`,
  };
}

function expectedImageGroundTruth(image) {
  if (Array.isArray(image.ground_truth_text)) return image.ground_truth_text.join("\n");
  return image.provisional_reference_text || "";
}

function loadCaseMap(textRecords, imageSelection) {
  const cases = new Map();
  for (const record of textRecords) {
    cases.set(record.item_id, {
      caseId: record.item_id,
      inputType: "text",
      sourceId: record.source_id,
      sourceEn: record.source_en,
      sourceSha256: sha256Text(record.source_en),
      imagePath: "",
      imageSha256: "",
      imageGroundTruthEn: "",
      imageGroundTruthStatus: "",
      sourceUrl: record.source_url,
    });
  }
  for (const image of imageSelection.selected_images || []) {
    cases.set(image.image_id, {
      caseId: image.image_id,
      inputType: "image",
      sourceId: image.image_id,
      sourceEn: "",
      sourceSha256: image.sha256,
      imagePath: image.source_absolute_path,
      imageSha256: image.sha256,
      imageGroundTruthEn: expectedImageGroundTruth(image),
      imageGroundTruthStatus: image.ground_truth_status,
      sourceUrl: image.source_pdf_url || "",
      layoutStratum: image.layout_stratum,
      dimensions: image.dimensions,
      imageRecord: image,
    });
  }
  return cases;
}

function getRawSelected(rawRecords, modelId, caseId) {
  const matches = [];
  for (const wrapper of rawRecords) {
    const selected = wrapper && wrapper.selected;
    if (selected && selected.case_id === caseId) matches.push({ wrapper, selected });
    else if (!selected && wrapper && wrapper.case_id === caseId) matches.push({ wrapper, selected: null });
  }
  // There should be exactly one wrapper per case.  Returning the last selected
  // preserves the runner's selected retry record while the duplicate count is
  // reported by the caller.
  const selectedMatches = matches.filter((item) => item.selected && item.selected.model_id === modelId);
  const modelMatches = selectedMatches.length ? selectedMatches : matches;
  return { matches: modelMatches, item: modelMatches.length ? modelMatches[modelMatches.length - 1] : null };
}

function expectedPromptText(config, model, caseRecord) {
  if (caseRecord.inputType !== "text") return null;
  if (model.template_mode === "custom_translategemma_adapter") return caseRecord.sourceEn;
  return `${config.shared_instruction}\n\n${caseRecord.sourceEn}`;
}

function inspectRequest(selected, config, model, caseRecord, issues) {
  const request = selected?.request_redacted;
  if (!request || typeof request !== "object") {
    addIssue(issues, "request_redacted_missing");
    return;
  }
  const serialized = JSON.stringify(request);
  if (/data:image\/[\w.+-]+;base64,/i.test(serialized)) addIssue(issues, "redacted_request_contains_base64_data_url");
  if (/data:[^\s"']+;base64,/i.test(serialized)) addIssue(issues, "redacted_request_contains_base64_payload");
  if (caseRecord.inputType === "image" && caseRecord.imageGroundTruthEn && serialized.includes(caseRecord.imageGroundTruthEn)) {
    addIssue(issues, "image_ground_truth_in_request_redacted");
  }
  if (caseRecord.inputType === "text") {
    const messages = request.messages;
    const first = Array.isArray(messages) ? messages[0] : null;
    if (!first || first.role !== "user" || typeof first.content !== "string") {
      addIssue(issues, "text_request_shape_not_one_user_string");
    } else {
      const expected = expectedPromptText(config, model, caseRecord);
      if (first.content !== expected) addIssue(issues, "text_request_source_not_exact");
      if (model.template_mode === "custom_translategemma_adapter" && first.content.includes(config.shared_instruction)) {
        addIssue(issues, "translategemma_text_request_contains_shared_instruction");
      }
    }
  } else {
    const messages = request.messages;
    const first = Array.isArray(messages) ? messages[0] : null;
    const content = first && first.role === "user" ? first.content : null;
    if (!Array.isArray(messages) || messages.length !== 1 || !first || first.role !== "user" || !Array.isArray(content)) {
      addIssue(issues, "image_request_shape_not_one_user_parts");
      return;
    }
    const textParts = content.filter((part) => part && part.type === "text");
    const imageParts = content.filter((part) => part && part.type === "image_url");
    if (textParts.length !== 1 || imageParts.length !== 1) addIssue(issues, "image_request_not_one_text_plus_one_image");
    if (textParts.length === 1 && textParts[0].text !== config.shared_instruction) addIssue(issues, "image_instruction_not_exact");
    const imageUrl = imageParts[0]?.image_url?.url;
    if (typeof imageUrl !== "string" || imageUrl !== "<redacted-local-image-data-url>") addIssue(issues, "image_data_url_not_redacted");
    const redactedImageSha = imageParts[0]?.image_url?.data_url_sha256;
    // The runner intentionally hashes the original data-URL string in this
    // redacted field, not the PNG bytes.  The authoritative PNG SHA-256 is
    // checked independently against the frozen image metadata above.
    if (typeof redactedImageSha !== "string" || !/^[0-9a-f]{64}$/i.test(redactedImageSha)) {
      addIssue(issues, "redacted_image_data_url_hash_missing_or_invalid");
    }
  }
}

function nativeImagePair(renderChecks, templateMode) {
  const evidence = renderChecks?.rendered_prompt_image_marker_evidence?.pairs || {};
  const pairName = templateMode === "custom_translategemma_adapter" ? "translategemma_start_end" : "gemma4_pipe_pair";
  const pair = evidence[pairName];
  if (!pair || pair.start_count !== 1 || pair.end_count !== 1 || pair.ordered !== true) return false;
  const legacy = evidence.legacy_pipe_pair;
  if (legacy && (legacy.start_count !== 0 || legacy.end_count !== 0)) return false;
  return true;
}

function renderGate(selected, config, model, caseRecord, issues) {
  if (!selected) return { pass: null, state: "not_observed", failures: [] };
  const checks = selected.render_checks && typeof selected.render_checks === "object" ? selected.render_checks : {};
  const failures = [];
  const requireTrue = (condition, code) => { if (condition !== true) failures.push(code); };
  requireTrue(checks.rendered_prompt_present, "rendered_prompt_missing");
  requireTrue(checks.rendered_prompt_exactly_one_user_one_model, "turn_trace_not_one_user_one_model");
  requireTrue(checks.rendered_prompt_generation_marker_present, "generation_marker_missing");
  requireTrue(checks.rendered_prompt_source_or_marker_present, "source_or_image_marker_missing");
  requireTrue(checks.reasoning_off_violation === false, "reasoning_off_violation");
  requireTrue(checks.request_control_token_leak === false, "request_control_token_leak");
  requireTrue(checks.content_control_token_leak === false, "content_control_token_leak");
  requireTrue(checks.reasoning_control_token_leak === false, "reasoning_control_token_leak");
  requireTrue(checks.finish_reason_valid === true, "finish_reason_invalid");
  requireTrue(checks.native_template_owner === model.template_mode, "native_template_owner_mismatch");
  if (caseRecord.inputType === "text") {
    requireTrue(checks.rendered_prompt_image_marker_present === false, "unexpected_image_marker_in_text");
    if (checks.rendered_prompt_source_occurrences !== 1) failures.push(`text_source_occurrences=${checks.rendered_prompt_source_occurrences}`);
    if (model.template_mode === "custom_translategemma_adapter") requireTrue(checks.request_shape?.tg_source_only_request, "tg_text_not_source_only");
  } else {
    requireTrue(checks.rendered_prompt_image_marker_present, "native_image_marker_missing");
    if (!nativeImagePair(checks, model.template_mode)) failures.push("native_image_marker_pair_not_exactly_one");
    requireTrue(checks.request_shape?.image_request_messages_shape_valid, "image_request_shape_invalid");
  }
  if (Array.isArray(checks.render_gate_failures) && checks.render_gate_failures.length) failures.push(...checks.render_gate_failures.map((x) => `runner:${x}`));
  if (selected.status === "ok" || selected.status === "technical_ok") {
    requireTrue(checks.render_gate_pass, "runner_render_gate_not_pass");
  }
  // Output control-token checks are independent of the runner's checker; the
  // regex intentionally treats image markers as a leak only in model output,
  // not in the rendered prompt itself.
  if (CONTROL_TOKEN_RE.test(asRawCell(selected.content))) failures.push("content_contains_control_token");
  if (CONTROL_TOKEN_RE.test(asRawCell(selected.reasoning_content))) failures.push("reasoning_contains_control_token");
  return { pass: failures.length === 0, state: "observed", failures };
}

function metricExpectations(selected) {
  const usage = selected?.usage && typeof selected.usage === "object" ? selected.usage : {};
  const timings = selected?.timings && typeof selected.timings === "object" ? selected.timings : {};
  const cacheDetails = usage.prompt_tokens_details && typeof usage.prompt_tokens_details === "object" ? usage.prompt_tokens_details : {};
  return {
    elapsed_seconds: selected?.elapsed_seconds,
    prompt_tokens: usage.prompt_tokens ?? timings.prompt_n,
    completion_tokens: usage.completion_tokens ?? timings.predicted_n,
    total_tokens: usage.total_tokens,
    prompt_tok_s: timings.prompt_per_second,
    generation_tok_s: timings.predicted_per_second,
    cache_tokens: cacheDetails.cached_tokens ?? timings.cache_n,
  };
}

function compareRawRecord(row, selected, wrapper, config, model, caseRecord, issues) {
  if (!selected) return;
  const expectedMetrics = metricExpectations(selected);
  const metricFields = ["elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_tok_s", "generation_tok_s", "cache_tokens"];
  for (const field of metricFields) {
    if (!metricEqual(row[field], expectedMetrics[field])) addIssue(issues, `raw_${field}_mismatch`);
  }
  const exactFields = {
    translation_zh_tw: selected.content,
    reasoning_content: selected.reasoning_content,
    http_status: selected.http_envelope?.http_status,
    finish_reason: selected.finish_reason,
    status: selected.status,
    request_sha256: selected.request_sha256,
    response_body_sha256: selected.response_body_sha256,
    config_sha256: selected.config_sha256,
    rendered_prompt_sha256: selected.render_checks?.rendered_prompt_sha256,
  };
  for (const [field, value] of Object.entries(exactFields)) {
    if (!exactCellEqual(row[field], value)) addIssue(issues, `raw_${field}_mismatch`);
  }
  const expectedRetry = wrapper?.retry_count ?? 0;
  if (!metricEqual(row.retry_count, expectedRetry)) addIssue(issues, "raw_retry_count_mismatch");
  if (!exactCellEqual(row.model_sha256, model.inventory.model_sha256)) addIssue(issues, "model_sha256_not_inventory");
  if (!exactCellEqual(row.mmproj_sha256, model.inventory.mmproj_sha256)) addIssue(issues, "mmproj_sha256_not_inventory");
  if (!exactCellEqual(row.template_mode, model.template_mode)) addIssue(issues, "template_mode_not_config");
  if (!exactCellEqual(row.embedded_template_sha256, model.inventory.embedded_template_sha256)) addIssue(issues, "embedded_template_sha256_not_inventory");
  if (!exactCellEqual(row.adapter_template_sha256, model.inventory.adapter_template_sha256)) addIssue(issues, "adapter_template_sha256_not_inventory");
  if (selected.model_id && selected.model_id !== row.model_id) addIssue(issues, "raw_model_id_mismatch");
  if (selected.case_id && selected.case_id !== row.case_id) addIssue(issues, "raw_case_id_mismatch");
  if (selected.input_type && selected.input_type !== row.input_type) addIssue(issues, "raw_input_type_mismatch");
  if (selected.source_sha256 && selected.source_sha256 !== caseRecord.sourceSha256) addIssue(issues, "raw_source_sha256_not_frozen");
  if (caseRecord.inputType === "image" && selected.image_sha256 && selected.image_sha256 !== caseRecord.imageSha256) addIssue(issues, "raw_image_sha256_not_frozen");
  if (selected.image_ground_truth_en && selected.image_ground_truth_en !== caseRecord.imageGroundTruthEn) addIssue(issues, "raw_image_ground_truth_not_frozen");
  inspectRequest(selected, config, model, caseRecord, issues);
}

async function loadRawIndex(rawPath, cache) {
  const key = String(rawPath);
  if (cache.has(key)) return cache.get(key);
  const records = await readJsonl(rawPath);
  cache.set(key, records);
  return records;
}

function parseRowsFromArtifact(csvValues, issues) {
  if (!Array.isArray(csvValues) || csvValues.length < 1) {
    addIssue(issues, "artifact_tool_empty_csv");
    return [];
  }
  const header = csvValues[0].map(asCell);
  if (JSON.stringify(header) !== JSON.stringify(CSV_FIELDS)) {
    addIssue(issues, "csv_header_mismatch");
  }
  const rows = [];
  for (const values of csvValues.slice(1)) {
    const row = {};
    for (let i = 0; i < header.length; i += 1) row[header[i]] = asCell(values?.[i]);
    rows.push(row);
  }
  return rows;
}

async function verifySmokeRetention(batchRoot, config, issues) {
  const smokeRoot = path.join(batchRoot, "technical_smoke");
  const perModel = [];
  for (const model of config.models) {
    const v1Path = path.join(smokeRoot, `${model.id}.summary.json`);
    const v2Path = path.join(smokeRoot, `${model.id}.summary.v2.json`);
    const rawPath = path.join(smokeRoot, `${model.id}.jsonl`);
    const v1 = await exists(v1Path);
    const v2 = await exists(v2Path);
    const raw = await exists(rawPath);
    if (!v1) addIssue(issues, "smoke_v1_missing", model.id);
    if (!v2) addIssue(issues, "smoke_v2_missing", model.id);
    if (!raw) addIssue(issues, "smoke_raw_missing", model.id);
    const item = { model_id: model.id, v1_path: v1Path, v2_path: v2Path, raw_path: rawPath, v1, v2, raw };
    if (v1) {
      try {
        const record = await readJson(v1Path);
        item.v1_sha256 = await sha256File(v1Path);
        item.v1_summary = {
          model_id: record.model_id,
          text_status: record.text_status,
          image_status: record.image_status,
          text_native_supported: record.text_native_supported,
          image_native_supported: record.image_native_supported,
        };
      } catch { addIssue(issues, "smoke_v1_invalid_json", model.id); }
    }
    if (v2) {
      try {
        const record = await readJson(v2Path);
        item.v2_sha256 = await sha256File(v2Path);
        if (record.derivation_only !== true) addIssue(issues, "smoke_v2_not_marked_derivation_only", model.id);
        item.v2_summary = {
          model_id: record.model_id,
          derivation_only: record.derivation_only,
          source_raw_jsonl: record.source_raw_jsonl,
          source_raw_jsonl_sha256: record.source_raw_jsonl_sha256,
          validator_sha256: record.validator_sha256,
        };
        if (record.source_raw_jsonl && raw) {
          const actual = await sha256File(rawPath);
          if (record.source_raw_jsonl_sha256 && record.source_raw_jsonl_sha256 !== actual) addIssue(issues, "smoke_v2_raw_hash_mismatch", model.id);
        }
      } catch { addIssue(issues, "smoke_v2_invalid_json", model.id); }
    }
    perModel.push(item);
  }
  const topV1 = path.join(smokeRoot, "summary.json");
  const topV2 = path.join(smokeRoot, "summary.v2.json");
  const top = { v1: await exists(topV1), v2: await exists(topV2), v1_path: topV1, v2_path: topV2 };
  if (!top.v1) addIssue(issues, "smoke_top_v1_missing");
  if (!top.v2) addIssue(issues, "smoke_top_v2_missing");
  return { per_model: perModel, top, all_model_v1_v2_raw_present: perModel.every((x) => x.v1 && x.v2 && x.raw) && top.v1 && top.v2 };
}

async function verifyImageArtifacts(imageSelection, issues) {
  const results = [];
  for (const image of imageSelection.selected_images || []) {
    const result = {
      image_id: image.image_id,
      path: image.source_absolute_path,
      declared_sha256: image.sha256,
      declared_dimensions: image.dimensions || null,
    };
    try {
      const actual = await pngInfo(image.source_absolute_path);
      Object.assign(result, { actual_sha256: actual.sha256, actual_dimensions: { width: actual.width, height: actual.height, color: actual.color }, valid_png: actual.validPng });
      if (!actual.validPng) addIssue(issues, "selected_image_not_valid_png", image.image_id);
      if (actual.sha256 !== image.sha256) addIssue(issues, "selected_image_sha256_mismatch", image.image_id);
      if (image.dimensions && (actual.width !== image.dimensions.width || actual.height !== image.dimensions.height || actual.color !== image.dimensions.color)) addIssue(issues, "selected_image_dimensions_mismatch", image.image_id);
    } catch (error) {
      addIssue(issues, "selected_image_unreadable", image.image_id);
      result.error = String(error.message || error);
    }
    results.push(result);
  }
  return results;
}

async function verifySourceArtifacts(batchRoot, manifestPath, textPath, imageSelectionPath, configPath, manifest, issues) {
  const hashes = {
    metadata_manifest_sha256: await sha256File(manifestPath),
    frozen_text_sha256: await sha256File(textPath),
    image_selection_sha256: await sha256File(imageSelectionPath),
    config_sha256: await sha256File(configPath),
  };
  if (manifest.frozen_artifacts?.frozen_text_sha256 && hashes.frozen_text_sha256 !== manifest.frozen_artifacts.frozen_text_sha256) addIssue(issues, "frozen_text_file_hash_mismatch");
  const declaredArtifactHashes = manifest.authoritative_source_bundle?.artifact_sha256 || {};
  const sourceBundle = manifest.authoritative_source_bundle?.path;
  const sourceArtifacts = [];
  if (sourceBundle && declaredArtifactHashes) {
    for (const [name, declared] of Object.entries(declaredArtifactHashes)) {
      const filePath = path.join(sourceBundle, name);
      const item = { name, path: filePath, declared_sha256: declared };
      try {
        item.actual_sha256 = await sha256File(filePath);
        item.matches = item.actual_sha256 === declared;
        if (!item.matches) addIssue(issues, "authoritative_artifact_hash_mismatch", name);
      } catch (error) {
        item.matches = false;
        item.error = String(error.message || error);
        addIssue(issues, "authoritative_artifact_unreadable", name);
      }
      sourceArtifacts.push(item);
    }
  }
  return { local: hashes, authoritative: sourceArtifacts };
}

async function sha256File(filePath) {
  return sha256Bytes(await fs.readFile(filePath));
}

async function verifyBatch(options) {
  const batchRoot = path.resolve(options.batchRoot);
  const resultsRoot = path.join(batchRoot, "results");
  const materialsRoot = path.join(batchRoot, "materials");
  const configPath = path.join(batchRoot, "configs", "translation_config.json");
  const manifestPath = path.join(materialsRoot, "metadata_manifest.json");
  const textPath = path.join(materialsRoot, "frozen_text_materials.jsonl");
  const imageSelectionPath = path.join(materialsRoot, "image_selection.json");
  const csvPath = path.join(resultsRoot, "translation_results.csv");
  const summaryPath = path.join(resultsRoot, "translation_run_summary.json");
  const progressPath = path.join(batchRoot, "progress.md");
  const outputPath = path.resolve(options.outputPath || path.join(batchRoot, "independent_qa", "qa_results.json"));
  const issues = [];
  const readiness = { csv_exists: await exists(csvPath), summary_exists: await exists(summaryPath), progress_exists: await exists(progressPath) };
  let summary = null;
  let progress = "";
  if (readiness.summary_exists) summary = await readJson(summaryPath);
  if (readiness.progress_exists) progress = await readText(progressPath);
  readiness.progress_generation_complete = /Status:\s*\*\*GENERATION_COMPLETE\*\*/.test(progress);
  readiness.summary_generation_complete = Boolean(summary && summary.actual_rows === EXPECTED_ROW_COUNT && summary.planned_rows === EXPECTED_ROW_COUNT);
  if (!readiness.csv_exists) readiness.missing = "results/translation_results.csv";
  if (!readiness.summary_exists) readiness.missing = "results/translation_run_summary.json";
  if (!readiness.progress_exists) readiness.missing = "progress.md";
  if (!readiness.progress_generation_complete) readiness.missing = "progress.md GENERATION_COMPLETE";
  if (!readiness.summary_generation_complete) readiness.missing = "translation_run_summary actual_rows/planned_rows 238";
  if (!readiness.csv_exists || !readiness.summary_exists || !readiness.progress_exists || !readiness.progress_generation_complete || !readiness.summary_generation_complete) {
    return { pending: true, batch_root: batchRoot, output_path: outputPath, readiness };
  }

  const [config, manifest, imageSelection, textJsonl, csvText] = await Promise.all([
    readJson(configPath), readJson(manifestPath), readJson(imageSelectionPath), readJsonl(textPath), readText(csvPath),
  ]);
  if (manifest.freeze_ready !== true) addIssue(issues, "materials_manifest_not_freeze_ready");
  if (textJsonl.length !== EXPECTED_TEXT_COUNT) addIssue(issues, "frozen_text_count_mismatch", textJsonl.length);
  if ((imageSelection.selected_images || []).length !== EXPECTED_IMAGE_COUNT) addIssue(issues, "selected_image_count_mismatch", imageSelection.selected_images?.length);
  if ((config.models || []).length !== EXPECTED_MODEL_COUNT) addIssue(issues, "configured_model_count_mismatch", config.models?.length);
  const caseMap = loadCaseMap(textJsonl, imageSelection);
  if (caseMap.size !== EXPECTED_TEXT_COUNT + EXPECTED_IMAGE_COUNT) addIssue(issues, "frozen_case_count_mismatch", caseMap.size);
  const inventory = await readJson(path.join(batchRoot, "raw", "model_inventory.json"));
  const modelById = new Map();
  for (const model of config.models || []) {
    const inv = inventory.models?.[model.id];
    if (!inv) addIssue(issues, "model_missing_inventory", model.id);
    modelById.set(model.id, { ...model, inventory: inv || {} });
  }
  const configSha256 = sha256Text(stableJson(config));
  const artifactWorkbook = await Workbook.fromCSV(csvText, { sheetName: "translation_results" });
  const artifactSheet = artifactWorkbook.worksheets.getItem("translation_results");
  const artifactUsedRange = artifactSheet.getUsedRange();
  const artifactValues = artifactUsedRange?.values || [];
  const artifactInspection = await artifactWorkbook.inspect({ kind: "sheet,table,region", sheetId: "translation_results", range: `A1:AK${Math.max(1, artifactValues.length)}`, maxChars: 4000, tableMaxRows: 3, tableMaxCols: 8, tableMaxCellChars: 120 });
  const rows = parseRowsFromArtifact(artifactValues, issues);
  if (rows.length !== EXPECTED_ROW_COUNT) addIssue(issues, "csv_row_count_mismatch", rows.length);
  const expectedKeys = new Set();
  for (const model of config.models || []) for (const caseRecord of caseMap.values()) expectedKeys.add(`${model.id}::${caseRecord.caseId}`);
  const rowKeys = new Map();
  for (const row of rows) {
    const key = `${row.model_id}::${row.case_id}`;
    rowKeys.set(key, (rowKeys.get(key) || 0) + 1);
  }
  for (const key of expectedKeys) if (!rowKeys.has(key)) addIssue(issues, "missing_model_case_cell", key);
  for (const [key, count] of rowKeys.entries()) {
    if (!expectedKeys.has(key)) addIssue(issues, "unexpected_model_case_cell", key);
    if (count !== 1) addIssue(issues, "duplicate_model_case_cell", `${key}=${count}`);
  }
  const rawCache = new Map();
  const rowResults = [];
  const rawPathCounts = new Map();
  let failureCount = 0;
  let renderPassCount = 0;
  let renderObservedCount = 0;
  for (const row of rows) {
    const rowIssues = [];
    const caseRecord = caseMap.get(row.case_id);
    const model = modelById.get(row.model_id);
    if (!caseRecord) addIssue(rowIssues, "case_not_in_frozen_materials");
    if (!model) addIssue(rowIssues, "model_not_in_config");
    if (caseRecord) {
      if (row.input_type !== caseRecord.inputType) addIssue(rowIssues, "input_type_not_frozen");
      if (row.source_id !== caseRecord.sourceId) addIssue(rowIssues, "source_id_not_frozen");
      if (row.source_en !== caseRecord.sourceEn) addIssue(rowIssues, "source_en_not_exact");
      if (row.source_sha256 !== caseRecord.sourceSha256) addIssue(rowIssues, "source_sha256_not_frozen");
      if (row.image_path !== caseRecord.imagePath) addIssue(rowIssues, "image_path_not_frozen");
      if (row.image_sha256 !== caseRecord.imageSha256) addIssue(rowIssues, "image_sha256_not_frozen");
      if (row.image_ground_truth_en !== caseRecord.imageGroundTruthEn) addIssue(rowIssues, "image_ground_truth_not_exact");
      if (row.image_ground_truth_status !== caseRecord.imageGroundTruthStatus) addIssue(rowIssues, "image_ground_truth_status_not_exact");
    }
    if (model) {
      if (row.config_sha256 !== configSha256) addIssue(rowIssues, "config_sha256_not_config");
      if (row.model_sha256 !== model.inventory.model_sha256) addIssue(rowIssues, "model_sha256_not_inventory");
      if (row.mmproj_sha256 !== model.inventory.mmproj_sha256) addIssue(rowIssues, "mmproj_sha256_not_inventory");
      if (row.template_mode !== model.template_mode) addIssue(rowIssues, "template_mode_not_config");
      if (row.embedded_template_sha256 !== asCell(model.inventory.embedded_template_sha256)) addIssue(rowIssues, "embedded_template_sha256_not_inventory");
      if (row.adapter_template_sha256 !== asCell(model.inventory.adapter_template_sha256)) addIssue(rowIssues, "adapter_template_sha256_not_inventory");
    }
    if (!KNOWN_NONCOMPLETED_STATUSES.has(row.status) && !COMPLETED_STATUSES.has(row.status)) addIssue(rowIssues, "unknown_status_preserved_for_review", row.status || "blank");
    if (!COMPLETED_STATUSES.has(row.status)) failureCount += 1;
    let selected = null;
    let wrapper = null;
    let rawState = "not_referenced";
    if (row.case_record_path) {
      const rawPath = resolvePath(batchRoot, row.case_record_path);
      rawPathCounts.set(rawPath, (rawPathCounts.get(rawPath) || 0) + 1);
      if (!(await exists(rawPath))) {
        addIssue(rowIssues, "case_record_path_missing");
        rawState = "missing";
      } else {
        try {
          const rawRecords = await loadRawIndex(rawPath, rawCache);
          const found = getRawSelected(rawRecords, row.model_id, row.case_id);
          if (found.matches.length !== 1) addIssue(rowIssues, "raw_case_record_match_count", found.matches.length);
          wrapper = found.item?.wrapper || null;
          selected = found.item?.selected || null;
          rawState = selected ? "selected" : "wrapper_failure";
          if (selected && model && caseRecord) compareRawRecord(row, selected, wrapper, config, model, caseRecord, rowIssues);
          if (!selected && !ALLOWED_EMPTY_RAW_STATUSES.has(row.status)) addIssue(rowIssues, "raw_selected_missing_for_status");
        } catch (error) {
          addIssue(rowIssues, "case_record_jsonl_invalid", String(error.message || error));
          rawState = "invalid";
        }
      }
    } else if (!ALLOWED_EMPTY_RAW_STATUSES.has(row.status)) {
      addIssue(rowIssues, "case_record_path_missing_for_attempted_case");
      rawState = "missing_for_attempted_case";
    }
    if (selected && model && caseRecord) {
      const gate = renderGate(selected, config, model, caseRecord, rowIssues);
      if (gate.pass !== null) renderObservedCount += 1;
      if (gate.pass === true) renderPassCount += 1;
      rowResults.push({ model_id: row.model_id, case_id: row.case_id, status: row.status, raw_state: rawState, render_gate_pass: gate.pass, render_gate_state: gate.state, render_gate_failures: gate.failures, issue_count: rowIssues.length, issues: rowIssues });
    } else {
      rowResults.push({ model_id: row.model_id, case_id: row.case_id, status: row.status, raw_state: rawState, render_gate_pass: null, render_gate_state: "not_observed", render_gate_failures: [], issue_count: rowIssues.length, issues: rowIssues });
    }
  }
  const smokeRetention = await verifySmokeRetention(batchRoot, config, issues);
  const imageArtifacts = await verifyImageArtifacts(imageSelection, issues);
  const sourceArtifacts = await verifySourceArtifacts(batchRoot, manifestPath, textPath, imageSelectionPath, configPath, manifest, issues);
  const nonEmptyRowIssueCount = rowResults.filter((x) => x.issue_count > 0).length;
  const allExpectedCells = rows.length === EXPECTED_ROW_COUNT && expectedKeys.size === EXPECTED_ROW_COUNT && [...rowKeys.entries()].every(([key, count]) => expectedKeys.has(key) && count === 1);
  const allRenderedObservedPass = rowResults.filter((x) => x.render_gate_state === "observed").every((x) => x.render_gate_pass === true);
  const qaStatus = issues.length === 0 && allExpectedCells && nonEmptyRowIssueCount === 0
    ? (failureCount === 0 ? "PASS_ALL_238" : "PASS_238_WITH_RECORDED_FAILURES")
    : "FAIL";
  const output = {
    schema_version: "independent-translation-csv-qa-v1",
    qa_status: qaStatus,
    read_only: true,
    quality_scoring_run: false,
    batch_root: batchRoot,
    checked_at: new Date().toISOString(),
    readiness,
    artifact_tool: {
      library: "@oai/artifact-tool",
      api: "Workbook.fromCSV",
      csv_imported_as_strings: true,
      used_range_rows: artifactValues.length,
      used_range_columns: artifactValues[0]?.length || 0,
      inspection: artifactInspection.ndjson,
    },
    expected: { models: EXPECTED_MODEL_COUNT, text_cases: EXPECTED_TEXT_COUNT, image_cases: EXPECTED_IMAGE_COUNT, cases_per_model: EXPECTED_TEXT_COUNT + EXPECTED_IMAGE_COUNT, rows: EXPECTED_ROW_COUNT },
    observed: { csv_rows: rows.length, unique_model_case_cells: rowKeys.size, failure_rows_preserved: failureCount, render_gate_observed: renderObservedCount, render_gate_pass: renderPassCount, row_issue_count: nonEmptyRowIssueCount },
    source_and_freeze: { manifest_freeze_ready: manifest.freeze_ready === true, config_sha256: configSha256, source_artifacts: sourceArtifacts, image_artifacts: imageArtifacts },
    smoke_retention: smokeRetention,
    checks: {
      exact_model_case_matrix: allExpectedCells,
      frozen_source_and_image_hashes: rows.length === EXPECTED_ROW_COUNT && nonEmptyRowIssueCount === 0,
      raw_translation_content_exact: rowResults.every((x) => !x.issues.some((issue) => issue.startsWith("raw_translation_zh_tw_mismatch"))),
      raw_finish_status_usage_timings_exact_or_zero_missing_compatible: rowResults.every((x) => !x.issues.some((issue) => issue.startsWith("raw_") && !issue.includes("translation_zh_tw"))),
      request_gt_redaction: rowResults.every((x) => !x.issues.some((issue) => issue.includes("ground_truth_in_request") || issue.includes("base64"))),
      hash_provenance: rowResults.every((x) => !x.issues.some((issue) => issue.includes("sha256_not_inventory") || issue.includes("config_sha256_not_config"))),
      smoke_v1_v2_retained: smokeRetention.all_model_v1_v2_raw_present,
      render_gate_all_observed_pass: allRenderedObservedPass,
      no_ocr_accuracy_claim: true,
    },
    status_counts: Object.fromEntries([...new Set(rows.map((row) => row.status))].sort().map((status) => [status, rows.filter((row) => row.status === status).length])),
    row_results: rowResults,
    issues,
    notes: [
      "This QA checks transport/record provenance and render evidence only; it does not judge translation quality or OCR CER/exactness.",
      "Unsupported, truncated, or other failures remain their recorded statuses and are never converted to zero or PASS.",
      "The image ground truth is checked for exact CSV/material identity and absence from redacted requests; img-09/img-10 remain machine-extracted/AI-visual-checked, not human gold.",
      "request_redacted.image_url.data_url_sha256 is the runner's hash of the original data-URL string; frozen PNG identity is checked separately from image_sha256 and the selected PNG artifact.",
    ],
  };
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(output, null, 2) + "\n", "utf8");
  return { pending: false, outputPath, output };
}

function parseArgs(argv) {
  const options = { batchRoot: DEFAULT_BATCH_ROOT, outputPath: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--batch-root") options.batchRoot = argv[++i];
    else if (argv[i] === "--output") options.outputPath = argv[++i];
    else if (argv[i] === "--help") {
      console.log("Usage: verify_translation_results.mjs [--batch-root PATH] [--output PATH]");
      process.exit(0);
    } else throw new Error(`unknown argument: ${argv[i]}`);
  }
  return options;
}

try {
  const result = await verifyBatch(parseArgs(process.argv.slice(2)));
  if (result.pending) {
    console.error(JSON.stringify({ qa_status: "PENDING", ...result.readiness }, null, 2));
    process.exitCode = 2;
  } else {
    console.log(JSON.stringify({ qa_status: result.output.qa_status, output_path: result.outputPath, observed: result.output.observed }, null, 2));
    process.exitCode = result.output.qa_status === "FAIL" ? 1 : 0;
  }
} catch (error) {
  console.error(error?.stack || error);
  process.exitCode = 1;
}
