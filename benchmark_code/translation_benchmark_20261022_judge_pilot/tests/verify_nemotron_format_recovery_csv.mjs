import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

const [outputDir, packetPath] = process.argv.slice(2);
assert(outputDir && packetPath, 'Provide recovery output directory and frozen packet');
const packet = JSON.parse(await fs.readFile(packetPath, 'utf8'));
const sourceByItem = new Map();
const targetByKey = new Map();
function visit(value) {
  if (Array.isArray(value)) return value.forEach(visit);
  if (!value || typeof value !== 'object') return;
  const itemId = value.item_id ?? value.qualification_id;
  const source = value.source_en ?? value.source_text ?? value.source;
  if (typeof itemId === 'string' && typeof source === 'string') sourceByItem.set(itemId, source);
  const candidates = value.candidates ?? value.variants;
  if (typeof itemId === 'string' && Array.isArray(candidates)) {
    for (const candidate of candidates) {
      if (!candidate || typeof candidate !== 'object') continue;
      const candidateId = candidate.blind_id ?? candidate.candidate_id;
      const target = candidate.target_zh_tw ?? candidate.target ?? candidate.translation;
      if (typeof candidateId === 'string' && typeof target === 'string') targetByKey.set(`${itemId}/${candidateId}`, target);
    }
  }
  Object.values(value).forEach(visit);
}
visit(packet);
const sourceHash = source => createHash('sha256').update(source).digest('hex');

async function readCsv(filename, sheetName) {
  const workbook = await Workbook.fromCSV(await fs.readFile(`${outputDir}/${filename}`, 'utf8'), { sheetName });
  const values = workbook.worksheets.getItem(sheetName).getUsedRange().values;
  assert(values.length >= 1, `${filename} missing header`);
  const headers = values[0].map(String);
  return { filename, headers, rows: values.slice(1).filter(row => row.some(value => value !== null && value !== '')), col: Object.fromEntries(headers.map((name, index) => [name, index])) };
}

const panel = `${outputDir}/panel_recovered`;
const comparison = await readCsv('panel_recovered/translation_comparison.fivejudges.owner.csv', 'Comparison');
assert.equal(comparison.rows.length, 21);
assert.equal(comparison.headers.filter(name => name.endsWith('_score_0_100')).length, 5);
for (const row of comparison.rows) {
  const key = `${row[comparison.col.item_id]}/${row[comparison.col.candidate_id]}`;
  assert(targetByKey.has(key), `unknown comparison key ${key}`);
  const source = sourceByItem.get(String(row[comparison.col.item_id]));
  assert.equal(row[comparison.col.source_en], source, `source ${key}`);
  assert.equal(row[comparison.col.target_zh_tw], targetByKey.get(key), `target ${key}`);
  assert.equal(row[comparison.col.source_text_sha256], sourceHash(source), `source hash ${key}`);
}
const long = await readCsv('panel_recovered/pointwise_results_long.fivejudges.owner.csv', 'Scores');
assert.equal(long.rows.length, 105);
assert.equal(new Set(long.rows.map(row => row[long.col.judge_id])).size, 5);
const expectedErrorCounts = new Map();
for (const row of long.rows) expectedErrorCounts.set(`${row[long.col.judge_id]}/${row[long.col.item_id]}/${row[long.col.candidate_id]}`, Number(row[long.col.error_count]));
const wide = await readCsv('panel_recovered/pointwise_results_wide.fivejudges.owner.csv', 'Scores');
assert.equal(wide.rows.length, 21);
assert.equal(wide.headers.filter(name => name.endsWith('_score_0_100')).length, 5);
const validity = await readCsv('panel_recovered/judge_validity.fivejudges.owner.csv', 'Validity');
assert.equal(validity.rows.length, 5);
for (const row of validity.rows) {
  assert.equal(row[validity.col.validation_status], 'PASS');
  assert.equal(Number(row[validity.col.pointwise_rows]), 21);
  assert.equal(Number(row[validity.col.qualification_variants]), 24);
  assert.equal(Number(row[validity.col.pairwise_rows]), 6);
}
const errors = await readCsv('panel_recovered/error_annotations.fivejudges.owner.csv', 'Errors');
assert.equal(errors.rows.length, 134, 'recovery adds two raw error annotations to original on panel');
const seenErrors = new Set();
for (const row of errors.rows) {
  const key = `${row[errors.col.judge_id]}/${row[errors.col.item_id]}/${row[errors.col.candidate_id]}/${row[errors.col.error_index]}`;
  assert(!seenErrors.has(key), `duplicate error ${key}`);
  seenErrors.add(key);
  const itemKey = `${row[errors.col.item_id]}/${row[errors.col.candidate_id]}`;
  const source = sourceByItem.get(String(row[errors.col.item_id]));
  assert(targetByKey.has(itemKey), `unknown error key ${itemKey}`);
  assert.equal(row[errors.col.source_text_sha256], sourceHash(source));
  const quote = row[errors.col.target_exact_quote] ?? '';
  assert(quote === '' || targetByKey.get(itemKey).includes(quote), `nonliteral quote ${key}`);
}
const observedErrorCounts = new Map();
for (const row of errors.rows) {
  const key = `${row[errors.col.judge_id]}/${row[errors.col.item_id]}/${row[errors.col.candidate_id]}`;
  observedErrorCounts.set(key, (observedErrorCounts.get(key) ?? 0) + 1);
}
for (const [key, count] of expectedErrorCounts) assert.equal(observedErrorCounts.get(key) ?? 0, count, `error count ${key}`);
const pairwise = await readCsv('panel_recovered/pairwise_judge_diagnostics.fivejudges.owner.csv', 'Pairwise');
assert.equal(pairwise.rows.length, 15);
assert.equal(new Set(pairwise.rows.map(row => row[pairwise.col.judge_id])).size, 5);
const summary = await readCsv('panel_recovered/judge_summary.fivejudges.owner.csv', 'JudgeSummary');
assert.equal(summary.rows.length, 5);

const provenance = await readCsv('recovery_composed_row_provenance.owner.csv', 'Provenance');
assert.equal(provenance.rows.length, 21);
assert.equal(provenance.rows.filter(row => row[provenance.col.row_source] === 'format_recovery').length, 1);
const recoveryRow = provenance.rows.find(row => row[provenance.col.row_source] === 'format_recovery');
assert.equal(`${recoveryRow[provenance.col.item_id]}/${recoveryRow[provenance.col.candidate_id]}`, 'src-03/cand-02');
assert.equal(String(recoveryRow[provenance.col.valid_for_composed_pointwise]).toLowerCase(), 'true');
assert.equal(String(recoveryRow[provenance.col.score_eligible]).toLowerCase(), 'true');
assert.equal(recoveryRow[provenance.col.logical_judge_id], 'nemotron-3.5-super-vl-120b-a12b-local-thinking-on');
assert(recoveryRow[provenance.col.source_bundle_path].includes('format-recovery-20261022/recovery_bundle_attempt-02.json'));
assert(recoveryRow[provenance.col.request_settings_json].includes('"reasoning_budget_tokens":8192'));
assert.equal(recoveryRow[provenance.col.retry_parent_request_id], 'pointwise-src-03-cand-02');
const customSummary = await readCsv('recovery_composed_judge_summary.owner.csv', 'JudgeSummary');
assert.equal(customSummary.rows.length, 5);
const nemo = customSummary.rows.find(row => String(row[customSummary.col.family]) === 'nvidia-nemotron');
assert.equal(Number(nemo[customSummary.col.pointwise_valid_n]), 21);
assert.equal(Number(nemo[customSummary.col.qualification_scalar_valid_variants]), 24);
const reliability = await readCsv('recovery_reliability_comparison.owner.csv', 'Reliability');
assert.equal(reliability.rows.length, 4);
const recovered = reliability.rows.find(row => row[reliability.col.comparison] === 'recovered_composed_n21');
assert.equal(Number(recovered[reliability.col.n_judges_actual]), 5);
assert.equal(Number(recovered[reliability.col.n_units_actual]), 21);
assert.equal(recovered[reliability.col.icc_ci_status], 'not available; implementation does not estimate ICC CI');
const invalid = await readCsv('recovery_original_invalid_annotations.owner.csv', 'InvalidAnnotations');
assert.equal(invalid.rows.length, 3);
for (const row of invalid.rows) {
  assert.equal(row[invalid.col.judge_id], 'nemotron-3.5-super-vl-120b-a12b-local-thinking-on');
  assert.equal(`${row[invalid.col.item_id]}/${row[invalid.col.candidate_id]}`, 'src-03/cand-02');
  assert.equal(row[invalid.col.raw_response_score], '92');
  assert.equal(row[invalid.col.raw_response_subscore_sum], '94.0');
  const rawBundleHash = row[invalid.col.raw_bundle_sha256] || row[invalid.col.bundle_sha256];
  assert.equal(rawBundleHash, 'fd29c121e25d8104021a9b8280da36eaaab574d0585d76b851ca17b335e962bc');
}

console.log(JSON.stringify({ status: 'PASS', artifactToolFromCSV: true, comparisonRows: comparison.rows.length, longRows: long.rows.length, wideRows: wide.rows.length, validityRows: validity.rows.length, errorRows: errors.rows.length, pairwiseRows: pairwise.rows.length, provenanceRows: provenance.rows.length, reliabilityRows: reliability.rows.length, preservedInvalidRows: invalid.rows.length, exactSourceTargetAndHashBinding: true, noDuplicateErrorAnnotations: true }));
