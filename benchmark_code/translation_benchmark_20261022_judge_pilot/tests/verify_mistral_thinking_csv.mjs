import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

const [root = '.'] = process.argv.slice(2);
const base = `${root}/translation_benchmark_20261022_judge_pilot`;
const packet = JSON.parse(await fs.readFile(`${base}/judge_packet.json`, 'utf8'));
const output = `${base}/mistral_thinking_comparison_20261022`;

async function table(path, sheetName) {
  const workbook = await Workbook.fromCSV(await fs.readFile(path, 'utf8'), { sheetName });
  const values = workbook.worksheets.getItem(sheetName).getUsedRange().values;
  assert(values.length >= 1, `empty table ${path}`);
  const headers = values[0].map(String);
  const index = Object.fromEntries(headers.map((name, i) => [name, i]));
  return { path, headers, index, rows: values.slice(1).filter(row => row.some(value => value !== null && value !== '')) };
}

const candidates = new Map();
for (const item of packet.pointwise.items) {
  for (const candidate of item.candidates) {
    candidates.set(`${item.item_id}/${candidate.blind_id}`, {
      source: item.source_en,
      target: candidate.target_zh_tw,
      sourceHash: createHash('sha256').update(item.source_en).digest('hex'),
    });
  }
}

const paired = await table(`${output}/mistral_thinking_off_vs_high_paired.owner.csv`, 'Paired');
assert.equal(paired.rows.length, 21);
assert.equal(new Set(paired.rows.map(row => `${row[paired.index.item_id]}/${row[paired.index.candidate_id]}`)).size, 21);
for (const row of paired.rows) {
  const key = `${row[paired.index.item_id]}/${row[paired.index.candidate_id]}`;
  const frozen = candidates.get(key);
  assert(frozen, `unexpected paired key ${key}`);
  assert.equal(row[paired.index.source_en], frozen.source, `source mismatch ${key}`);
  assert.equal(row[paired.index.target_zh_tw], frozen.target, `target mismatch ${key}`);
  assert.equal(row[paired.index.source_text_sha256], frozen.sourceHash, `source hash mismatch ${key}`);
  for (const mode of ['off', 'on']) {
    const score = row[paired.index[`${mode}_score_0_100`]];
    const eligible = String(row[paired.index[`${mode}_score_eligible`]]).toLowerCase() === 'true';
    if (score === '' || score === null) assert(!eligible, `${mode} missing score marked eligible ${key}`);
    else assert(Number.isFinite(Number(score)) && Number(score) >= 0 && Number(score) <= 100, `invalid ${mode} score ${key}`);
  }
}

const summary = await table(`${output}/mistral_thinking_off_vs_high_judge_summary.owner.csv`, 'Summary');
assert.equal(summary.rows.length, 2);
assert.deepEqual(new Set(summary.rows.map(row => row[summary.index.mode])), new Set(['off', 'on']));
for (const row of summary.rows) {
  assert.equal(Number(row[summary.index.pointwise_valid_rows]), 21);
  assert.equal(Number(row[summary.index.qualification_variants]), 24);
  assert.equal(Number(row[summary.index.pairwise_rows]), 6);
  assert.equal(Number(row[summary.index.pairwise_complete_swaps]), 3);
  assert(summary.index.template_verification_status !== undefined, 'template status column missing');
  if (row[summary.index.mode] === 'on') assert.equal(row[summary.index.template_verification_status], 'PENDING_PROVENANCE_INCOMPLETE');
}

const errors = await table(`${output}/mistral_thinking_off_vs_high_error_annotations.owner.csv`, 'Errors');
for (const row of errors.rows) {
  const key = `${row[errors.index.item_id]}/${row[errors.index.candidate_id]}`;
  const frozen = candidates.get(key);
  assert(frozen, `unexpected error key ${key}`);
  const quote = row[errors.index.target_exact_quote] ?? '';
  assert(quote === '' || frozen.target.includes(quote), `nonliteral quote ${key}`);
  const sourceHash = row[errors.index.source_text_sha256] ?? '';
  assert(sourceHash === '' || sourceHash === frozen.sourceHash, `error source hash mismatch ${key}`);
}

const pairwise = await table(`${output}/mistral_thinking_off_vs_high_pairwise.owner.csv`, 'Pairwise');
assert.equal(pairwise.rows.length, 6);
assert.deepEqual(new Set(pairwise.rows.map(row => row[pairwise.index.mode])), new Set(['off', 'on']));

const sensitivity = await table(`${output}/mistral_thinking_off_vs_high_panel_sensitivity.owner.csv`, 'Sensitivity');
assert.equal(sensitivity.rows.length, 14);
assert.deepEqual(new Set(sensitivity.rows.map(row => row[sensitivity.index.mode])), new Set(['off', 'on']));
for (const row of sensitivity.rows) assert.equal(row[sensitivity.index.candidate_coverage_status], 'complete_three_sources');

const reliability = await table(`${output}/mistral_thinking_off_vs_high_reliability.owner.csv`, 'Reliability');
assert.equal(reliability.rows.length, 2);
for (const row of reliability.rows) {
  assert.equal(Number(row[reliability.index.n_judges_actual]), 5);
  assert.equal(Number(row[reliability.index.n_units_actual]), 21);
  assert.equal(row[reliability.index.icc_ci_status], 'not available; implementation does not estimate ICC CI');
}

console.log(JSON.stringify({
  status: 'PASS',
  pairedRows: paired.rows.length,
  summaryRows: summary.rows.length,
  errorRows: errors.rows.length,
  pairwiseRows: pairwise.rows.length,
  sensitivityRows: sensitivity.rows.length,
  reliabilityRows: reliability.rows.length,
  exactFrozenSourceTargetAndHashes: true,
  noScoreImputation: true,
}));
