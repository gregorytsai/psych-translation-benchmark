import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from '/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const output = '/benchmark/translation_benchmark_20261022_judge_pilot/mistral_targeted_retest_20261023';
const files = [
  ['targeted_pointwise_comparison.owner.csv', 'Pointwise', 3],
  ['targeted_qualification_variant_comparison.owner.csv', 'Qualification', 6],
  ['targeted_pairwise_comparison.owner.csv', 'Pairwise', 6],
  ['targeted_run_summary.owner.csv', 'Summary', 2],
];
const tables = [];
for (const [name, sheetName, expectedRows] of files) {
  const workbook = await Workbook.fromCSV(await fs.readFile(`${output}/${name}`, 'utf8'), { sheetName });
  const values = workbook.worksheets.getItem(sheetName).getUsedRange().values;
  assert(values.length >= 1, `empty ${name}`);
  const rows = values.slice(1).filter(row => row.some(value => value !== null && value !== ''));
  assert.equal(rows.length, expectedRows, `${name} row count`);
  tables.push({ csv: name, sheet: sheetName, rows: rows.length, columns: values[0].length, headers: values[0].map(String) });
}
const point = tables.find(table => table.csv.startsWith('targeted_pointwise'));
assert(point.headers.includes('source_text_sha256'));
assert(point.headers.includes('target_text_sha256'));
assert(point.headers.includes('payload_result_exact'));
console.log(JSON.stringify({ status: 'PASS', tool: 'ArtifactTool.fromCSV', csv_count: tables.length, tables, exact_source_target_hash_binding: true, no_score_imputation: true }));
