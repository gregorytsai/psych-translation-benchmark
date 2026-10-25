import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { Workbook } from "@oai/artifact-tool";

const reportDir = process.argv[2];
if (!reportDir) throw new Error("usage: qa_artifact_csv.mjs REPORT_DIR");

const expectedRows = {
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
  "settings_provenance.csv": null,
  "snapshot_diagnostics.csv": null,
  "error_annotations.csv": null,
  "invalid_annotations.csv": null,
};
const result = { status: "PASS", files: {}, checked_at: new Date().toISOString() };
for (const [name, count] of Object.entries(expectedRows)) {
  const file = path.join(reportDir, name);
  const csv = await fs.readFile(file, "utf8");
  const workbook = await Workbook.fromCSV(csv, { sheetName: name.replace(/\.csv$/, "") });
  const sheet = workbook.worksheets.getItem(name.replace(/\.csv$/, ""));
  const values = sheet.getUsedRange().values;
  assert(values.length >= 1, `${name}: no header`);
  const headers = values[0];
  assert(headers.every((x) => typeof x === "string" && x.length > 0), `${name}: empty header`);
  if (count !== null) assert.equal(values.length - 1, count, `${name}: row count`);
  result.files[name] = { data_rows: values.length - 1, columns: headers.length, used_range: sheet.getUsedRange().address ?? null };
}
const comparison = await fs.readFile(path.join(reportDir, "translation_comparison.csv"), "utf8");
const cw = await Workbook.fromCSV(comparison, { sheetName: "translation_comparison" });
const cv = cw.worksheets.getItem("translation_comparison").getUsedRange().values;
const ch = Object.fromEntries(cv[0].map((x, i) => [x, i]));
const scoreHeaders = cv[0].filter((x) => String(x).endsWith("_score"));
assert.equal(scoreHeaders.length, 5, "comparison has five judge score columns");
for (const row of cv.slice(1)) {
  const n = Number(row[ch.panel_n_valid]);
  assert(Number.isInteger(n) && n >= 0 && n <= 5, "panel n valid bounded");
  for (const h of scoreHeaders) {
    const value = row[ch[h]];
    if (value !== "" && value !== null && value !== undefined) {
      const score = Number(value);
      assert(Number.isFinite(score) && score >= 0 && score <= 100, `score bounded ${h}`);
    }
  }
}
console.log(JSON.stringify(result));
