import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const csvPath = "/benchmark/translation_benchmark_20260916_luna/deployment_translation_results_7models.csv";
const previewPath = "/private/tmp/deployment_translation_results_7models_preview.png";
const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Deployment7Models" });
const sheet = workbook.worksheets.getItem("Deployment7Models");
const used = sheet.getUsedRange();
const values = used.values;
const headers = values[0];
const data = values.slice(1);
const expectedHeaders = [
  "rank", "model", "case_id", "input_type", "source_english", "translation_zh_tw",
  "terminology_or_content_score", "semantic_or_medication_score", "fluency_score",
  "completeness_score", "total_score", "model_text_average", "model_image_accuracy",
  "model_combined_mean", "api_elapsed_s", "prompt_tok_s", "generation_tok_s",
  "template_status", "deployable", "deployment_requirements", "errors", "source_raw",
];
if (headers.join("\u001f") !== expectedHeaders.join("\u001f")) {
  throw new Error(`header mismatch: ${headers.join(",")}`);
}
if (data.length !== 28) throw new Error(`expected 28 data rows, got ${data.length}`);
const models = new Set(data.map((row) => row[1]));
if (models.size !== 7) throw new Error(`expected 7 models, got ${models.size}`);
const perModel = new Map();
for (const row of data) {
  perModel.set(row[1], (perModel.get(row[1]) || 0) + 1);
}
for (const [model, count] of perModel) {
  if (count !== 4) throw new Error(`${model} has ${count} rows, expected 4`);
}
const numericCols = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
for (const [index, row] of data.entries()) {
  for (const col of numericCols) {
    if (col === 9 && row[3] === "image" && row[col] === "") continue;
    if (row[col] === "") throw new Error(`blank numeric cell row ${index + 2}, col ${headers[col]}`);
    if (!Number.isFinite(Number(row[col]))) throw new Error(`non-numeric row ${index + 2}, col ${headers[col]}: ${row[col]}`);
  }
  if (row[3] === "image" && row[9] !== "") throw new Error(`image completeness should be blank row ${index + 2}`);
  if (row[5].includes("\n") && !csvText.includes(`"${row[5].replaceAll('"', '""')}"`)) {
    throw new Error(`multiline translation was not RFC4180 quoted for row ${index + 2}`);
  }
}
const inspect = await workbook.inspect({
  kind: "table",
  range: "Deployment7Models!A1:V12",
  include: "values,formulas",
  table_max_rows: 12,
  table_max_cols: 22,
});
console.log("rows=28 models=7 rows_per_model=4 columns=22");
console.log(`numeric_columns_valid=${numericCols.length}`);
console.log(inspect.ndjson);
sheet.getRange("A1:V29").format.wrapText = true;
sheet.getRange("A1:V1").format.font.bold = true;
const widths = {
  A: 48, B: 120, C: 150, D: 62, E: 300, F: 360, G: 82, H: 82, I: 82,
  J: 82, K: 82, L: 95, M: 100, N: 100, O: 94, P: 100, Q: 104, R: 160,
  S: 100, T: 280, U: 340, V: 300,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}29`).format.columnWidthPx = width;
}
sheet.freezePanes.freezeRows(1);
workbook.recalculate();
const preview = await workbook.render({
  sheetName: "Deployment7Models",
  range: "A1:N29",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
console.log(`rendered=${previewPath}`);
