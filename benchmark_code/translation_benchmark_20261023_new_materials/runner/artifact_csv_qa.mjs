import fs from "node:fs/promises";
import crypto from "node:crypto";
import { Workbook } from "@oai/artifact-tool";

const csvPath = process.argv[2];
const outPath = process.argv[3];
if (!csvPath || !outPath) throw new Error("usage: artifact_csv_qa.mjs INPUT.csv OUTPUT.json");

const csvText = await fs.readFile(csvPath, "utf8");
const csvSha256 = crypto.createHash("sha256").update(csvText, "utf8").digest("hex");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "translation_results" });
const overview = await workbook.inspect({
  select: "$.sheets[*]",
  include: ["name", "range", "regions.range", "regions.rowCount", "regions.columnCount"],
});
const regions = await workbook.inspect({
  select: "$.sheets[*].regions[*]",
  include: ["sheet", "range", "rowCount", "columnCount", "columns.name"],
});
const sheet = workbook.worksheets.getItem("translation_results");
const used = sheet.getUsedRange();
const values = used.values;
const headers = values[0] ?? [];
const expectedRows = 239; // header + 7*34 records
const expectedColumns = 38;
const qa = {
  checked_at: new Date().toISOString(),
  csv_path: csvPath,
  csv_sha256: csvSha256,
  workbook_from_csv: true,
  sheet_name: "translation_results",
  used_range: used.address ?? null,
  row_count_including_header: values.length,
  data_row_count: Math.max(0, values.length - 1),
  column_count: headers.length,
  expected_rows_including_header: expectedRows,
  expected_data_rows: expectedRows - 1,
  expected_columns: expectedColumns,
  header_nonempty: headers.every((value) => typeof value === "string" && value.length > 0),
  first_header: headers[0] ?? null,
  last_header: headers.at(-1) ?? null,
  first_data_sample: values[1] ?? null,
  last_data_sample: values.at(-1) ?? null,
  overview_records: overview.records ?? [],
  region_records: regions.records ?? [],
  pass: values.length === expectedRows && headers.length === expectedColumns && headers.every((value) => typeof value === "string" && value.length > 0),
};
await fs.writeFile(outPath, JSON.stringify(qa, null, 2) + "\n", "utf8");
console.log(JSON.stringify({ pass: qa.pass, rows: qa.data_row_count, columns: qa.column_count, csv_sha256: csvSha256 }));
if (!qa.pass) process.exitCode = 1;
