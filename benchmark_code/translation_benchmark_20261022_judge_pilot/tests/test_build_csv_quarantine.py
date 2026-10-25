from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import build_csv_results as bcr
from validate_import import validate_import


class BuildCsvQuarantineTests(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def make_packet(self, root: Path, candidate_ids: tuple[str, ...] = ("cand-a",)) -> tuple[Path, str, str]:
        source = "English source"
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        packet = {
            "packet_id": "pilot-p",
            "pointwise": {
                "items": [{
                    "item_id": "i1",
                    "source_id": "s1",
                    "source_en": source,
                    "source_text_sha256": source_hash,
                    "candidates": [
                        {"blind_id": candidate_id, "target_zh_tw": f"目標 {candidate_id}"}
                        for candidate_id in candidate_ids
                    ],
                }],
            },
        }
        packet_path = root / "judge_packet.json"
        self.write_json(packet_path, packet)
        return packet_path, source_hash, hashlib.sha256(packet_path.read_bytes()).hexdigest()

    def make_mapping(self, root: Path, candidate_ids: tuple[str, ...] = ("cand-a",)) -> Path:
        mapping_path = root / "mapping.json"
        self.write_json(mapping_path, {
            "mapping": {
                candidate_id: {"candidate_family": "test-family", "model_alias": f"alias-{candidate_id}"}
                for candidate_id in candidate_ids
            },
        })
        return mapping_path

    def pointwise_record(
        self,
        source_hash: str,
        candidate_id: str = "cand-a",
        judge_id: str = "judge-qwen",
        score: int | float = 90,
        run_id: str = "run-1",
        errors: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "track": "pointwise",
            "status": "ok",
            "item_id": "i1",
            "source_id": "s1",
            "source_text_sha256": source_hash,
            "candidate_id": candidate_id,
            "judge_id": judge_id,
            "run_id": run_id,
            "score_0_100": score,
            "subscores": {
                "terminology_locale": 27,
                "semantic_safety": 27,
                "fluency": 18,
                "completeness_format": 18,
            },
            "critical": False,
            "errors": errors or [],
        }

    def qualification_record(self, source_hash: str) -> dict[str, object]:
        return {
            "track": "qualification",
            "status": "ok",
            "item_id": "i1",
            "source_id": "s1",
            "source_text_sha256": source_hash,
            "candidate_id": "cand-a",
            "judge_id": "judge-qwen",
            "run_id": "run-qualification",
            "score_0_100": 95,
            # Deliberately totals 90.  Qualification scalar mismatch is a
            # warning and must not quarantine an otherwise valid pointwise row.
            "subscores": {
                "terminology_locale": 27,
                "semantic_safety": 27,
                "fluency": 18,
                "completeness_format": 18,
            },
            "critical": False,
            "errors": [],
        }

    def make_bundle(self, packet_sha: str, records: list[dict[str, object]], judge_id: str = "judge-qwen") -> dict[str, object]:
        return {
            "packet_id": "pilot-p",
            "packet_sha256": packet_sha,
            "judge": {"judge_id": judge_id, "family": "qwen"},
            "records": records,
        }

    def output_paths(self, root: Path) -> tuple[Path, ...]:
        return (
            root / "long.csv",
            root / "wide.csv",
            root / "comparison.csv",
            root / "validity.json",
            root / "validity.csv",
        )

    def build(
        self,
        packet_path: Path,
        mapping_path: Path,
        input_paths: list[Path],
        output_root: Path,
        errors_output: Path | None = None,
    ) -> dict[str, object]:
        outputs = self.output_paths(output_root)
        return bcr.build(packet_path, mapping_path, input_paths, *outputs, errors_output=errors_output)

    def test_error_annotations_preserve_original_fields_without_translation_duplication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-error-annotations-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)
            errors = [
                {
                    "severity": "major",
                    "type": "accuracy/terminology, verbatim",
                    "target_exact_quote": "目標 cand-a",
                    "source_evidence": 'source phrase, "verbatim"\nsecond line',
                    "source_span": "English source",
                    "target_anchor": "目標 cand-a",
                    "confidence": 0.83,
                },
                {
                    "severity": "minor",
                    "type": "fluency",
                    "target_exact_quote": "目標",
                    "source_evidence": "literal café",
                    "source_span": "source phrase",
                    "target_anchor": "目標",
                    "confidence": 0.42,
                },
            ]
            input_path = root / "judge-errors.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [self.pointwise_record(source_hash, errors=errors)]))
            errors_output = root / "owner" / "error_annotations.csv"

            result = self.build(packet_path, mapping_path, [input_path], root / "outputs", errors_output=errors_output)

            self.assertEqual(result["error_rows"], 2)
            with errors_output.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(bcr.ERROR_ANNOTATION_FIELDS))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["error_index"], "0")
            self.assertEqual(rows[0]["item_id"], "i1")
            self.assertEqual(rows[0]["source_id"], "s1")
            self.assertEqual(rows[0]["candidate_id"], "cand-a")
            self.assertEqual(rows[0]["model_alias"], "alias-cand-a")
            self.assertEqual(rows[0]["judge_id"], "judge-qwen")
            self.assertEqual(rows[0]["judge_family"], "qwen")
            self.assertEqual(rows[0]["target_exact_quote"], "目標 cand-a")
            self.assertEqual(rows[0]["source_evidence"], 'source phrase, "verbatim"\nsecond line')
            self.assertEqual(rows[0]["exact_quote_status"], "PASS")
            self.assertEqual(rows[0]["mapping_scope"], "owner_only_postjudge")
            self.assertEqual(rows[1]["error_index"], "1")
            self.assertEqual(rows[1]["target_exact_quote"], "目標")
            self.assertEqual(rows[1]["source_evidence"], "literal café")

            with (root / "outputs" / "comparison.csv").open(encoding="utf-8", newline="") as handle:
                comparison_reader = csv.DictReader(handle)
                comparison_rows = list(comparison_reader)
            self.assertNotIn("source_evidence", comparison_reader.fieldnames)
            self.assertNotIn("error_index", comparison_reader.fieldnames)
            self.assertEqual(comparison_rows[0]["target_zh_tw"], "目標 cand-a")

    def test_errors_output_with_no_errors_writes_header_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-no-errors-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)
            input_path = root / "judge-clean.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [self.pointwise_record(source_hash)]))
            errors_output = root / "owner" / "error_annotations.csv"

            result = self.build(packet_path, mapping_path, [input_path], root / "outputs", errors_output=errors_output)

            self.assertEqual(result["error_rows"], 0)
            with errors_output.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader), [])
                self.assertEqual(reader.fieldnames, list(bcr.ERROR_ANNOTATION_FIELDS))

    def test_existing_cli_arguments_work_without_optional_errors_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-cli-optional-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)
            input_path = root / "judge-clean.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [self.pointwise_record(source_hash)]))
            outputs = self.output_paths(root / "cli-outputs")
            argv = [
                "--packet", str(packet_path),
                "--mapping", str(mapping_path),
                "--input", str(input_path),
                "--long-output", str(outputs[0]),
                "--wide-output", str(outputs[1]),
                "--translation-comparison-output", str(outputs[2]),
                "--validity-json-output", str(outputs[3]),
                "--validity-csv-output", str(outputs[4]),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                status = bcr.main(argv)
            self.assertEqual(status, 0)
            self.assertTrue(outputs[0].exists())
            self.assertFalse((root / "cli-outputs" / "error_annotations.csv").exists())

    def test_raw_qwen_bad_quote_is_rejected_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-qwen-quote-") as directory:
            root = Path(directory)
            # Copy the existing raw Qwen import into the temp directory and
            # mutate only the in-memory copy.  The frozen/raw fixture itself
            # is never edited by this regression test.
            packet_path = HERE / "judge_packet.json"
            mapping_path = HERE / "private" / "candidate_mapping.private.json"
            bad = json.loads((HERE / "raw" / "qwen3.8-flash-next-local-pilot-v3-thinking-off-preserve.import-v2.json").read_text(encoding="utf-8"))
            pointwise = next(record for record in bad["records"] if record.get("track") == "pointwise")
            pointwise["pointwise"]["errors"].append({"target_exact_quote": "not present in target"})
            input_path = root / "qwen-bad-quote.json"
            self.write_json(input_path, bad)
            audit = validate_import(packet_path, input_path)
            self.assertEqual(audit["status"], "FAIL")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                self.build(packet_path, mapping_path, [input_path], root / "outputs")
            self.assertFalse((root / "outputs").exists())

    def test_packet_hash_and_source_hash_mismatches_are_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-bindings-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)

            packet_hash_bad = self.make_bundle("0" * 64, [self.pointwise_record(source_hash)])
            packet_hash_path = root / "packet-hash-mismatch.json"
            self.write_json(packet_hash_path, packet_hash_bad)
            with self.assertRaisesRegex(ValueError, "quarantined"):
                self.build(packet_path, mapping_path, [packet_hash_path], root / "packet-hash-output")
            self.assertFalse((root / "packet-hash-output").exists())

            source_hash_bad = self.make_bundle(packet_sha, [self.pointwise_record("f" * 64)])
            source_hash_path = root / "source-hash-mismatch.json"
            self.write_json(source_hash_path, source_hash_bad)
            with self.assertRaisesRegex(ValueError, "quarantined"):
                self.build(packet_path, mapping_path, [source_hash_path], root / "source-hash-output")
            self.assertFalse((root / "source-hash-output").exists())

    def test_partial_qualification_does_not_quarantine_pointwise(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-partial-qualification-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)
            input_path = root / "partial-qualification.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [
                self.pointwise_record(source_hash, score=90),
                self.qualification_record(source_hash),
            ]))

            result = self.build(packet_path, mapping_path, [input_path], root / "outputs")
            self.assertEqual(result["long_rows"], 1)
            with (root / "outputs" / "wide.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["judge_qwen_score_eligible_for_mean"], "True")
            self.assertEqual(rows[0]["score_mean_unweighted"], "90.0")
            self.assertEqual(rows[0]["score_mean_policy"], "validated_binding_and_row_shape_pass")
            validity = json.loads((root / "outputs" / "validity.json").read_text(encoding="utf-8"))
            self.assertEqual(validity["rows"][0]["validation_status"], "PASS")
            self.assertEqual(validity["rows"][0]["pointwise_track_status"], "eligible")
            self.assertEqual(validity["rows"][0]["qualification_track_status"], "partial")

    def test_invalid_pointwise_scalar_is_retained_but_excluded_from_mean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-pointwise-shape-") as directory:
            root = Path(directory)
            candidate_ids = ("cand-a", "cand-b")
            packet_path, source_hash, packet_sha = self.make_packet(root, candidate_ids)
            mapping_path = self.make_mapping(root, candidate_ids)
            invalid = self.pointwise_record(source_hash, candidate_id="cand-a", score=99)
            valid = self.pointwise_record(source_hash, candidate_id="cand-b", score=90)
            input_path = root / "pointwise-shape-invalid.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [invalid, valid]))
            audit = validate_import(packet_path, input_path)
            self.assertEqual(audit["status"], "FAIL")
            self.assertEqual(audit["track_eligibility"]["pointwise"]["status"], "quarantined")

            result = self.build(packet_path, mapping_path, [input_path], root / "outputs")
            self.assertEqual(result["long_rows"], 2)
            with (root / "outputs" / "long.csv").open(encoding="utf-8", newline="") as handle:
                long_rows = {row["candidate_id"]: row for row in csv.DictReader(handle)}
            self.assertIn("subscores sum", long_rows["cand-a"]["score_eligibility_issues"])
            self.assertEqual(long_rows["cand-b"]["score_eligibility_issues"], "")
            with (root / "outputs" / "wide.csv").open(encoding="utf-8", newline="") as handle:
                rows = {row["candidate_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["cand-a"]["judge_qwen_score_0_100"], "99")
            self.assertEqual(rows["cand-a"]["judge_qwen_score_eligible_for_mean"], "False")
            self.assertEqual(rows["cand-a"]["score_mean_unweighted"], "")
            self.assertEqual(rows["cand-a"]["score_eligible_n"], "0")
            self.assertEqual(rows["cand-b"]["judge_qwen_score_eligible_for_mean"], "True")
            self.assertEqual(rows["cand-b"]["score_mean_unweighted"], "90.0")

    def test_repeated_same_judge_item_candidate_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-csv-duplicate-unit-") as directory:
            root = Path(directory)
            packet_path, source_hash, packet_sha = self.make_packet(root)
            mapping_path = self.make_mapping(root)
            input_path = root / "duplicate-pointwise-unit.json"
            self.write_json(input_path, self.make_bundle(packet_sha, [
                self.pointwise_record(source_hash, score=90, run_id="run-1"),
                self.pointwise_record(source_hash, score=90, run_id="run-2"),
            ]))
            audit = validate_import(packet_path, input_path)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["duplicates"]["logical_repeat_rows"], 1)
            with self.assertRaisesRegex(ValueError, "duplicate pointwise judge unit"):
                self.build(packet_path, mapping_path, [input_path], root / "outputs")
            self.assertFalse((root / "outputs").exists())

    def test_gemini_context_uses_explicit_metadata_or_fresh_chat_fallback(self) -> None:
        self.assertEqual(
            bcr._judge_context("unknown", {"judge_context_packaging": "operator_declared_context"}),
            "operator_declared_context",
        )
        self.assertEqual(
            bcr._judge_context("google-gemini"),
            "fresh_chat_per_source_or_orientation",
        )
        self.assertEqual(
            bcr._judge_context("unknown", {"provider": "gemini_ui"}),
            "fresh_chat_per_source_or_orientation",
        )


if __name__ == "__main__":
    unittest.main()
