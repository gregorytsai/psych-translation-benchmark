import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import import_gemini_batches as igb
import make_gemini_prompts_v2 as v2
import aggregate_pilot as ap


class GeminiV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet_path = HERE / "judge_packet.json"
        cls.instructions_path = HERE / "judge_instructions.txt"
        cls.manifest_path = HERE / "gemini_copy_paste_v2" / "manifest.json"
        cls.packet = json.loads(cls.packet_path.read_text(encoding="utf-8"))
        cls.packet_sha = hashlib.sha256(cls.packet_path.read_bytes()).hexdigest()
        cls.specs = {spec["batch_id"]: spec for spec in v2.batch_specs(cls.packet)}

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pointwise_raw(self, batch_id="pointwise-src-01", candidate_ids=None):
        spec = self.specs[batch_id]
        payload = spec["task_payload"]
        candidate_ids = candidate_ids or list(spec["expected"]["candidate_ids"])
        records = []
        for candidate in payload["candidates"]:
            if candidate["blind_id"] not in candidate_ids:
                continue
            records.append({
                "item_id": payload["item_id"],
                "source_id": payload["source_id"],
                "source_text_sha256": payload["source_text_sha256"],
                "blind_id": candidate["blind_id"],
                "abstain": False,
                "score_0_100": 90,
                "subscores": {"terminology_locale": 27, "semantic_safety": 27, "fluency": 18, "completeness_format": 18},
                "critical": False,
                "errors": [],
            })
        return self.raw_envelope(spec, records)

    def qualification_raw(self):
        spec = self.specs["qualification-all"]
        records = []
        for item in spec["task_payload"]["qualification_items"]:
            records.append({
                "qualification_id": item["qualification_id"],
                "source_id": item["source_id"],
                "source_text_sha256": item["source_text_sha256"],
                "judgments": [
                    {
                        "blind_id": variant["blind_id"],
                        "abstain": False,
                        "score_0_100": 90,
                        "subscores": {"terminology_locale": 27, "semantic_safety": 27, "fluency": 18, "completeness_format": 18},
                        "critical": False,
                        "errors": [],
                    }
                    for variant in item["variants"]
                ],
            })
        return self.raw_envelope(spec, records)

    def pairwise_raw(self, batch_id):
        spec = self.specs[batch_id]
        records = []
        for task in spec["task_payload"]["pairwise_tasks"]:
            records.append({
                "pair_id": task["pair_id"],
                "item_id": task["item_id"],
                "source_id": task["source_id"],
                "source_text_sha256": task["source_text_sha256"],
                "orientation": task["orientation"],
                "candidate_a": task["candidate_a"]["blind_id"],
                "candidate_b": task["candidate_b"]["blind_id"],
                "winner": "A",
                "confidence": 0.75,
                "critical_override": False,
                "reason_codes": ["better_semantic_fidelity"],
            })
        return self.raw_envelope(spec, records)

    def raw_envelope(self, spec, records):
        return {
            "packet_id": self.packet["packet_id"],
            "packet_sha256": self.packet_sha,
            "judge_id": "<operator fills>",
            "model_label_from_ui": "<operator fills exact visible Gemini UI label>",
            "task_id": "<operator fills exact task id>",
            "track": spec["track"],
            "batch_id": "<operator fills exact batch id>",
            "records": records,
        }

    def adapt(self, root, batch_id, raw, *, allow_partial=False):
        raw_path = root / f"{batch_id}.raw.json"
        output = root / f"{batch_id}.adapted.json"
        self.write_json(raw_path, raw)
        return igb.adapt_capture(
            self.manifest_path,
            self.packet_path,
            self.instructions_path,
            raw_path,
            batch_id,
            "gemini-ui-test",
            "Gemini test UI label",
            f"run-{batch_id}",
            output,
            family="google-gemini",
            allow_partial=allow_partial,
        )

    def test_manifest_contains_frozen_instructions_and_six_named_batches(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["packet_sha256"], self.packet_sha)
        self.assertEqual(manifest["instructions_sha256"], hashlib.sha256(self.instructions_path.read_bytes()).hexdigest())
        names = {entry["filename"] for entry in manifest["files"]}
        self.assertEqual(names, {"qualification_all.md", "pointwise-src-01.md", "pointwise-src-02.md", "pointwise-src-03.md", "pairwise_AB.md", "pairwise_BA.md"})
        instructions = self.instructions_path.read_text(encoding="utf-8")
        for name in names:
            prompt = (self.manifest_path.parent / name).read_text(encoding="utf-8")
            self.assertIn(instructions, prompt)
            self.assertNotIn("candidate_mapping", prompt.lower())
            self.assertNotIn("answer_key", prompt.lower())
        self.assertEqual(v2.validate_rendered_manifest(self.manifest_path.parent, manifest, instructions)["status"], "PASS")

    def test_adapt_preserves_records_and_adds_only_provenance_envelope(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-adapt-") as directory:
            root = Path(directory)
            raw = self.pointwise_raw()
            before = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            adapted = self.adapt(root, "pointwise-src-01", raw)
            on_disk_raw = json.loads((root / "pointwise-src-01.raw.json").read_text(encoding="utf-8"))
            after = json.dumps(on_disk_raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.assertEqual(before, after)
            self.assertEqual(adapted["records"], raw["records"])
            self.assertTrue(adapted["provenance"]["judgment_payloads_unchanged"])
            self.assertEqual(adapted["coverage"]["status"], "complete")
            self.assertEqual(adapted["judge"]["provider"], "gemini_ui")

    def test_malformed_json_is_quarantined_without_output(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-malformed-") as directory:
            root = Path(directory)
            raw = root / "bad.json"
            raw.write_text('{"records": [', encoding="utf-8")
            output = root / "out.json"
            with self.assertRaises(igb.GeminiImportError):
                igb.adapt_capture(self.manifest_path, self.packet_path, self.instructions_path, raw, "pointwise-src-01", "j", "Gemini UI", "r", output)
            self.assertFalse(output.exists())

    def test_nonexact_quote_and_string_error_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-evidence-") as directory:
            root = Path(directory)
            raw = self.pointwise_raw()
            raw["records"][0]["errors"] = [{"target_exact_quote": "invented text", "source_span": "Panic disorder", "target_anchor": "驚恐"}]
            with self.assertRaises(igb.GeminiImportError):
                self.adapt(root, "pointwise-src-01", raw)
            raw = self.pointwise_raw()
            raw["records"][0]["errors"] = ["model wrote an error sentence"]
            with self.assertRaises(igb.GeminiImportError):
                self.adapt(root, "pointwise-src-01", raw)

    def test_source_span_and_anchor_are_diagnostic_not_transport_quarantine(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-evidence-audit-") as directory:
            root = Path(directory)
            raw = self.pointwise_raw()
            target = self.specs["pointwise-src-01"]["task_payload"]["candidates"][0]["target_zh_tw"]
            raw["records"][0]["errors"] = [{
                "target_exact_quote": target[:2],
                "source_evidence": "a near-exact explanatory sentence is allowed",
                "source_span": "invented source span",
                "target_anchor": "invented target anchor",
            }, {"target_exact_quote": "", "source_evidence": "omitted fact"}]
            adapted = self.adapt(root, "pointwise-src-01", raw)
            audit = adapted["coverage"]["evidence_audit"]
            self.assertEqual(audit["status"], "partial")
            self.assertEqual(audit["n_invalid"], 1)
            self.assertEqual(audit["n_unavailable"], 1)
            self.assertEqual(audit["entries"][0]["source_span_status"], "invalid_nonliteral")
            self.assertEqual(audit["entries"][0]["target_anchor_status"], "invalid_nonliteral")
            self.assertEqual(adapted["records"][0]["errors"][0]["source_span"], "invented source span")

    def test_assembler_rechecks_raw_sha_and_record_hashes(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-provenance-") as directory:
            root = Path(directory)
            self.adapt(root, "pointwise-src-01", self.pointwise_raw())
            adapted_path = root / "pointwise-src-01.adapted.json"
            adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
            adapted["records"][0]["score_0_100"] = 1
            self.write_json(adapted_path, adapted)
            with self.assertRaises(igb.GeminiImportError):
                igb.assemble_bundles(self.manifest_path, self.packet_path, self.instructions_path, [adapted_path], root / "assembled.json", allow_partial=True)

    def test_partial_bundle_is_explicit_and_does_not_fill_zero(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-partial-") as directory:
            root = Path(directory)
            raw = self.pointwise_raw(candidate_ids=["cand-01"])
            with self.assertRaises(igb.GeminiImportError):
                self.adapt(root, "pointwise-src-01", raw)
            adapted = self.adapt(root, "pointwise-src-01", raw, allow_partial=True)
            self.assertEqual(adapted["coverage"]["status"], "partial_exploratory")
            self.assertEqual(adapted["coverage"]["missing_record_keys"], [f"src-01/cand-0{i}" for i in range(2, 8)])
            self.assertEqual(len(adapted["records"]), 1)
            self.assertFalse(any(record.get("score_0_100") == 0 for record in adapted["records"]))

    def test_qualification_nested_variants_and_scalar_mismatch_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-qualification-") as directory:
            root = Path(directory)
            raw = self.qualification_raw()
            raw["records"][0]["judgments"][0]["score_0_100"] = 10
            adapted = self.adapt(root, "qualification-all", raw)
            self.assertEqual(adapted["records"][0]["judgments"][0]["score_0_100"], 10)
            self.assertEqual(len(adapted["records"]), 12)
            self.assertEqual(sum(len(record["judgments"]) for record in adapted["records"]), 24)

    def test_assembler_rejects_duplicate_batch_and_marks_partial(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-assemble-") as directory:
            root = Path(directory)
            first = self.adapt(root, "pointwise-src-01", self.pointwise_raw())
            first_path = root / "pointwise-src-01.adapted.json"
            with self.assertRaises(igb.GeminiImportError):
                igb.assemble_bundles(self.manifest_path, self.packet_path, self.instructions_path, [first_path, first_path], root / "duplicate.json")
            output = root / "partial.json"
            assembled = igb.assemble_bundles(self.manifest_path, self.packet_path, self.instructions_path, [first_path], output, allow_partial=True)
            self.assertEqual(assembled["coverage"]["status"], "partial_exploratory")
            self.assertEqual(assembled["coverage"]["missing_batches"], ["pairwise-AB", "pairwise-BA", "pointwise-src-02", "pointwise-src-03", "qualification-all"])
            self.assertEqual(len(assembled["records"]), len(first["records"]))
            self.assertTrue(all(record["batch_id"] == "pointwise-src-01" for record in assembled["records"]))

    def test_assembler_full_synthetic_six_batch_bundle(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-full-") as directory:
            root = Path(directory)
            paths = []
            for batch_id in ("qualification-all", "pointwise-src-01", "pointwise-src-02", "pointwise-src-03", "pairwise-AB", "pairwise-BA"):
                raw = self.qualification_raw() if batch_id == "qualification-all" else self.pairwise_raw(batch_id) if batch_id.startswith("pairwise") else self.pointwise_raw(batch_id)
                self.adapt(root, batch_id, raw)
                paths.append(root / f"{batch_id}.adapted.json")
            assembled = igb.assemble_bundles(self.manifest_path, self.packet_path, self.instructions_path, paths, root / "assembled.json")
            self.assertEqual(assembled["coverage"]["status"], "complete")
            self.assertEqual(assembled["coverage"]["observed_batch_count"], 6)
            self.assertEqual(len(assembled["records"]), 12 + 21 + 6)
            self.assertTrue(all(record["run_id"].startswith("run-") for record in assembled["records"]))
            pointwise = [record for record in assembled["records"] if record["track"] == "pointwise"]
            self.assertEqual(len(pointwise), 21)
            self.assertEqual({record["source_id"] for record in pointwise}, {"src-01", "src-02", "src-03"})
            self.assertEqual({record["raw_source_id"] for record in pointwise}, {"source-01", "source-02", "source-03"})
            alias = assembled["provenance"]["source_id_analysis_normalization"]
            self.assertEqual(alias["status"], "applied")
            self.assertEqual(len(alias["per_record"]), 21)

    def test_gemini_analysis_alias_keeps_three_source_clusters_with_three_existing_judges(self):
        with tempfile.TemporaryDirectory(prefix="gemini-v2-source-alias-") as directory:
            root = Path(directory)
            gemini_paths = []
            for batch_id in ("qualification-all", "pointwise-src-01", "pointwise-src-02", "pointwise-src-03", "pairwise-AB", "pairwise-BA"):
                raw = self.qualification_raw() if batch_id == "qualification-all" else self.pairwise_raw(batch_id) if batch_id.startswith("pairwise") else self.pointwise_raw(batch_id)
                self.adapt(root, batch_id, raw)
                gemini_paths.append(root / f"{batch_id}.adapted.json")
            gemini_assembled_path = root / "gemini-assembled.json"
            igb.assemble_bundles(self.manifest_path, self.packet_path, self.instructions_path, gemini_paths, gemini_assembled_path)
            pointwise_items = self.packet["pointwise"]["items"]
            existing_paths = []
            for judge_index in range(3):
                records = []
                for item in pointwise_items:
                    for candidate in item["candidates"]:
                        records.append({
                            "track": "pointwise",
                            "item_id": item["item_id"],
                            "source_id": item["item_id"],
                            "source_text_sha256": hashlib.sha256(item["source_en"].encode()).hexdigest(),
                            "blind_id": candidate["blind_id"],
                            "judge_id": f"existing-{judge_index}",
                            "run_id": f"existing-run-{judge_index}",
                            "score_0_100": 90,
                            "subscores": {"terminology_locale": 27, "semantic_safety": 27, "fluency": 18, "completeness_format": 18},
                            "critical": False,
                            "errors": [],
                        })
                bundle = {
                    "packet_id": self.packet["packet_id"],
                    "packet_sha256": self.packet_sha,
                    "judge_id": f"existing-{judge_index}",
                    "family": f"existing-family-{judge_index}",
                    "judge": {"judge_id": f"existing-{judge_index}", "family": f"existing-family-{judge_index}", "provider": "test"},
                    "records": records,
                }
                existing_path = root / f"existing-{judge_index}.json"
                self.write_json(existing_path, bundle)
                existing_paths.append(existing_path)
            metadata, rows = ap.load_bundles([*existing_paths, gemini_assembled_path])
            result = ap.aggregate(rows, metadata, required_judges=5, bootstrap_replicates=20)
            self.assertEqual(result["sample_scope"]["n_source_clusters"], 3)
            self.assertEqual(result["sample_scope"]["n_pointwise_units_collapsed"], 84)
            self.assertEqual(len(result["reliability_overall"]["matrix_units_for_icc"]), 21)
            self.assertTrue(all(len(unit) == 3 for unit in result["reliability_overall"]["matrix_units_for_icc"]))
            self.assertEqual(result["reliability_overall"]["icc_2_1_2_k"]["n_judges"], 4)
            self.assertEqual(result["judge_scope"]["n_distinct_judges"], 4)
            self.assertEqual(result["judge_scope"]["formal_panel"], False)
            self.assertEqual(result["pointwise"]["cand-01"]["n_items"], 3)
            self.assertEqual(result["pointwise"]["cand-01"]["n_distinct_judges"], 4)


if __name__ == "__main__":
    unittest.main()
