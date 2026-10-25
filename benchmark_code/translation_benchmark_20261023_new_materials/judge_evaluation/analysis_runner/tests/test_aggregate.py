#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve()
SCRIPT = HERE.parents[1] / "aggregate_new_panel.py"
spec = importlib.util.spec_from_file_location("aggregate_new_panel", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class AggregateFaultFixtures(unittest.TestCase):
    def fixture(self, name):
        return json.loads((HERE.parent / "fixtures" / name).read_text(encoding="utf-8"))

    def test_bool_is_not_integer(self):
        parsed = self.fixture("pointwise_retry_valid.json")
        parsed["score_0_100"] = True
        parsed["subscores"]["fluency"] = True
        result = module.validate_pointwise(parsed, parsed["blind_id"])
        self.assertFalse(result["valid"])
        self.assertIn("bad_score_integer_or_range", result["errors"])
        self.assertIn("bad_subscore:fluency", result["errors"])

    def test_sum_mismatch_is_invalid_and_not_repaired(self):
        parsed = self.fixture("pointwise_invalid_sum.json")
        result = module.validate_pointwise(parsed, parsed["blind_id"])
        self.assertFalse(result["valid"])
        self.assertIn("subscores_sum_mismatch", result["errors"])

    def test_first_valid_retry_is_selected_without_quality_comparison(self):
        invalid = self.fixture("pointwise_invalid_sum.json")
        valid = self.fixture("pointwise_retry_valid.json")
        attempts = [
            {"attempt": 0, "path": "first.json", "raw_hash": "a", "parsed": invalid, "status": "ok"},
            {"attempt": 1, "path": "repair.json", "raw_hash": "b", "parsed": valid, "status": "ok"},
        ]
        expected = {"blind_id": valid["blind_id"], "target_zh_tw": "這裡有錯誤片段", "source_kind": "text", "source_en": "source span"}
        selected = module.select_first_valid(attempts, expected, "pointwise", expected)
        self.assertEqual(selected["selected"]["attempt"], 1)
        self.assertTrue(selected["selected"]["validation_local"]["scalar_valid"])
        self.assertEqual(selected["selected_attempt_kind"], "unknown_recovery")
        self.assertFalse(selected["format_repair_used"])
        self.assertTrue(selected["unknown_recovery_used"])
        self.assertEqual(len(selected["all_attempts"]), 2)

    def test_explicit_format_repair_requires_raw_marker(self):
        invalid = self.fixture("pointwise_invalid_sum.json")
        valid = self.fixture("pointwise_retry_valid.json")
        attempts = [
            {"attempt": 0, "path": "first.json", "raw_hash": "a", "parsed": invalid, "status": "ok"},
            {"attempt": 1, "path": "repair.json", "raw_hash": "b", "parsed": valid, "status": "ok", "raw": {"repair": True}},
        ]
        expected = {"blind_id": valid["blind_id"], "target_zh_tw": "這裡有錯誤片段", "source_kind": "text", "source_en": "source span"}
        selected = module.select_first_valid(attempts, expected, "pointwise", expected)
        self.assertEqual(selected["selected_attempt_kind"], "format_repair")
        self.assertTrue(selected["format_repair_used"])

    def test_infrastructure_recovery_requires_no_prior_parsed_judgment(self):
        valid = self.fixture("pointwise_retry_valid.json")
        attempts = [
            {"attempt": 0, "path": "transport.json", "raw_hash": "a", "parsed": None, "status": "failed"},
            {"attempt": 1, "path": "retry1.json", "raw_hash": "b", "parsed": valid, "status": "ok"},
        ]
        expected = {"blind_id": valid["blind_id"], "target_zh_tw": "這裡有錯誤片段", "source_kind": "text", "source_en": "source span"}
        selected = module.select_first_valid(attempts, expected, "pointwise", expected)
        self.assertEqual(selected["selected_attempt_kind"], "infrastructure_recovery")
        self.assertTrue(selected["infrastructure_recovery_used"])

    def test_technical_execution_gate_excludes_structurally_valid_failed_raw(self):
        fixture = self.fixture("pointwise_technical_failed.json")
        expected = {"blind_id": fixture["parsed"]["blind_id"], "target_zh_tw": "目標", "source_kind": "text", "source_en": "source"}
        selected = module.select_first_valid([fixture], expected, "pointwise", expected)
        self.assertEqual(selected["status"], "technical_invalid")
        self.assertFalse(selected["selected"]["technical_diagnostics_local"]["trusted"])
        self.assertIn("raw_status_not_success", selected["selected"]["technical_diagnostics_local"]["errors"])
        self.assertIn("finish_reason_not_stop", selected["selected"]["technical_diagnostics_local"]["errors"])

    def test_model_identity_mismatch_is_not_accepted(self):
        parsed = self.fixture("pointwise_retry_valid.json")
        raw = {
            "attempt": 0,
            "status": "ok",
            "exit_code": 0,
            "provenance": {"requested_model": "model-A"},
            "resolved_model": "model-B",
            "parsed": parsed,
        }
        expected = {"blind_id": parsed["blind_id"], "target_zh_tw": "目標", "source_kind": "text", "source_en": "source"}
        selected = module.select_first_valid([raw], expected, "pointwise", expected)
        self.assertEqual(selected["status"], "technical_invalid")
        self.assertIn("model_identity_mismatch", selected["selected"]["technical_diagnostics_local"]["errors"])

    def test_native_image_gate_failure_never_becomes_a_score(self):
        parsed = self.fixture("pointwise_retry_valid.json")
        raw = {
            "attempt": 0,
            "status": "ok",
            "exit_code": 0,
            "actual_template_render": {"gate": {"pass": False, "actual_image": False, "native_image_marker_if_image": False}},
            "parsed": parsed,
        }
        expected = {"blind_id": parsed["blind_id"], "target_zh_tw": "目標", "source_kind": "image", "source_en": ""}
        selected = module.select_first_valid([raw], expected, "pointwise", expected)
        self.assertEqual(selected["status"], "technical_invalid")
        self.assertFalse(selected["selected"]["technical_diagnostics_local"]["trusted"])
        self.assertIn("native_image_render_missing", selected["selected"]["technical_diagnostics_local"]["errors"])

    def test_kendall_w_tie_correction_and_undefined_constant(self):
        self.assertAlmostEqual(module.kendall_w([[1, 1], [1, 1], [2, 2]]), 1.0)
        self.assertIsNone(module.kendall_w([[1, 1], [1, 1], [1, 1]]))
        imperfect = module.kendall_w([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
        self.assertIsNotNone(imperfect)
        self.assertGreaterEqual(imperfect, 0.0)
        self.assertLess(imperfect, 1.0)

    def test_terminal_coverage_excludes_known_unsupported_image_route(self):
        point_jobs = {
            ("judge-01", "p1"): {"source_kind": "text"},
            ("judge-03", "p2"): {"source_kind": "image"},
        }
        pair_jobs = {}
        attempts = {("judge-01", "pointwise", "p1"): [{"status": "ok", "parsed": {"status": "ok"}}]}
        coverage = module.terminal_attempt_coverage(point_jobs, pair_jobs, attempts)
        self.assertEqual(coverage["pointwise_design_unsupported_jobs"], 1)
        self.assertEqual(coverage["pointwise_terminal_expected_jobs"], 1)
        self.assertTrue(coverage["pointwise_terminal_complete"])

    def test_ok_without_parsed_result_is_interim_not_terminal(self):
        point_jobs = {("judge-01", "p1"): {"source_kind": "text"}}
        attempts = {("judge-01", "pointwise", "p1"): [{"status": "ok", "parsed": None}]}
        coverage = module.terminal_attempt_coverage(point_jobs, {}, attempts)
        self.assertEqual(coverage["pointwise_pending_jobs"], 1)
        self.assertEqual(coverage["pointwise_terminal_jobs"], 0)
        self.assertFalse(coverage["pointwise_terminal_complete"])

    def test_complete_checkpoint_missing_is_terminal_without_score(self):
        checkpoint = {
            "schema_version": module.GEMINI_CHECKPOINT_SCHEMA,
            "judge": {"judge_slot": "judge-05"},
            "counts": {"planned": 2, "ok": 0, "failed": 0, "missing": 2, "skipped": 0},
            "last_index": 2,
            "results": [
                {"request_id": "pointwise-pw-missing-1", "status": "missing"},
                {"request_id": "pairwise-pair-missing-1", "status": "missing"},
            ],
        }
        terminal = module.complete_gemini_checkpoint_terminal_statuses(checkpoint)
        self.assertEqual(len(terminal), 2)
        coverage = module.terminal_attempt_coverage(
            {("judge-05", "pw-missing-1"): {"source_kind": "text"}},
            {("judge-05", "pair-missing-1"): {"source_kind": "text"}},
            {},
            terminal,
        )
        self.assertEqual(coverage["pointwise_checkpoint_missing_terminal_jobs"], 1)
        self.assertEqual(coverage["pairwise_checkpoint_missing_terminal_jobs"], 1)
        self.assertEqual(coverage["pointwise_terminal_jobs"], 1)
        self.assertEqual(coverage["pairwise_terminal_jobs"], 1)
        self.assertEqual(coverage["pointwise_pending_jobs"], 0)
        self.assertTrue(coverage["pointwise_terminal_complete"])
        self.assertTrue(coverage["pairwise_terminal_complete"])

    def test_incomplete_checkpoint_does_not_close_terminal_gate(self):
        checkpoint = {
            "schema_version": module.GEMINI_CHECKPOINT_SCHEMA,
            "judge": {"judge_slot": "judge-05"},
            "counts": {"planned": 2, "ok": 0, "failed": 0, "missing": 2, "skipped": 0},
            "last_index": 1,
            "results": [{"request_id": "pointwise-pw-missing-1", "status": "missing"}],
        }
        self.assertEqual(module.complete_gemini_checkpoint_terminal_statuses(checkpoint), {})

    def test_common_case_intersection_waits_for_terminal_and_uses_thresholds(self):
        rows = []
        for material_index in range(30):
            material_id = f"text-{material_index:02d}"
            complete = material_index < 12
            for model_index, model_id in enumerate(module.MODEL_IDS):
                row = {
                    "material_id": material_id,
                    "source_id": f"src-{material_index % 6:02d}",
                    "source_cluster": f"src-{material_index % 6:02d}",
                    "source_kind": "text",
                    "model_id": model_id,
                    "panel_complete": complete,
                }
                for slot_index, slot in enumerate(module.TEXT_PANEL_SLOTS):
                    row[f"{slot}_score"] = 70 + ((material_index + model_index + slot_index) % 20) if complete else None
                if complete:
                    row["panel_mean"] = sum(row[f"{slot}_score"] for slot in module.TEXT_PANEL_SLOTS) / len(module.TEXT_PANEL_SLOTS)
                rows.append(row)
        not_ready = module.common_case_intersection(rows, "text", {"all_primary_terminal": False})
        self.assertEqual(not_ready["status"], "QUEUE_TERMINAL_INCOMPLETE")
        self.assertFalse(not_ready["analysis_ready"])
        ready = module.common_case_intersection(rows, "text", {"all_primary_terminal": True})
        self.assertTrue(ready["analysis_ready"])
        self.assertEqual(ready["eligible_item_count"], 12)
        self.assertEqual(ready["source_page_count"], 6)
        self.assertEqual(len(ready["excluded_case_ids"]), 18)
        module.annotate_common_case(rows, {"text": ready})
        reliability = module.build_reliability(
            rows,
            [],
            {"all_primary_terminal": True},
            {"text": ready},
        )
        self.assertEqual(reliability[0]["analysis_status"], "LIMITED_COMMON_CASE")
        self.assertEqual(reliability[0]["common_case_items"], 12)
        self.assertIsNotNone(reliability[0]["icc_2_1_absolute_agreement"])
        self.assertIsNotNone(reliability[0]["kendall_w_system_level"])
        bootstrap = module.build_bootstrap(
            rows,
            {},
            {"all_primary_terminal": True},
            {"text": ready},
        )
        self.assertEqual(len([r for r in bootstrap if r["track"] == "text"]), len(module.MODEL_IDS))
        self.assertTrue(all(r["status"] == "AVAILABLE_LIMITED_COMMON_CASE_SOURCE_CLUSTER_BOOTSTRAP" for r in bootstrap if r["track"] == "text"))
        lofo = module.build_lofo(rows, {"all_primary_terminal": True}, {"text": ready})
        self.assertEqual(lofo[0]["status"], "LIMITED_COMMON_CASE_LOFO")
        lofo_means = lofo[0]["candidate_means_noncomparative"]
        self.assertEqual(set(lofo_means), set(module.MODEL_IDS))
        self.assertGreater(len({round(float(v), 8) for v in lofo_means.values()}), 1)
        for model_index, model_id in enumerate(module.MODEL_IDS):
            expected = []
            for material_index in range(12):
                expected.append(sum(70 + ((material_index + model_index + slot_index) % 20) for slot_index in range(1, 5)) / 4)
            self.assertAlmostEqual(lofo_means[model_id], sum(expected) / len(expected), places=8)
        common_summary = module.build_common_case_model_summary(rows, {"text": ready}, {})
        common_text_summary = [r for r in common_summary if r["track"] == "text"]
        self.assertEqual(len(common_text_summary), len(module.MODEL_IDS))
        self.assertEqual({r["common_case_items"] for r in common_text_summary}, {12})
        self.assertGreater(len({round(float(r["common_case_panel_mean"]), 8) for r in common_text_summary}), 1)

    def test_image_english_visible_flag_is_observation(self):
        material = {"source_kind": "image", "ground_truth_lines": ["SERTRALINE — TAKE 50 mg ONCE DAILY"]}
        self.assertEqual(module.match_untranslated_visible_english("SERTRALINE — TAKE 50 mg ONCE DAILY", material), (True, "SERTRALINE — TAKE 50 mg ONCE DAILY"))
        self.assertEqual(module.match_untranslated_visible_english("舍曲林，每日一次", material), (False, ""))

    def test_who_provisional_reference_is_private_flag_only(self):
        material = self.fixture("image_who_reference_fixture.json")
        self.assertEqual(material["reference_text_kind"], "provisional_reference_text")
        self.assertIn("not_human_gold", material["ground_truth_status"])
        flag, span = module.match_untranslated_visible_english(
            "WHO policy brief on the health aspects of decriminalization of suicide and suicide attempts\n中文翻譯",
            material,
        )
        self.assertTrue(flag)
        self.assertEqual(span, material["ground_truth_lines"][0])

    def test_frozen_who_material_falls_back_to_provisional_reference(self):
        materials, _, _ = module.load_materials()
        self.assertEqual(materials["img-09"]["reference_text_kind"], "provisional_reference_text")
        self.assertIn("WHO policy brief on", materials["img-09"]["ground_truth_lines"])

    def test_pairwise_inconsistency_maps_to_abstain_not_tie(self):
        fixture = self.fixture("pairwise_inconsistent.json")
        rows = [
            {"judge_slot": "judge-01", "pair_key": "pair-1", "material_id": "text-01", "source_kind": "text", "model_a": "a", "model_b": "b", "canonical_left_model_id": "a", "canonical_right_model_id": "b", "orientation": "AB", "winner": fixture["AB"]["winner"], "valid": True, "status": "valid"},
            {"judge_slot": "judge-01", "pair_key": "pair-1", "material_id": "text-01", "source_kind": "text", "model_a": "b", "model_b": "a", "canonical_left_model_id": "a", "canonical_right_model_id": "b", "orientation": "BA", "winner": fixture["BA"]["winner"], "valid": True, "status": "valid"},
        ]
        out = module.build_pairwise_canonical(rows)
        self.assertEqual(out[0]["canonical_status"], "ABSTAIN_INCONSISTENT")
        self.assertIsNone(out[0]["canonical_winner"])

    def test_stable_read_rejects_malformed_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            value, digest, reason = module.read_json_stable(path)
            self.assertIsNone(value)
            self.assertIsNone(digest)
            self.assertTrue(str(reason).startswith("json_parse:"))

    def test_gemini_ui_embedded_response_is_ingested_with_provenance(self):
        parsed = self.fixture("pointwise_retry_valid.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            response_path = tmp_path / "response.raw.json"
            request_path = tmp_path / "request.json"
            response_path.write_text(json.dumps(parsed), encoding="utf-8")
            request = {
                "schema_version": module.GEMINI_UI_REQUEST_SCHEMA,
                "request_id": "pointwise-pw-ui-test",
                "job_id": "pw-ui-test",
                "judge_slot": "judge-05",
                "judge_id": "gemini-test",
                "requested_model": "Gemini test",
                "route": "antigravity_native_ui",
                "source_kind": "text",
                "schema_embedded": True,
                "public_schema_sha256": module._public_schema_sha256("pointwise"),
                "instruction_sha256": "i" * 64,
                "prompt_sha256": "p" * 64,
                "response_raw_path": str(response_path),
                "response": parsed,
            }
            records = module.gemini_ui_attempt_payloads(request_path, request)
            self.assertEqual(len(records), 1)
            raw_path, raw_hash, payload = records[0]
            self.assertEqual(raw_path, response_path)
            self.assertEqual(len(raw_hash), 64)
            record = module.raw_record_from_json(raw_path, payload, raw_hash, "gemini_ui_response")
            self.assertIsNotNone(record)
            self.assertEqual(record["provenance"]["schema_regime"], "schema_embedded_v2")
            expected = {"blind_id": parsed["blind_id"], "target_zh_tw": "目標", "source_kind": "text", "source_en": "source"}
            selected = module.select_first_valid([record], expected, "pointwise", expected)
            self.assertTrue(selected["selected"]["technical_diagnostics_local"]["trusted"])
            self.assertTrue(selected["selected"]["validation_local"]["scalar_valid"])

    def test_mistral_image_na_is_design_unsupported_not_technical_invalid(self):
        job = {
            "job_id": "pw-mistral-image",
            "material_id": "img-01",
            "source_kind": "image",
            "model_id": "gemma4-12b",
            "blind_id": "b-0123456789ab",
        }
        attempts = {
            ("judge-03", "pointwise", "pw-mistral-image"): [{
                "path": "mistral-image-na.json",
                "raw_hash": "a" * 64,
                "attempt": 0,
                "status": "N/A_not_supported",
                "parsed": None,
            }]
        }
        rows = module.build_pointwise_rows({("judge-03", "pw-mistral-image"): job}, attempts, {})
        self.assertEqual(rows[0]["status"], "unsupported_by_design")
        self.assertEqual(rows[0]["technical_errors"], [])
        self.assertFalse(rows[0]["scalar_valid"])

    def test_gemini_route_contaminated_payload_is_not_ingested(self):
        parsed = self.fixture("pointwise_retry_valid.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            response_path = tmp_path / "response.raw.json"
            request_path = tmp_path / "request.json"
            response_path.write_text(json.dumps(parsed), encoding="utf-8")
            request = {
                "schema_version": module.GEMINI_UI_REQUEST_SCHEMA,
                "request_id": "pointwise-pw-route-contaminated",
                "job_id": "pw-route-contaminated",
                "judge_slot": "judge-05",
                "judge_id": "gemini-test",
                "requested_model": "Gemini test",
                "route": "antigravity_native_ui",
                "route_contaminated": {
                    "status": "captured_native_ui",
                    "block_reason": "workspace_find_request_denied",
                },
                "source_kind": "text",
                "schema_embedded": True,
                "public_schema_sha256": module._public_schema_sha256("pointwise"),
                "instruction_sha256": "i" * 64,
                "prompt_sha256": "p" * 64,
                "response_raw_path": str(response_path),
                "response": parsed,
            }
            self.assertEqual(module.gemini_ui_attempt_payloads(request_path, request), [])


if __name__ == "__main__":
    unittest.main()
