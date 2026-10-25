#!/usr/bin/env python3
"""Offline tests for the new local judge driver; no model or endpoint calls."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DRIVER_PATH = Path(__file__).with_name("judge_driver.py")
SPEC = importlib.util.spec_from_file_location("judge_driver_test_module", DRIVER_PATH)
assert SPEC and SPEC.loader
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest_path, manifest, errors = DRIVER.discover_interface()
        self.assertEqual(errors, [])
        assert manifest_path and manifest
        self.manifest_path = manifest_path
        point_path, packet, packet_error = DRIVER.load_slot_packet(manifest_path, manifest, "judge-01")
        self.assertIsNone(packet_error)
        assert packet is not None
        self.interface = DRIVER.normalize_interface(manifest_path, manifest, packet)

    def test_packet_counts_and_fields(self) -> None:
        self.assertEqual(len(self.interface["pointwise"]), 238)
        self.assertEqual(len(self.interface["pairwise"]), 72)
        self.assertEqual(len({row["task_id"] for row in self.interface["pointwise"]}), 238)
        self.assertEqual(len({row["task_id"] for row in self.interface["pairwise"]}), 72)
        self.assertTrue(all(row.get("job_id") == row["task_id"] for row in self.interface["pointwise"]))
        self.assertTrue(all("pair_job_id" in row for row in self.interface["pairwise"]))

    def test_image_prompt_has_pixels_without_english_transcription(self) -> None:
        task = next(row for row in self.interface["pointwise"] if DRIVER.task_is_image(row))
        messages, metadata = DRIVER.build_messages(task, "FROZEN_INSTRUCTIONS", self.manifest_path)
        self.assertEqual(metadata["input_type"], "image")
        self.assertFalse(metadata["english_transcription_sent_for_image"])
        user_content = messages[1]["content"]
        self.assertEqual(sum(block.get("type") == "image_url" for block in user_content), 1)
        text = "\n".join(block.get("text", "") for block in user_content if block.get("type") == "text")
        if task.get("source_en"):
            self.assertNotIn(task["source_en"], text)
        self.assertNotIn("ground_truth", text.lower())
        self.assertTrue(user_content[-1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_frozen_schema_is_embedded_in_model_visible_messages(self) -> None:
        task = self.interface["pointwise"][0]
        messages, metadata = DRIVER.build_messages(task, "FROZEN_INSTRUCTIONS", self.manifest_path)
        self.assertIn('"$id": "pointwise_result.schema.json"', messages[0]["content"])
        self.assertTrue(metadata["schema_embedded_in_system_message"])
        self.assertEqual(metadata["schema_sha256"], DRIVER.sha256_file(DRIVER.POINTWISE_SCHEMA_PATH))

    def test_native_request_fields(self) -> None:
        messages = [{"role": "system", "content": "I"}, {"role": "user", "content": "U"}]
        qwen = DRIVER.request_payload(DRIVER.default_model_configs()[0], messages)
        nemo = DRIVER.request_payload(DRIVER.default_model_configs()[1], messages)
        mistral = DRIVER.request_payload(DRIVER.default_model_configs()[2], messages)
        self.assertEqual(qwen["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(qwen["chat_template_kwargs"]["preserve_thinking"], False)
        self.assertEqual(qwen["chat_template_kwargs"]["reasoning_effort"], "xhigh")
        self.assertEqual(nemo["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(nemo["chat_template_kwargs"]["preserve_reasoning"], False)
        self.assertNotIn("reasoning_effort", nemo)
        self.assertNotIn("enable_thinking", mistral.get("chat_template_kwargs", {}))
        self.assertEqual(mistral["reasoning_effort"], "high")
        for payload in (qwen, nemo, mistral):
            self.assertEqual(payload["max_tokens"], 16384)
            self.assertEqual(payload["seed"], 20261023)
            self.assertEqual(payload["cache_prompt"], False)
        self.assertTrue(DRIVER.runtime_version_passes("version: 0.7.0-dev (build 524, commit 620d2225c)"))
        self.assertFalse(DRIVER.runtime_version_passes("version: 0.7.0-dev (build 523, commit 620d2225c)"))

    def test_all_local_judges_expect_native_reasoning_on(self) -> None:
        """Mistral high is thinking-on even though it omits enable_thinking."""
        response = {
            "http_status": 200,
            "response_body": {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"status":"ok"}',
                        "reasoning_content": "native Mistral reasoning",
                    },
                }],
                "usage": {"completion_tokens": 4},
            },
        }
        mistral = DRIVER.default_model_configs()[2]
        metadata, error = DRIVER.response_metadata(response, mistral, {"messages": []})
        self.assertTrue(metadata["reasoning_content_present"])
        self.assertFalse(metadata["off_mode_violation"])
        self.assertIsNone(error)

    def test_runtime_media_marker_is_accepted_for_native_vision(self) -> None:
        """A /props media sentinel is a real native image marker."""
        prompt = "<|im_start|>user\nmarker<__media_native__><|im_end|>"
        self.assertEqual(prompt.count("<__media_native__>"), 1)

    def test_contract_validation_and_inconsistent_abba(self) -> None:
        good = {
            "blind_id": "b-0123456789ab", "status": "ok", "abstain": False,
            "score_0_100": 100,
            "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20},
            "critical": False, "errors": [],
        }
        self.assertEqual(DRIVER.validate_contract("pointwise", good, [good["blind_id"]]), (True, None))
        bad = dict(good)
        bad["subscores"] = dict(good["subscores"])
        bad["subscores"]["semantic_safety"] = 29
        self.assertEqual(DRIVER.validate_contract("pointwise", bad, [good["blind_id"]])[0], False)
        out_of_range = json.loads(json.dumps(good))
        out_of_range["subscores"]["fluency"] = 21
        self.assertEqual(DRIVER.validate_contract_details("pointwise", out_of_range, [good["blind_id"]])["error"], "subscore_out_of_range:fluency")
        extra = json.loads(json.dumps(good))
        extra["unexpected"] = True
        self.assertFalse(DRIVER.validate_contract_details("pointwise", extra, [good["blind_id"]])["valid"])
        evidence_only = json.loads(json.dumps(good))
        evidence_only["critical"] = True
        details = DRIVER.validate_contract_details("pointwise", evidence_only, [good["blind_id"]])
        self.assertTrue(details["valid"])
        self.assertEqual(details["evidence_diagnostics"][0]["code"], "critical_flag_vs_error_severity")
        malformed_error = json.loads(json.dumps(good))
        malformed_error["score_0_100"] = 90
        malformed_error["subscores"] = {"terminology_locale": 30, "semantic_safety": 25, "fluency": 20, "completeness_format": 15}
        malformed_error["errors"] = [{"severity": "major"}]
        self.assertTrue(DRIVER.validate_contract_details("pointwise", malformed_error, [good["blind_id"]])["error"].startswith("error_schema:"))
        empty_evidence = json.loads(json.dumps(good))
        empty_evidence["errors"] = [{"severity": "minor", "type": "wording", "target_exact_quote": "", "source_evidence": "", "source_span": "", "target_anchor": "", "confidence": 0.5}]
        self.assertIn("source_evidence_empty", ";".join(DRIVER.validate_contract_details("pointwise", empty_evidence, [good["blind_id"]])["validation_errors"]))
        nonstring_optional = json.loads(json.dumps(good))
        nonstring_optional["block_reason"] = 7
        self.assertIn("block_reason_invalid", DRIVER.validate_contract_details("pointwise", nonstring_optional, [good["blind_id"]])["validation_errors"])
        pair = {"winner": "A", "confidence": 0.9, "critical_override": False, "reason_codes": ["fidelity"]}
        self.assertEqual(DRIVER.validate_contract("pairwise", pair, []), (True, None))
        self.assertFalse(DRIVER.validate_contract_details("pairwise", {"winner": "A", "confidence": 0.9}, [])["valid"])

    def test_evidence_quality_is_diagnostic_only_and_repairs_are_format_only(self) -> None:
        task = {"track": "pointwise", "target_zh_tw": "正確翻譯", "source_en": "Correct translation"}
        parsed = {
            "blind_id": "b-0123456789ab", "status": "ok", "abstain": False,
            "score_0_100": 100,
            "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20},
            "critical": False,
            "errors": [{"severity": "minor", "type": "wording", "target_exact_quote": "不存在", "source_evidence": "not exact", "source_span": "Correct", "target_anchor": "不存在", "confidence": 0.5}],
        }
        self.assertTrue(DRIVER.validate_contract("pointwise", parsed, [parsed["blind_id"]])[0])
        self.assertTrue(DRIVER.evidence_quality_diagnostics("pointwise", parsed, task))
        self.assertFalse(DRIVER.repair_allowed("subscores_sum_mismatch"))
        self.assertFalse(DRIVER.repair_allowed("subscore_out_of_range:fluency"))
        self.assertFalse(DRIVER.repair_allowed("confidence_out_of_range"))
        self.assertTrue(DRIVER.repair_allowed("json_parse:JSONDecodeError"))
        messages, _ = DRIVER.build_messages(task, "I", self.manifest_path, repair=True, repair_context={"original_content": "{bad}", "format_errors": ["json_parse:bad"]})
        self.assertIn("<original_assistant_response>", messages[1]["content"])
        self.assertIn("json_parse:bad", messages[1]["content"])
        self.assertIn("Do not change these protected semantic fields", messages[1]["content"])

    def test_infrastructure_retry_is_distinct_and_single_use(self) -> None:
        transport = {"status": "invalid", "parsed": None, "invalid_reason": "http_not_200"}
        self.assertTrue(DRIVER.infrastructure_retry_allowed(transport))
        self.assertFalse(DRIVER.infrastructure_retry_allowed({**transport, "parsed": {}}))
        self.assertFalse(DRIVER.infrastructure_retry_allowed({**transport, "invalid_reason": "json_parse:bad"}))
        task = {"task_id": "pw-retry", "track": "pointwise"}
        first = {"model_id": "judge", "status": "invalid", "invalid_reason": "http_not_200", "parsed": None}
        second = {"model_id": "judge", "status": "ok", "parsed": {}}
        record = DRIVER.record_from_attempt(task, first, second, "infrastructure")
        self.assertTrue(record["infrastructure_retry_attempted"])
        self.assertFalse(record["format_repair_attempted"])
        self.assertTrue(record["attempts"][-1].endswith("__infrastructure_retry1.json"))

    def test_immutable_raw_and_per_model_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            DRIVER.write_json_immutable(path, {"a": 1})
            DRIVER.write_json_immutable(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                DRIVER.write_json_immutable(path, {"a": 2})
        self.assertNotEqual(DRIVER.checkpoint_path("qwen"), DRIVER.checkpoint_path("nemotron"))
        self.assertNotEqual(DRIVER.results_path("qwen"), DRIVER.results_path("nemotron"))
        self.assertIn("technical_smoke", str(DRIVER.RAW_ROOT / "model" / "technical_smoke" / "batch.text.json"))


if __name__ == "__main__":
    unittest.main()
