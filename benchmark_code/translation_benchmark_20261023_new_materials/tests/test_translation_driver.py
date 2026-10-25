"""Offline contract checks for the new-materials translation driver.

These tests intentionally do not start llama-server or make HTTP requests.  The
Gemma image assertions read the already captured raw smoke records so a
checker correction can be validated without re-running an image case.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_driver() -> Any:
    path = ROOT / "runner" / "run_translation.py"
    spec = importlib.util.spec_from_file_location("new_materials_translation_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import translation driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TranslationDriverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = load_driver()
        cls.config = json.loads((ROOT / "configs" / "translation_config.json").read_text(encoding="utf-8"))
        cls.selection = json.loads((ROOT / "materials" / "image_selection.json").read_text(encoding="utf-8"))
        cls.image = cls.selection["selected_images"][0]
        cls.text = json.loads((ROOT / "materials" / "frozen_text_materials.jsonl").read_text(encoding="utf-8").splitlines()[0])

    def model(self, model_id: str) -> dict[str, Any]:
        return next(model for model in self.config["models"] if model["id"] == model_id)

    def image_case(self) -> dict[str, Any]:
        # Include a sentinel GT value to prove the payload builder ignores case
        # metadata even when a future judging field is present on the case.
        return {
            "input_type": "image",
            "source_en": None,
            "image_path": self.image["source_absolute_path"],
            "image_ground_truth_en": "GT_SENTINEL_MUST_NOT_REACH_PROMPT",
        }

    @staticmethod
    def content_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
        content = payload["messages"][0]["content"]
        if not isinstance(content, list):
            raise AssertionError("expected an OpenAI content-part array")
        return content

    def read_selected_smoke(self, filename: str, case_id: str) -> dict[str, Any]:
        path = ROOT / "technical_smoke" / filename
        self.assertTrue(path.is_file(), path)
        for line in path.read_text(encoding="utf-8").splitlines():
            wrapper = json.loads(line)
            selected = wrapper.get("selected", wrapper)
            if selected.get("case_id") == case_id:
                return selected
        self.fail(f"{case_id} not found in {path}")

    def test_translategemma_text_is_source_only(self) -> None:
        model = self.model("translategemma-27b")
        case = {"input_type": "text", "source_en": self.text["source_en"], "image_path": None}
        payload = self.driver.make_payload(self.config, model, case)
        content = payload["messages"][0]["content"]
        self.assertEqual(content, self.text["source_en"])
        self.assertNotIn(self.config["shared_instruction"], content)
        self.assertNotIn("<start_of_turn>", content)

    def test_translategemma_image_has_one_text_and_one_image_part(self) -> None:
        payload = self.driver.make_payload(self.config, self.model("translategemma-12b"), self.image_case())
        parts = self.content_parts(payload)
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        self.assertEqual(parts[1]["image_url"]["url"][:22], "data:image/png;base64,")
        self.assertNotIn("GT_SENTINEL_MUST_NOT_REACH_PROMPT", json.dumps(payload, ensure_ascii=False))

    def test_gemma_image_has_one_instruction_and_one_image_url(self) -> None:
        payload = self.driver.make_payload(self.config, self.model("gemma4-26b"), self.image_case())
        parts = self.content_parts(payload)
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        self.assertEqual(parts[0]["text"].count(self.config["shared_instruction"]), 1)
        self.assertNotIn("GT_SENTINEL_MUST_NOT_REACH_PROMPT", json.dumps(payload, ensure_ascii=False))

    def test_image_redaction_removes_data_url_and_gt(self) -> None:
        payload = self.driver.make_payload(self.config, self.model("translategemma-4b"), self.image_case())
        redacted = self.driver.redacted(payload)
        redacted_text = json.dumps(redacted, ensure_ascii=False)
        self.assertNotIn("data:image/png;base64,", redacted_text)
        self.assertNotIn("GT_SENTINEL_MUST_NOT_REACH_PROMPT", redacted_text)
        image_part = redacted["messages"][0]["content"][1]
        self.assertEqual(image_part["image_url"]["url"], "<redacted-local-image-data-url>")
        self.assertRegex(image_part["image_url"]["data_url_sha256"], r"^[0-9a-f]{64}$")

    def test_gemma_raw_smokes_use_current_paired_image_markers(self) -> None:
        for filename in ("gemma4-e4b.jsonl", "gemma4-12b.jsonl", "gemma4-26b.jsonl", "gemma4-31b-qat.jsonl"):
            record = self.read_selected_smoke(filename, "technical-image-marker")
            self.assertEqual(record["status"], "technical_ok", filename)
            self.assertEqual(record["http_envelope"]["http_status"], 200, filename)
            self.assertEqual(record["finish_reason"], "stop", filename)
            self.assertTrue(record["content"].strip(), filename)
            prompt = record["verbose"]["prompt"]
            # Gemma4's native pair is <|image> ... <image|>, not <|image|>.
            self.assertEqual(prompt.count("<|image>"), 1, filename)
            self.assertEqual(prompt.count("<image|>"), 1, filename)
            self.assertEqual(prompt.count("<|image|>"), 0, filename)
            self.assertFalse(record["verbose"]["generation_settings"]["reasoning_in_content"], filename)

    def test_translategemma_raw_smokes_have_one_adapter_render(self) -> None:
        for filename in ("translategemma-27b.jsonl", "translategemma-12b.jsonl", "translategemma-4b.jsonl"):
            record = self.read_selected_smoke(filename, "technical-image-marker")
            self.assertEqual(record["status"], "technical_ok", filename)
            prompt = record["verbose"]["prompt"]
            self.assertEqual(prompt.count("<start_of_turn>user"), 1, filename)
            self.assertEqual(prompt.count("<start_of_turn>model"), 1, filename)
            self.assertEqual(prompt.count("<end_of_turn>"), 1, filename)
            self.assertEqual(prompt.count("<start_of_image>"), 1, filename)
            self.assertEqual(prompt.count("<end_of_image>"), 1, filename)
            self.assertEqual(prompt.count("You are a professional English (en) to Chinese (zh-TW) translator."), 1, filename)

    def test_raw_smoke_records_are_retained_for_text_and_image(self) -> None:
        for path in sorted((ROOT / "technical_smoke").glob("*.jsonl")):
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(records), 2, path.name)
            case_ids = {item.get("selected", item).get("case_id") for item in records}
            self.assertIn("technical-text-marker", case_ids, path.name)
            self.assertIn("technical-image-marker", case_ids, path.name)
            for wrapper in records:
                selected = wrapper.get("selected", wrapper)
                self.assertIn("status", selected, path.name)
                self.assertIn("attempts", wrapper, path.name)

    def test_model_inventory_has_all_semantically_paired_routes(self) -> None:
        inventory_path = ROOT / "raw" / "model_inventory.json"
        self.assertTrue(inventory_path.is_file())
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(set(inventory["models"]), {model["id"] for model in self.config["models"]})
        for model in self.config["models"]:
            item = inventory["models"][model["id"]]
            self.assertTrue(Path(item["model_path"]).is_file(), model["id"])
            self.assertTrue(Path(item["mmproj_path"]).is_file(), model["id"])
            self.assertEqual(item["mmproj_metadata"]["general.type"], "mmproj", model["id"])
            expected_arch = "gemma3" if model["family"] == "translategemma" else "gemma4"
            self.assertEqual(item["model_metadata"]["general.architecture"], expected_arch, model["id"])

    @unittest.skipUnless(shutil.which("vm_stat"), "macOS vm_stat is unavailable")
    def test_ram_headroom_result_uses_the_32_gib_threshold(self) -> None:
        headroom = self.driver.vm_headroom()
        self.assertEqual(headroom["required_gib"], 32)
        self.assertEqual(
            headroom["pass"],
            headroom["available_bytes_conservative"] >= 32 * self.driver.GIB,
        )


if __name__ == "__main__":
    unittest.main()
