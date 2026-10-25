#!/usr/bin/env python3
"""Offline contract tests for the new-material blind packet.

These tests do not call a model, endpoint, browser, LiteLLM, or network.  They
check the generated public/private binding and the packet-level anti-leakage
invariants that a local/cloud worker must not assume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import unittest
from collections import Counter


ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
PRIVATE = ROOT / "private"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_strings(child)
    elif isinstance(value, str):
        yield value


class PacketContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        cls.ready = json.loads((ROOT / "packet_ready.json").read_text(encoding="utf-8"))
        cls.candidate_map = json.loads((PRIVATE / "candidate_map.json").read_text(encoding="utf-8"))
        cls.source_map = json.loads((PRIVATE / "source_and_image_map.json").read_text(encoding="utf-8"))
        cls.job_keymap = json.loads((PRIVATE / "job_keymap.json").read_text(encoding="utf-8"))
        cls.pair_selection = json.loads((PRIVATE / "pair_selection.json").read_text(encoding="utf-8"))
        with (ROOT.parent / "results" / "translation_results.csv").open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.model_ids = sorted({row["model_id"] for row in cls.rows})

    def test_manifest_and_root_gate(self):
        self.assertEqual(self.manifest["status"], "PACKET_READY_FOR_ROOT_QA")
        self.assertFalse(self.manifest["scoring_authorized"])
        self.assertEqual(self.ready["status"], "PACKET_READY_FOR_ROOT_QA")
        self.assertFalse(self.ready["scoring_authorized"])
        self.assertEqual(self.manifest["input_counts"], {
            "translation_result_rows": 238,
            "translator_slots": 7,
            "materials_total": 34,
            "text_materials": 30,
            "image_materials": 4,
            "models_each_case": 7,
        })
        self.assertEqual(self.manifest["job_counts"]["pointwise_jobs_per_judge"], 238)
        self.assertEqual(self.manifest["job_counts"]["pointwise_jobs_total"], 1190)
        self.assertEqual(self.manifest["job_counts"]["pairwise_jobs_per_judge"], 72)
        self.assertEqual(self.manifest["job_counts"]["pairwise_jobs_total"], 360)
        self.assertEqual(self.manifest["job_counts"]["substantive_jobs_total"], 1550)
        self.assertEqual(self.manifest["tracks"]["qualification"]["status"], "NOT_RERUN")

    def test_pointwise_counts_randomized_and_blind(self):
        all_job_ids = set()
        for slot in self.manifest["judge_slots"]:
            jobs = read_jsonl(PUBLIC / "pointwise" / f"{slot}.jsonl")
            self.assertEqual(len(jobs), 238)
            self.assertEqual(len({job["job_id"] for job in jobs}), 238)
            self.assertTrue(all(job["track"] == "pointwise" for job in jobs))
            self.assertEqual(Counter(job["source_kind"] for job in jobs), {"text": 210, "image": 28})
            self.assertEqual(len({job["blind_id"] for job in jobs}), 7)
            self.assertEqual(len({job["blind_source_id"] for job in jobs}), 34)
            for job in jobs:
                self.assertNotIn(job["job_id"], all_job_ids)
                all_job_ids.add(job["job_id"])
                self.assertIn(job["blind_id"], set(self.candidate_map["mapping_by_judge"][slot].values()))
                self.assertNotIn("source_id", job)
                self.assertNotIn("model_id", job)
                self.assertNotIn("source_owner", job)
                self.assertNotIn("source_en", job if job["source_kind"] == "image" else {})
                if job["source_kind"] == "text":
                    self.assertIn("source_en", job)
                    self.assertNotIn("image_alias", job)
                else:
                    self.assertIn("image_alias", job)
                    self.assertTrue(job["native_image_required"])
                    self.assertNotIn("source_en", job)

    def test_pairwise_scope_and_chain_coverage(self):
        expected_sources = self.pair_selection["selection"]["selected_material_ids_in_order"]
        self.assertEqual(expected_sources[-2:], ["img-09", "img-04"])
        self.assertEqual(len(expected_sources), 6)
        for source_id, details in self.pair_selection["pairs_by_source"].items():
            chain = details["candidate_chain_model_ids_private"]
            self.assertEqual(len(chain), 7)
            self.assertEqual(set(chain), set(self.model_ids))
            self.assertEqual(len(details["edges_private"]), 6)
        for slot in self.manifest["judge_slots"]:
            jobs = read_jsonl(PUBLIC / "pairwise" / f"{slot}.jsonl")
            self.assertEqual(len(jobs), 72)
            pair_keys = {job["pair_key"] for job in jobs}
            self.assertEqual(len(pair_keys), 36)
            self.assertEqual(Counter(job["orientation"] for job in jobs), {"AB": 36, "BA": 36})
            by_pair = {}
            for job in jobs:
                self.assertEqual(job["track"], "pairwise")
                self.assertIn(job["candidate_a"]["blind_id"], {"A", "B"})
                self.assertIn(job["candidate_b"]["blind_id"], {"A", "B"})
                self.assertNotEqual(job["candidate_a"]["blind_id"], job["candidate_b"]["blind_id"])
                self.assertNotIn("source_id", job)
                self.assertNotIn("model_id", job)
                self.assertNotIn("source_owner", job)
                by_pair.setdefault(job["pair_key"], []).append(job)
                if job["source_kind"] == "image":
                    self.assertNotIn("source_en", job)
                    self.assertTrue(job["native_image_required"])
                else:
                    self.assertIn("source_en", job)
            self.assertEqual({len(items) for items in by_pair.values()}, {2})
            self.assertEqual({item["orientation"] for items in by_pair.values() for item in items}, {"AB", "BA"})
            self.assertEqual({job["blind_source_id"] for job in jobs}, {
                self.source_map["source_alias_by_judge"][slot][source]
                for source in expected_sources
            })

    def test_images_are_byte_identical_and_public_aliases_are_opaque(self):
        image_rows = [row for row in self.rows if row["input_type"] == "image"]
        expected_hashes = {row["image_sha256"] for row in image_rows}
        aliases = self.source_map["image_aliases"]
        self.assertEqual(set(item["sha256"] for item in aliases.values()), expected_hashes)
        for image_id, details in aliases.items():
            path = ROOT / "public" / "images" / details["public_alias"]
            self.assertTrue(path.exists())
            self.assertEqual(sha256_file(path), details["sha256"])
            self.assertNotIn(image_id, details["public_alias"])

    def test_public_payload_has_no_identity_or_runtime_metadata(self):
        model_tokens = [token.lower() for token in self.model_ids]
        model_tokens += sorted({row["model_name"].lower() for row in self.rows})
        forbidden_keys = {"model", "model_id", "model_name", "model_alias", "source_owner", "rank", "old_score", "historical_score", "template_path", "template_sha256", "elapsed_seconds", "prompt_tokens", "completion_tokens"}
        for path in sorted((PUBLIC / "pointwise").glob("*.jsonl")) + sorted((PUBLIC / "pairwise").glob("*.jsonl")):
            for job in read_jsonl(path):
                self.assertFalse(forbidden_keys & set(job), f"forbidden public key in {path.name}")
                strings = [item.lower() for item in nested_strings(job)]
                self.assertFalse(any(token in text for token in model_tokens for text in strings), f"model identity leaked in {path.name}")
                # No old pilot qualification payload or score appears in a job.
                self.assertNotIn("qualification", job)
                self.assertNotIn("score_0_100", job)
        instructions = (PUBLIC / "instructions.md").read_text(encoding="utf-8")
        self.assertIn("native visual\njudging is mandatory", instructions.lower())
        self.assertIn("old pilot's qualification", instructions)
        self.assertIn("ABSTAIN", instructions)
        self.assertIn("30/30/20/20", instructions)

    def test_private_keymap_binds_every_public_job_without_duplicates(self):
        pointwise = self.job_keymap["pointwise"]
        pairwise = self.job_keymap["pairwise"]
        public_point_ids = set()
        public_pair_ids = set()
        for slot in self.manifest["judge_slots"]:
            public_point_ids |= {job["job_id"] for job in read_jsonl(PUBLIC / "pointwise" / f"{slot}.jsonl")}
            public_pair_ids |= {job["pair_job_id"] for job in read_jsonl(PUBLIC / "pairwise" / f"{slot}.jsonl")}
        self.assertEqual(public_point_ids, set(pointwise))
        self.assertEqual(public_pair_ids, set(pairwise))
        self.assertEqual(len(pointwise), 1190)
        self.assertEqual(len(pairwise), 360)
        self.assertTrue(all(value["model_id"] in self.model_ids for value in pointwise.values()))
        self.assertTrue(all(value["orientation"] in {"AB", "BA"} for value in pairwise.values()))


def write_test_report() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PacketContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": "new-material-packet-test-report-v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "packet_root": str(ROOT),
        "offline_only": True,
    }
    (ROOT / "tests" / "test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--report" in sys.argv[1:]:
        raise SystemExit(write_test_report())
    unittest.main(verbosity=2)
