#!/usr/bin/env python3
"""Offline tests for the isolated, first-valid recovery lane."""

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "post_recovery_runner.py"
spec = importlib.util.spec_from_file_location("post_recovery_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class PostRecoveryPolicyTests(unittest.TestCase):
    def test_only_registry_targets_are_accepted_and_first_recovery_record_is_kept(self):
        target = ("judge-05", "pointwise", "pw-target")
        non_target = ("judge-05", "pointwise", "pw-valid-original")
        registry = [{
            "judge_slot": target[0], "track": target[1], "job_id": target[2],
            "material_id": "text-01", "baseline_status": "missing", "baseline_valid": "false",
        }]
        baseline = {
            non_target: [{"attempt": 0, "status": "ok", "parsed": {"score_0_100": 99}}],
            target: [{"attempt": 0, "status": "missing", "parsed": None}],
        }
        recovery = {
            non_target: [{"attempt": 1, "status": "ok", "parsed": {"score_0_100": 100}, "path": "/recovery/non-target.json"}],
            target: [
                {"attempt": 1, "status": "ok", "parsed": {"score_0_100": 82}, "path": "/recovery/target-first.json"},
                # A second parsed payload is a policy violation, never a score-selection opportunity.
                {"attempt": 2, "status": "ok", "parsed": {"score_0_100": 100}, "path": "/recovery/target-second.json"},
            ],
        }
        merged, stats = runner.merge_target_attempts(baseline, recovery, registry)
        self.assertEqual(stats["ignored_non_target_records"], 1)
        self.assertEqual(stats["score_payload_overflow_records"], 1)
        self.assertEqual(len(merged[non_target]), 1)
        self.assertEqual(len(merged[target]), 2)
        self.assertEqual(merged[target][-1]["path"], "/recovery/target-first.json")
        self.assertTrue(merged[target][-1]["recovery_lane"])

    def test_recovery_terminal_status_requires_a_real_terminal_record(self):
        registry = [{
            "judge_slot": "judge-05", "track": "pairwise", "job_id": "pair-target",
            "material_id": "text-01", "baseline_status": "missing", "baseline_valid": "false",
        }]
        key = ("judge-05", "pairwise", "pair-target")
        pending = runner.recovery_terminal_statuses({key: [{"status": "ok", "parsed": None}]}, registry)
        self.assertFalse(pending["complete"])
        self.assertEqual(len(pending["pending_keys"]), 1)
        complete = runner.recovery_terminal_statuses({key: [{"status": "missing", "parsed": None}]}, registry)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["terminal_statuses"]["/".join(key)], "missing")

    def test_archived_baseline_recomputes_identical_common_subset(self):
        # This is an offline fixture test against the frozen archived snapshot;
        # it reads raw and CSV only and never starts a model/API.
        aggregator = runner.load_aggregator()
        attempts, diagnostics, files, checkpoints = runner.collect_baseline_snapshot(aggregator)
        data = runner.build_data(aggregator, attempts, files, diagnostics, checkpoints)
        result = runner.same_subset_invariance(aggregator, data)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["baseline_common_case_counts"], {"text": 19, "image": 2})
        self.assertFalse(result["binding_mismatches"])
        self.assertFalse(result["reliability_mismatches"])
        self.assertFalse(result["model_mean_mismatches"])

    def test_gemini_capture_is_bound_to_manifest_parent_and_response_hash(self):
        aggregator = runner.load_aggregator()
        roots = [runner.RECOVERY_ROOT / "gemini", runner.RECOVERY_ROOT / "qwen"]
        all_attempts, _diagnostics, _files, _checkpoints = aggregator.collect_raw_snapshot(extra_roots=roots)
        raw_recovery, _observed = runner.split_recovery_attempts(all_attempts, roots)
        normalized, issues = runner.validate_recovery_attempts(raw_recovery, roots, runner.read_registry())
        key = ("judge-05", "pointwise", "pw-6abaa66ed2f8")
        self.assertIn(key, normalized)
        self.assertTrue(normalized[key][0]["provenance"]["recovery_manifest_verified"])
        self.assertEqual(normalized[key][0]["provenance"]["route_status"], "clean")
        self.assertFalse(any(key[2] in issue for issue in issues))

    def test_all_current_gemini_targets_are_strict_valid_without_qwen(self):
        aggregator = runner.load_aggregator()
        registry = runner.read_registry()
        roots = [runner.RECOVERY_ROOT / "gemini", runner.RECOVERY_ROOT / "qwen"]
        all_attempts, _diagnostics, _files, _checkpoints = aggregator.collect_raw_snapshot(extra_roots=roots)
        raw_recovery, _observed = runner.split_recovery_attempts(all_attempts, roots)
        recovery, issues = runner.validate_recovery_attempts(raw_recovery, roots, registry)
        self.assertFalse(issues)
        gemini_registry = [row for row in registry if row["judge_slot"] == "judge-05"]
        self.assertEqual(len(gemini_registry), 23)
        self.assertEqual(sum(len(recovery.get((row["judge_slot"], row["track"], row["job_id"]), [])) for row in gemini_registry), 23)
        baseline, _diag, _files, _checkpoints = runner.collect_baseline_snapshot(aggregator)
        merged, _stats = runner.merge_target_attempts(baseline, recovery, registry)
        self.assertEqual(_stats["metadata_invalid_records"], 0)
        self.assertEqual(_stats["score_payload_overflow_records"], 0)
        materials, _images, _manifest = aggregator.load_materials()
        point_jobs, pair_jobs, _private = aggregator.load_jobs()
        point_rows = aggregator.build_pointwise_rows(point_jobs, merged, materials)
        gem_point_ids = {row["job_id"] for row in gemini_registry if row["track"] == "pointwise"}
        point_selected = [row for row in point_rows if row["judge_slot"] == "judge-05" and row["job_id"] in gem_point_ids]
        self.assertEqual(len(point_selected), 14)
        self.assertTrue(all(row["scalar_valid"] for row in point_selected))
        pair_rows = aggregator.build_pairwise_rows(pair_jobs, merged)
        gem_pair_ids = {row["job_id"] for row in gemini_registry if row["track"] == "pairwise"}
        pair_selected = [row for row in pair_rows if row["judge_slot"] == "judge-05" and row["pair_job_id"] in gem_pair_ids]
        self.assertEqual(len(pair_selected), 9)
        self.assertTrue(all(row["valid"] for row in pair_selected))


if __name__ == "__main__":
    unittest.main()
