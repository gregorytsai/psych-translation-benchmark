import hashlib
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import aggregate_pilot as ap
import build_csv_results as bcr
import make_gemini_prompts as mgp
import qualification_summary as qs
from validate_import import validate_import


def normalized(row, metadata=None):
    return ap._normalize_row(row, metadata or {})


def pointwise(item="i1", source="s1", candidate="cand-a", judge="j1", score=95, critical=False, run="r1", source_hash=None, errors=None, status="ok"):
    source_hash = source_hash or hashlib.sha256(("source:" + source).encode()).hexdigest()
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        remaining = float(score)
        terminology = min(30.0, remaining)
        remaining -= terminology
        semantic = min(30.0, remaining)
        remaining -= semantic
        fluency = min(20.0, remaining)
        remaining -= fluency
        completeness = remaining
        subscores = {
            "terminology_locale": int(terminology) if terminology.is_integer() else terminology,
            "semantic_safety": int(semantic) if semantic.is_integer() else semantic,
            "fluency": int(fluency) if fluency.is_integer() else fluency,
            "completeness_format": int(completeness) if completeness.is_integer() else completeness,
        }
    else:
        subscores = None
    return normalized(
        {
            "track": "pointwise",
            "status": status,
            "item_id": item,
            "source_id": source,
            "source_text_sha256": source_hash,
            "candidate_id": candidate,
            "judge_id": judge,
            "run_id": run,
            "score_0_100": score,
            "critical": critical,
            "errors": errors or [],
            "subscores": subscores,
        }
    )


class AggregatePilotTests(unittest.TestCase):
    def test_owner_translation_comparison_has_all_21_exact_nonempty_rows(self):
        output_root = Path(tempfile.mkdtemp(prefix="judge-pilot-csv-test-"))
        try:
            result = bcr.build(
                HERE / "judge_packet.json",
                HERE / "private/candidate_mapping.private.json",
                [
                    HERE / "raw/gpt-6.1-sol-codex-blind.json",
                    HERE / "raw/qwen3.8-flash-next-local-pilot-v3-thinking-off-preserve.import-v2.json",
                ],
                output_root / "long.csv",
                output_root / "wide.csv",
                output_root / "translation_comparison.csv",
                output_root / "validity.json",
                output_root / "validity.csv",
            )
            self.assertEqual(result["translation_comparison_rows"], 21)
            with (output_root / "translation_comparison.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 21)
            self.assertTrue(all(row["source_en"] for row in rows))
            self.assertTrue(all(row["target_zh_tw"] for row in rows))
            self.assertEqual({row["source_id"] for row in rows}, {"src-01", "src-02", "src-03"})
            self.assertTrue(all(row["score_mean_policy"] == "validated_binding_and_row_shape_pass" for row in rows))
        finally:
            for child in sorted(output_root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            output_root.rmdir()

    def test_krippendorff_missing_unequal_rater_weighting(self):
        result = ap.krippendorff_alpha([["a", "a"], ["a", "b", "b"]], "nominal")
        self.assertEqual(result["status"], "ok")
        # Coincidence matrix weighting gives Do=.4, De=.6, alpha=1/3.
        self.assertAlmostEqual(result["estimate"], 1 / 3)

    def test_span_f1_zero_for_nonoverlap_and_one_only_for_both_empty(self):
        nonoverlap = ap.span_agreement([
            pointwise(judge="j1", errors=[{"target_start": 0, "target_end": 2}]),
            pointwise(judge="j2", errors=[{"target_start": 4, "target_end": 6}]),
        ])
        self.assertEqual(nonoverlap["f1"], 0.0)
        one_empty = ap.span_agreement([
            pointwise(judge="j1", errors=[{"target_start": 0, "target_end": 2}]),
            pointwise(judge="j2", errors=[]),
        ])
        self.assertEqual(one_empty["f1"], 0.0)
        both_empty = ap.span_agreement([
            pointwise(judge="j1", errors=[]),
            pointwise(judge="j2", errors=[]),
        ])
        self.assertEqual(both_empty["f1"], 1.0)

    def test_span_missing_coordinates_are_partial_not_nonoverlap(self):
        rows = [
            pointwise(judge="j1", errors=[{"severity": "major", "type": "accuracy", "target_exact_quote": "甲"}]),
            pointwise(judge="j2", errors=[{"severity": "major", "type": "accuracy", "target_exact_quote": "乙"}]),
        ]
        agreement = ap.span_agreement(rows)
        self.assertEqual(agreement["status"], "unavailable")
        self.assertEqual(agreement["missing_offsets"], 2)
        self.assertNotIn("f1", agreement)
        consensus = ap._span_consensus(rows)
        self.assertEqual(consensus["status"], "partial")
        self.assertEqual(consensus["missing_offsets"], 2)
        self.assertEqual(consensus["singleton_error_clusters"], 0)

    def test_span_consensus_uses_strict_majority_for_even_panels(self):
        rows = [
            pointwise(judge=f"j{i}", errors=[{"target_start": 0, "target_end": 2}])
            for i in range(4)
        ]
        rows[2]["_errors"] = []
        rows[3]["_errors"] = []
        consensus = ap._span_consensus(rows)
        self.assertEqual(consensus["panel_consensus_error_clusters"], 0)
        self.assertEqual(consensus["disputed_error_clusters"], 1)

    def test_ordinal_delta_is_pooled_probability_distance(self):
        result = ap.krippendorff_alpha([[0, 2], [0, 1]], "ordinal", [0, 1, 2])
        self.assertEqual(result["distance"], "Krippendorff ordinal squared pooled cumulative midpoint probability")
        # The 0-vs-2 disagreement uses (p0/2+p1+p2/2)^2, not an index distance.
        self.assertAlmostEqual(result["observed_disagreement"], 0.265625)

    def test_gwet_ac2_does_not_use_weighted_kappa_chance_term(self):
        units = [[0, 0], [1, 1], [2, 2], [0, 1]]
        result = ap.gwet_ac(units, "ordinal", [0, 1, 2], "linear")
        weighted_kappa_pe = sum(
            (0.375 if first == 0 else 0.25 if first == 2 else 0.375)
            * (0.375 if second == 0 else 0.25 if second == 2 else 0.375)
            * (1 - abs(first - second) / 2)
            for first in range(3)
            for second in range(3)
        )
        self.assertAlmostEqual(result["chance_agreement_gwet"], 0.546875)
        self.assertNotAlmostEqual(result["chance_agreement_gwet"], weighted_kappa_pe)
        self.assertIn("not weighted-kappa", result["chance_term_note"])

    def test_gwet_observed_agreement_weights_units_not_extra_pairs(self):
        # The second unit has three annotations and therefore three pairs;
        # standard equal-unit Gwet observed agreement must not let it dominate.
        result = ap.gwet_ac([[0, 0], [0, 1, 1]], "nominal", [0, 1])
        self.assertAlmostEqual(result["observed_agreement"], (1.0 + (1.0 / 3.0)) / 2.0)
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_units_with_pairs"], 2)

    def test_binary_positive_agreement_is_unavailable_when_no_positive_labels(self):
        result = ap.binary_agreement([
            pointwise(judge="j1", critical=False),
            pointwise(judge="j2", critical=False),
        ])
        self.assertAlmostEqual(result["raw_agreement"], 1.0)
        self.assertIsNone(result["positive_agreement"])
        self.assertEqual(result["positive_pair_count"], 0)
        self.assertAlmostEqual(result["negative_agreement"], 1.0)

    def test_icc_absolute_agreement_preserves_rater_offset(self):
        result = ap.icc_two_way_random_absolute([[1, 2], [2, 3], [3, 4]])
        self.assertAlmostEqual(result["icc_2_1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["icc_2_k"], 0.8)

    def test_repeated_runs_count_one_distinct_critical_judge(self):
        rows = [
            pointwise(run="r1", critical=True, score=80),
            pointwise(run="r2", critical=True, score=78),
            pointwise(run="r3", critical=False, score=90),
        ]
        result = ap.aggregate(rows, {"packet_id": "p", "packet_sha256": "x"})
        summary = result["pointwise"]["cand-a"]
        self.assertEqual(summary["n_distinct_judges"], 1)
        self.assertEqual(summary["critical_vote_count_distinct_judges"], 1)
        self.assertEqual(summary["missingness"]["n_ok_rows"], 1)
        self.assertEqual(summary["missingness"]["status_counts"], {"ok": 1})
        self.assertEqual(summary["deployment_status"], "INSUFFICIENT_PANEL")
        self.assertEqual(result["missingness"]["collapse_audit"]["n_repeat_runs"], 2)

    def test_one_judge_can_never_auto_pass_as_five_judge_panel(self):
        rows = [pointwise(score=99)]
        result = ap.aggregate(rows, {"packet_id": "p", "packet_sha256": "x"})
        self.assertNotEqual(result["pointwise"]["cand-a"]["deployment_status"], "AUTO_PASS_FOR_BENCHMARK")
        self.assertFalse(result["judge_scope"]["formal_panel"])

    def test_five_distinct_judges_can_reach_benchmark_pass(self):
        rows = [pointwise(judge=f"j{index}", score=99) for index in range(1, 6)]
        result = ap.aggregate(rows, {"packet_id": "p", "packet_sha256": "x"})
        self.assertEqual(result["pointwise"]["cand-a"]["deployment_status"], "AUTO_PASS_FOR_BENCHMARK")
        self.assertEqual(result["pointwise"]["cand-a"]["icc_2_1_2_k"]["n_judges"], 5)

    def test_source_bootstrap_reports_three_cluster_warning(self):
        result = ap.source_cluster_bootstrap({"s1": [90], "s2": [91], "s3": [92]}, replicates=40)
        self.assertEqual(result["n_source_clusters"], 3)
        self.assertEqual(result["stability"], "unstable_small_source_scope")
        self.assertIn("3 source clusters", result["warning"])

    def test_abba_maps_winner_to_canonical_id_and_excludes_position_sensitive(self):
        h = hashlib.sha256(b"source:s1").hexdigest()
        rows = []
        for judge in ("j1", "j2"):
            rows.append(normalized({"track": "pairwise", "item_id": "i1", "source_id": "s1", "source_text_sha256": h, "pair_id": "p1", "judge_id": judge, "run_id": "r1", "orientation": "AB", "candidate_a": "cand-a", "candidate_b": "cand-b", "winner": "A"}))
            rows.append(normalized({"track": "pairwise", "item_id": "i1", "source_id": "s1", "source_text_sha256": h, "pair_id": "p1", "judge_id": judge, "run_id": "r1", "orientation": "BA", "candidate_a": "cand-b", "candidate_b": "cand-a", "winner": "B"}))
        # j3 always picks the first displayed option, so it is position-sensitive.
        rows.append(normalized({"track": "pairwise", "item_id": "i1", "source_id": "s1", "source_text_sha256": h, "pair_id": "p1", "judge_id": "j3", "run_id": "r1", "orientation": "AB", "candidate_a": "cand-a", "candidate_b": "cand-b", "winner": "A"}))
        rows.append(normalized({"track": "pairwise", "item_id": "i1", "source_id": "s1", "source_text_sha256": h, "pair_id": "p1", "judge_id": "j3", "run_id": "r1", "orientation": "BA", "candidate_a": "cand-b", "candidate_b": "cand-a", "winner": "A"}))
        result = ap.pairwise_abba_copeland(rows)
        self.assertEqual(result["swap_counts"]["consistent"], 2)
        self.assertEqual(result["swap_counts"]["position_sensitive"], 1)
        self.assertEqual(result["pair_outcomes"][0]["winner"], "cand-a")
        self.assertEqual(result["ranking"][0], "cand-a")

    def test_disconnected_pairwise_graph_with_unseen_candidate_has_no_global_copeland(self):
        rows = []
        for pair_id, left, right in (("p1", "cand-a", "cand-b"), ("p2", "cand-c", "cand-d")):
            h = hashlib.sha256(f"source:{pair_id}".encode()).hexdigest()
            for orientation, candidate_a, candidate_b, winner in (
                ("AB", left, right, "A"),
                ("BA", right, left, "B"),
            ):
                rows.append(normalized({
                    "track": "pairwise", "item_id": "i1", "source_id": "s1",
                    "source_text_sha256": h, "pair_id": pair_id, "judge_id": "j1",
                    "run_id": "r1", "orientation": orientation,
                    "candidate_a": candidate_a, "candidate_b": candidate_b,
                    "winner": winner,
                }))
        result = ap.pairwise_abba_copeland(rows, ["cand-a", "cand-b", "cand-c", "cand-d", "cand-e"])
        self.assertFalse(result["coverage"]["global_ranking_available"])
        self.assertIsNone(result["ranking"])
        self.assertEqual(result["coverage"]["n_connected_components"], 3)
        self.assertEqual({tuple(component) for component in result["coverage"]["connected_components"]}, {("cand-a", "cand-b"), ("cand-c", "cand-d"), ("cand-e",)})
        self.assertEqual(len(result["component_rankings"]), 3)
        self.assertEqual(result["pair_outcomes"][0]["candidates"], ["cand-a", "cand-b"])

    def test_validate_import_binds_packet_and_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "judge_packet.json"
            source_hash = hashlib.sha256(b"English source").hexdigest()
            packet = {
                "packet_id": "pilot-p",
                "items": [{"item_id": "i1", "source_id": "s1", "source_text_sha256": source_hash, "candidates": [{"candidate_id": "cand-a"}]}],
            }
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            bundle = {
                "packet_id": "pilot-p",
                "packet_sha256": ap.sha256_file(packet_path),
                "judge": {"judge_id": "j1", "family": "family-a", "ui_model_version": "user-filled"},
                "pointwise": [pointwise(source="s1", source_hash=source_hash, candidate="cand-a", judge="j1")],
            }
            input_path = root / "results.json"
            input_path.write_text(json.dumps(bundle), encoding="utf-8")
            audit = validate_import(packet_path, input_path)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["records"]["n_judges"], 1)
            bundle["pointwise"][0]["source_text_sha256"] = hashlib.sha256(b"wrong").hexdigest()
            input_path.write_text(json.dumps(bundle), encoding="utf-8")
            bad = validate_import(packet_path, input_path)
            self.assertEqual(bad["status"], "FAIL")
            self.assertTrue(any("does not match canonical packet" in error for error in bad["errors"]))

    def test_validate_import_marks_repeat_as_warning_not_extra_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hash = hashlib.sha256(b"English source").hexdigest()
            packet_path = root / "judge_packet.json"
            packet_path.write_text(json.dumps({"items": [{"item_id": "i1", "source_text_sha256": source_hash, "candidates": [{"candidate_id": "cand-a"}]}]}), encoding="utf-8")
            row_a = pointwise(source="i1", source_hash=source_hash, judge="j1", run="r1")
            row_b = pointwise(source="i1", source_hash=source_hash, judge="j1", run="r2")
            bundle = {"packet_sha256": ap.sha256_file(packet_path), "judge_id": "j1", "records": [row_a, row_b]}
            input_path = root / "results.json"
            input_path.write_text(json.dumps(bundle), encoding="utf-8")
            audit = validate_import(packet_path, input_path)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["duplicates"]["logical_repeat_rows"], 1)
            self.assertEqual(audit["records"]["n_judges"], 1)

    def test_validate_import_checks_nested_runner_source_and_target_without_overriding_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "English source"
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            packet_path = root / "judge_packet.json"
            packet_path.write_text(json.dumps({
                "packet_id": "pilot-p",
                "pointwise": {"items": [{
                    "item_id": "i1",
                    "source_label": "s1",
                    "source_en": source,
                    "candidates": [{"blind_id": "cand-a", "target_zh_tw": "正確目標"}],
                }]},
            }), encoding="utf-8")
            row = {
                "status": "ok",
                "track": "pointwise",
                "item_id": "i1",
                "source_id": "i1",
                "source_text_sha256": source_hash,
                "judge_id": "j1",
                "candidate_id": "cand-a",
                "item": {"item_id": "i1", "source_en": source},
                "candidate": {"blind_id": "cand-a", "raw_translation_zh_tw": "正確目標"},
                "pointwise": {
                    "score_0_100": 95,
                    "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 15},
                    "critical": False,
                    "errors": [],
                },
            }
            bundle = {"packet_id": "pilot-p", "packet_sha256": ap.sha256_file(packet_path), "judge_id": "j1", "records": [row]}
            input_path = root / "results.json"
            input_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(validate_import(packet_path, input_path)["status"], "PASS")
            row["candidate"]["raw_translation_zh_tw"] = "被替換的目標"
            input_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            bad_target = validate_import(packet_path, input_path)
            self.assertEqual(bad_target["status"], "FAIL")
            self.assertEqual(bad_target["track_eligibility"]["pointwise"]["status"], "quarantined")
            self.assertTrue(any("target text does not match canonical" in error for error in bad_target["errors"]))
            row["candidate"]["raw_translation_zh_tw"] = "正確目標"
            row["pointwise"]["errors"] = [{"target_exact_quote": "不存在的目標引文"}]
            input_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            bad_quote = validate_import(packet_path, input_path)
            self.assertEqual(bad_quote["status"], "FAIL")
            self.assertTrue(any("quote validation failed" in error for error in bad_quote["errors"]))
            row["pointwise"]["errors"] = []
            row["item"]["source_en"] = "caller supplied a different source"
            input_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            bad_source = validate_import(packet_path, input_path)
            self.assertEqual(bad_source["status"], "FAIL")
            self.assertTrue(any("does not match supplied source text" in error for error in bad_source["errors"]))

    def test_qualification_judgments_array_is_flattened_per_variant(self):
        source_hash = hashlib.sha256(b"The source").hexdigest()
        metadata, rows = ap._iter_track_rows(
            {
                "track": "qualification",
                "records": [
                    {
                        "qualification_id": "q-01",
                        "source_id": "q-01",
                        "source_text_sha256": source_hash,
                        "judgments": [
                            {"blind_id": "qual-A", "score_0_100": 90, "critical": False, "errors": []},
                            {"blind_id": "qual-B", "score_0_100": 50, "critical": True, "errors": []},
                        ],
                    }
                ],
            }
        )
        normalized_rows = [ap._normalize_row(row, metadata) for row in rows]
        self.assertEqual(len(normalized_rows), 2)
        self.assertEqual({row["_track"] for row in normalized_rows}, {"qualification"})
        self.assertEqual({row["_candidate_id"] for row in normalized_rows}, {"qual-A", "qual-B"})
        self.assertEqual({row["_item_id"] for row in normalized_rows}, {"q-01"})

    def test_nested_runner_qualification_envelope_is_flattened_per_variant(self):
        source_hash = hashlib.sha256(b"The source").hexdigest()
        metadata, rows = ap._iter_track_rows(
            {
                "judge": {"judge_id": "j-live", "family": "qwen"},
                "records": [{
                    "track": "qualification",
                    "item_id": "q-01",
                    "source_id": "q-01",
                    "source_text_sha256": source_hash,
                    "status": "ok",
                    "qualification": {"judgments": [
                        {"blind_id": "qual-A", "score_0_100": 95, "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 15}, "critical": False, "errors": []},
                        {"blind_id": "qual-B", "score_0_100": 50, "subscores": {"terminology_locale": 20, "semantic_safety": 10, "fluency": 10, "completeness_format": 10}, "critical": True, "errors": []},
                    ]},
                }],
            }
        )
        normalized_rows = [ap._normalize_row(row, metadata) for row in rows]
        self.assertEqual(len(normalized_rows), 2)
        self.assertEqual({row["_track"] for row in normalized_rows}, {"qualification"})
        self.assertEqual({row["_candidate_id"] for row in normalized_rows}, {"qual-A", "qual-B"})
        self.assertEqual({row["_item_id"] for row in normalized_rows}, {"q-01"})
        self.assertEqual({row["_judge_id"] for row in normalized_rows}, {"j-live"})

    def test_invalid_qualification_scalar_is_excluded_but_evidence_fields_remain(self):
        row = ap._normalize_row({
            "track": "qualification", "status": "ok", "item_id": "q-01",
            "source_id": "q-01", "source_text_sha256": hashlib.sha256(b"source").hexdigest(),
            "candidate_id": "qual-A", "judge_id": "j1", "family": "fam-a",
            "score_0_100": 10,
            "subscores": {"terminology_locale": 25, "semantic_safety": 0, "fluency": 18, "completeness_format": 5},
            "critical": True,
            "errors": [{"severity": "critical", "type": "accuracy/negation", "target_exact_quote": "target"}],
        }, {})
        self.assertEqual(row["score_0_100"], 10)
        self.assertIsNone(row["_score"])
        self.assertTrue(row["_critical"])
        self.assertEqual(row["_qualification_score_audit"]["status"], "invalid")
        aggregate = ap.aggregate([row], {"packet_id": "pilot-p"})
        self.assertEqual(aggregate["qualification"]["score_audit"]["invalid"], 1)
        self.assertEqual(aggregate["qualification"]["score_audit"]["scalar_score_aggregation"]["n_excluded"], 1)

    def test_validate_import_audits_all_flattened_qualification_scalar_sums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "The source"
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            packet_path = root / "judge_packet.json"
            packet_path.write_text(json.dumps({
                "packet_id": "pilot-p",
                "qualification": {"items": [{
                    "qualification_id": "q-01",
                    "source_en": source,
                    "variants": [
                        {"blind_id": "qual-A", "target_zh_tw": "乾淨"},
                        {"blind_id": "qual-B", "target_zh_tw": "錯誤"},
                    ],
                }]},
            }), encoding="utf-8")
            bundle = {
                "packet_id": "pilot-p",
                "packet_sha256": ap.sha256_file(packet_path),
                "judge_id": "j1",
                "records": [{
                    "track": "qualification",
                    "item_id": "q-01",
                    "source_id": "q-01",
                    "source_text_sha256": source_hash,
                    "status": "ok",
                    "qualification": {"judgments": [
                        {"blind_id": "qual-A", "abstain": False, "score_0_100": 10, "subscores": {"terminology_locale": 25, "semantic_safety": 0, "fluency": 18, "completeness_format": 5}, "critical": True, "errors": []},
                        {"blind_id": "qual-B", "abstain": False, "score_0_100": 100, "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20}, "critical": False, "errors": []},
                    ]},
                }],
            }
            input_path = root / "results.json"
            input_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            audit = validate_import(packet_path, input_path)
            # A malformed scalar sum is a warning/partial-validity condition;
            # the row's critical/error evidence remains importable.
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["records"]["tracks"], {"qualification": 2})
            self.assertEqual(audit["records"]["qualification_score_audit"]["valid"], 1)
            self.assertEqual(audit["records"]["qualification_score_audit"]["invalid"], 1)
            self.assertEqual(audit["score_shape"]["by_track"]["qualification"], {"checked": 2, "valid": 1, "invalid": 1})
            self.assertEqual(audit["score_shape"]["status"], "PASS_WITH_WARNINGS")
            self.assertEqual(audit["track_eligibility"]["qualification"]["status"], "partial")
            self.assertFalse(audit["track_eligibility"]["qualification"]["candidate_eligible_for_formal_scalar_qualification"])
            self.assertEqual(audit["track_eligibility"]["qualification"]["evidence_critical_status"], "eligible_as_diagnostic_only")
            self.assertEqual(audit["track_eligibility"]["qualification"]["formal_result_status"], "not_evaluated_without_private_key")
            self.assertTrue(any("qualification scalar score audit invalid" in warning for warning in audit["warnings"]))

    def test_missing_and_failed_rows_are_explicit(self):
        h = hashlib.sha256(b"source:s1").hexdigest()
        rows = [
            pointwise(judge="j1", status="missing", score=None),
            pointwise(judge="j2", status="failed", score=None),
            pointwise(judge="j3", score=90),
        ]
        result = ap.aggregate(rows, {"packet_id": "p", "packet_sha256": "x"})
        missing = result["missingness"]
        self.assertEqual(missing["raw_status_counts"]["missing"], 1)
        self.assertEqual(missing["raw_status_counts"]["failed"], 1)
        self.assertEqual(result["pointwise"]["cand-a"]["missingness"]["n_failed_rows"], 1)
        self.assertEqual(result["pointwise"]["cand-a"]["deployment_status"], "INSUFFICIENT_PANEL")

    def test_family_balanced_mean_and_leave_one_out_are_descriptive(self):
        rows = []
        for judge, family, score in (("j1", "fam-a", 100), ("j2", "fam-a", 80), ("j3", "fam-b", 60)):
            rows.append(pointwise(judge=judge, score=score))
            rows[-1]["_family"] = family
        balanced = ap.family_balanced_mean(rows)
        # fam-a contributes its judge mean (90) once, fam-b contributes 60 once.
        self.assertAlmostEqual(balanced["estimate"], 75.0)
        sensitivity = ap.family_leave_one_out(rows)
        self.assertEqual(sensitivity["status"], "ok")
        self.assertIn("fam-a", sensitivity["leave_one_family_out"])
        result = ap.aggregate(rows, {"packet_id": "p", "packet_sha256": "x"})
        self.assertEqual(result["judge_scope"]["observed_families"], ["fam-a", "fam-b"])
        self.assertEqual(result["family_sensitivity"]["hierarchical_crossed_model"]["status"], "not_fit")

    def test_merge_independent_judge_bundles_preserves_judge_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for judge, family in (("j1", "fam-a"), ("j2", "fam-b")):
                bundle = {
                    "packet_id": "p",
                    "packet_sha256": "a" * 64,
                    "judge_id": judge,
                    "family": family,
                    "records": [pointwise(judge=judge)],
                }
                path = root / f"{judge}.json"
                path.write_text(json.dumps(bundle), encoding="utf-8")
                paths.append(path)
            metadata, rows = ap.load_bundles(paths)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["_family"] for row in rows}, {"fam-a", "fam-b"})
            self.assertEqual(len(metadata["judges"]), 2)

    def test_nested_judge_envelope_is_inherited_by_section_records(self):
        # The live runner stores the stable judge/family identity under
        # top-level ``judge`` while qualification/pointwise/pairwise records
        # are section children.  Validate the actual normalization path.
        metadata, raw_rows = ap._iter_track_rows({
            "judge": {"judge_id": "j-live", "family": "fam-live"},
            "pointwise": [{
                "item_id": "i1", "source_id": "s1",
                "source_text_sha256": hashlib.sha256(b"source").hexdigest(),
                "candidate_id": "cand-a", "status": "missing", "reason": "test",
            }],
        })
        normalized_rows = [ap._normalize_row(row, metadata) for row in raw_rows]
        self.assertEqual(normalized_rows[0]["_judge_id"], "j-live")
        self.assertEqual(normalized_rows[0]["_family"], "fam-live")

    def test_gemini_renderer_uses_declared_frozen_instruction_hash(self):
        packet = ap.read_json(HERE / "judge_packet.json")
        self.assertEqual(mgp.file_sha256(mgp.INSTRUCTIONS), packet["judge_instructions_sha256"])

    def test_qualification_mutation_flag_proxy_is_separate_from_critical_severity_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hash = hashlib.sha256(b"qualification source").hexdigest()
            bundle = {
                "packet_id": "pilot-p",
                "packet_sha256": "a" * 64,
                "judge_id": "j1",
                "family": "fam-a",
                "records": [
                    {"track": "qualification", "status": "ok", "item_id": "q-01", "candidate_id": "qual-A", "source_text_sha256": source_hash, "judge_id": "j1", "critical": False, "errors": []},
                    {"track": "qualification", "status": "ok", "item_id": "q-01", "candidate_id": "qual-B", "source_text_sha256": source_hash, "judge_id": "j1", "critical": False, "errors": [{"severity": "major", "type": "accuracy/negation"}]},
                    {"track": "qualification", "status": "ok", "item_id": "q-02", "candidate_id": "qual-A", "source_text_sha256": source_hash, "judge_id": "j1", "critical": False, "errors": []},
                    {"track": "qualification", "status": "ok", "item_id": "q-02", "candidate_id": "qual-B", "source_text_sha256": source_hash, "judge_id": "j1", "critical": False, "errors": []},
                ],
            }
            key = {
                "packet_id": "pilot-p",
                "qualification": {
                    "q-01": {"clean_blind_id": "qual-A", "mutated_blind_id": "qual-B", "expected_behavior": "detect_error", "expected_error_type": "accuracy/negation", "expected_severity": "critical"},
                    "q-02": {"clean_blind_id": "qual-A", "mutated_blind_id": "qual-B", "expected_behavior": "invariance", "expected_error_type": "none", "expected_severity": "none"},
                },
            }
            bundle_path = root / "bundle.json"
            key_path = root / "answer_key.private.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            key_path.write_text(json.dumps(key), encoding="utf-8")
            result = qs.evaluate([bundle_path], key_path)
            report = result["per_judge"]["j1"]
            self.assertEqual(report["mutation_flag_rate_proxy"]["estimate"], 1.0)
            self.assertIsNone(report["mutation_evidence_supported_rate"]["estimate"])
            self.assertEqual(report["coverage"]["available_variants"], 4)
            self.assertEqual(report["coverage"]["missing_variants"], 0)
            self.assertEqual(result["evidence_context"]["status"], "unavailable")
            self.assertEqual(report["critical_severity_recall"]["union_top_level_or_error_severity"]["estimate"], 0.0)
            self.assertEqual(report["invariance_accuracy"]["estimate"], 1.0)

    def test_qualification_evidence_requires_programmed_mutation_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "The patient denies suicidal thoughts."
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            packet = {
                "packet_id": "pilot-p",
                "qualification": {"items": [{
                    "qualification_id": "q-01",
                    "source_en": source,
                    "variants": [
                        {"blind_id": "qual-A", "target_zh_tw": "病人有自殺想法。"},
                        {"blind_id": "qual-B", "target_zh_tw": "病人否認有自殺想法。"},
                    ],
                }]},
            }
            bundle = {
                "packet_id": "pilot-p",
                "packet_sha256": "a" * 64,
                "judge_id": "j1",
                "records": [{
                    "track": "qualification", "status": "ok", "item_id": "q-01",
                    "candidate_id": "qual-A", "source_text_sha256": source_hash,
                    "judge_id": "j1", "critical": False,
                    # This is an unrelated style flag: it must not become
                    # evidence for the keyed negation mutation.
                    "errors": [{"severity": "minor", "type": "style", "target_exact_quote": "病人", "source_span": "The patient"}],
                }, {
                    "track": "qualification", "status": "ok", "item_id": "q-01",
                    "candidate_id": "qual-B", "source_text_sha256": source_hash,
                    "judge_id": "j1", "critical": False, "errors": [],
                }],
            }
            packet_path = root / "judge_packet.json"
            bundle_path = root / "bundle.json"
            key_path = root / "answer_key.private.json"
            packet_path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            key_path.write_text(json.dumps({
                "packet_id": "pilot-p",
                "qualification": {"q-01": {
                    "clean_blind_id": "qual-B", "mutated_blind_id": "qual-A",
                    "expected_behavior": "detect_error", "expected_error_type": "accuracy/negation",
                    "expected_severity": "critical",
                }},
            }), encoding="utf-8")
            result = qs.evaluate([bundle_path], key_path, packet_path)
            report = result["per_judge"]["j1"]
            self.assertEqual(report["mutation_flag_rate_proxy"]["estimate"], 1.0)
            self.assertEqual(report["mutation_evidence_supported_rate"]["estimate"], 0.0)
            self.assertEqual(report["item_results"][0]["mutation_evidence_status"], "unaligned_error_evidence")

            bundle["records"][0]["errors"] = [{
                "severity": "critical", "type": "accuracy/negation",
                "target_exact_quote": "有自殺想法",
                "source_evidence": "denies suicidal thoughts — the denial is reversed",
                "source_span": "denies suicidal thoughts",
            }]
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            supported = qs.evaluate([bundle_path], key_path, packet_path)["per_judge"]["j1"]
            self.assertEqual(supported["mutation_evidence_supported_rate"]["estimate"], 1.0)

    def test_qualification_small_denominator_is_exposed_by_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hash = hashlib.sha256(b"source").hexdigest()
            bundle = {
                "packet_id": "pilot-p", "packet_sha256": "a" * 64,
                "judge_id": "j1", "records": [{
                    "track": "qualification", "status": "ok", "item_id": "q-01",
                    "candidate_id": "qual-B", "source_text_sha256": source_hash,
                    "judge_id": "j1", "critical": True, "errors": [{"severity": "critical", "type": "accuracy/negation"}],
                }],
            }
            key = {
                "packet_id": "pilot-p",
                "qualification": {
                    "q-01": {"clean_blind_id": "qual-A", "mutated_blind_id": "qual-B", "expected_behavior": "detect_error", "expected_error_type": "accuracy/negation", "expected_severity": "critical"},
                    "q-02": {"clean_blind_id": "qual-A", "mutated_blind_id": "qual-B", "expected_behavior": "detect_error", "expected_error_type": "accuracy/number", "expected_severity": "critical"},
                },
            }
            bundle_path = root / "bundle.json"
            key_path = root / "answer_key.private.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            key_path.write_text(json.dumps(key), encoding="utf-8")
            report = qs.evaluate([bundle_path], key_path)["per_judge"]["j1"]
            self.assertEqual(report["mutation_flag_rate_proxy"]["estimate"], 1.0)
            self.assertEqual(report["mutation_flag_rate_proxy"]["denominator"], 1)
            self.assertEqual(report["coverage"]["status"], "incomplete")
            self.assertEqual(report["coverage"]["missing_variants"], 3)
            self.assertEqual(report["coverage"]["detect_error_mutated_coverage"]["denominator"], 2)
            self.assertEqual(report["coverage"]["detect_error_mutated_coverage"]["numerator"], 1)


if __name__ == "__main__":
    unittest.main()
