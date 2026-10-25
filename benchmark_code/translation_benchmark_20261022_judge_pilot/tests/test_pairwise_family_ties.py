from __future__ import annotations

import unittest

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import aggregate_pilot as ap


def pairwise_rows(votes: list[tuple[str, str | None, str]]) -> list[dict[str, object]]:
    """Build one complete AB/BA pair for each judge's canonical vote."""

    rows: list[dict[str, object]] = []
    for judge_id, family, winner in votes:
        for orientation, candidate_a, candidate_b in (
            ("AB", "cand-a", "cand-b"),
            ("BA", "cand-b", "cand-a"),
        ):
            displayed_winner = "TIE"
            if winner == "cand-a":
                displayed_winner = "A" if orientation == "AB" else "B"
            elif winner == "cand-b":
                displayed_winner = "B" if orientation == "AB" else "A"
            rows.append({
                "_status": "ok",
                "_item_id": "item-1",
                "_judge_id": judge_id,
                "_family": family,
                "orientation": orientation,
                "pair_id": "pair-1",
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "winner": displayed_winner,
            })
    return rows


class PairwiseFamilyTieTests(unittest.TestCase):
    def test_one_tie_family_and_one_candidate_family_preserve_tie_outcome(self) -> None:
        result = ap.pairwise_abba_copeland(pairwise_rows([
            ("judge-tie", "family-tie", "TIE"),
            ("judge-a", "family-a", "cand-a"),
        ]), ["cand-a", "cand-b"])

        self.assertEqual(result["pair_outcomes"][0]["winner"], "TIE")
        balanced = result["family_balanced_pair_outcomes"][0]
        self.assertEqual(balanced["winner"], "TIE")
        self.assertEqual(balanced["family_vote_scores"], {"cand-a": 0.5, "cand-b": 0.0})
        self.assertEqual(balanced["family_tie_vote_score"], 0.5)
        self.assertEqual(balanced["family_outcome_vote_scores"], {"cand-a": 0.5, "cand-b": 0.0, "TIE": 0.5})

    def test_all_tie_families_remain_tie_and_retain_candidate_graph_keys(self) -> None:
        result = ap.pairwise_abba_copeland(pairwise_rows([
            ("judge-tie-a", "family-a", "TIE"),
            ("judge-tie-b", "family-b", "TIE"),
        ]), ["cand-a", "cand-b"])

        self.assertEqual(result["pair_outcomes"][0]["winner"], "TIE")
        balanced = result["family_balanced_pair_outcomes"][0]
        self.assertEqual(balanced["winner"], "TIE")
        self.assertEqual(balanced["family_vote_scores"], {"cand-a": 0.0, "cand-b": 0.0})
        self.assertEqual(balanced["family_tie_vote_score"], 1.0)
        self.assertEqual(balanced["family_outcome_vote_scores"], {"cand-a": 0.0, "cand-b": 0.0, "TIE": 1.0})

    def test_unequal_judges_within_family_are_family_balanced(self) -> None:
        result = ap.pairwise_abba_copeland(pairwise_rows([
            ("judge-a-1", "family-a", "cand-a"),
            ("judge-a-2", "family-a", "cand-a"),
            ("judge-b-1", "family-b", "cand-b"),
        ]), ["cand-a", "cand-b"])

        self.assertEqual(result["pair_outcomes"][0]["winner"], "cand-a")
        balanced = result["family_balanced_pair_outcomes"][0]
        self.assertEqual(balanced["winner"], "TIE")
        self.assertEqual(balanced["family_vote_scores"], {"cand-a": 0.5, "cand-b": 0.5})
        self.assertEqual(balanced["family_tie_vote_score"], 0.0)

    def test_no_tie_behavior_and_missing_family_exclusion_are_unchanged(self) -> None:
        result = ap.pairwise_abba_copeland(pairwise_rows([
            ("judge-a", "family-a", "cand-a"),
            ("judge-missing-family", None, "cand-b"),
        ]), ["cand-a", "cand-b"])

        self.assertEqual(result["pair_outcomes"][0]["winner"], "TIE")
        balanced = result["family_balanced_pair_outcomes"][0]
        self.assertEqual(balanced["winner"], "cand-a")
        self.assertEqual(balanced["n_families"], 1)
        self.assertEqual(balanced["family_vote_scores"], {"cand-a": 1.0, "cand-b": 0.0})
        self.assertEqual(balanced["family_tie_vote_score"], 0.0)

    def test_position_sensitive_pair_remains_abstention(self) -> None:
        rows = pairwise_rows([
            ("judge-position", "family-position", "cand-a"),
        ])
        rows[-1]["winner"] = "A"  # BA also picks displayed A, not canonical cand-a.
        result = ap.pairwise_abba_copeland(rows, ["cand-a", "cand-b"])

        self.assertEqual(result["swap_counts"]["position_sensitive"], 1)
        self.assertEqual(result["pair_outcomes"], [])
        self.assertEqual(result["family_balanced_pair_outcomes"], [])


if __name__ == "__main__":
    unittest.main()
