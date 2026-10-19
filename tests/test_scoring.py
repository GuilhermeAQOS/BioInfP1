"""Tests for the MSA sum-of-pairs score (msa_ga.scoring.score_msa).

score_msa must evaluate the given alignment as-is: every unordered pair of
rows, column by column, using the linear gap rule from Lab 3 (score_pos).
It must not re-align the rows with Needleman-Wunsch.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

from substmatrix import SubstMatrix

from msa_ga.io_utils import load_substitution_matrix
from msa_ga.scoring import score_msa


def dna_matrix(match: int = 1, mismatch: int = -1) -> SubstMatrix:
    sm = SubstMatrix()
    sm.create_submat(match, mismatch, "ACGT")
    return sm


class TestScoreMsaDNA(unittest.TestCase):
    def setUp(self) -> None:
        self.sm = dna_matrix()
        self.gap = -2

    def test_two_identical_rows(self) -> None:
        # ACGT vs ACGT: 4 matches
        self.assertEqual(score_msa(["ACGT", "ACGT"], self.sm, self.gap), 4)

    def test_one_mismatch(self) -> None:
        # ACGT vs ACTT: match, match, mismatch, match -> 1+1-1+1 = 2
        self.assertEqual(score_msa(["ACGT", "ACTT"], self.sm, self.gap), 2)

    def test_linear_gap_penalty(self) -> None:
        # AT-G vs A-TG: A/A=1, T/-=-2, -/T=-2, G/G=1 -> -2
        self.assertEqual(score_msa(["AT-G", "A-TG"], self.sm, self.gap), -2)

    def test_both_gaps_in_a_column_score_zero(self) -> None:
        # A-T vs A-T: A/A=1, -/- =0, T/T=1 -> 2
        self.assertEqual(score_msa(["A-T", "A-T"], self.sm, self.gap), 2)

    def test_single_row_has_no_pairs(self) -> None:
        self.assertEqual(score_msa(["ACGT"], self.sm, self.gap), 0)

    def test_empty_msa(self) -> None:
        self.assertEqual(score_msa([], self.sm, self.gap), 0)

    def test_three_rows_sums_all_unordered_pairs(self) -> None:
        # The running example from the scoring discussion.
        # 0 vs 1: AT-G / A-TG -> -2
        # 0 vs 2: AT-G / ATTG ->  1
        # 1 vs 2: A-TG / ATTG ->  1
        # SP = 0
        msa = ["AT-G", "A-TG", "ATTG"]
        self.assertEqual(score_msa(msa, self.sm, self.gap), 0)

    def test_identical_triplet_counts_three_pairs(self) -> None:
        # 3 unordered pairs, each 3 matches -> 9
        self.assertEqual(score_msa(["AAA", "AAA", "AAA"], self.sm, self.gap), 9)

    def test_scores_the_given_alignment_not_an_optimal_one(self) -> None:
        # Same residues (AA vs AA). The gapped placement is worse than
        # aligning them without extra gaps; the scorer must keep the given
        # columns, not run Needleman-Wunsch.
        # A-A vs -AA: A/-=-2, -/A=-2, A/A=1 -> -3
        # AA  vs  AA: 1+1 = 2
        self.assertEqual(score_msa(["A-A", "-AA"], self.sm, self.gap), -3)
        self.assertEqual(score_msa(["AA", "AA"], self.sm, self.gap), 2)

    def test_match_mismatch_values_are_taken_from_sm(self) -> None:
        sm = dna_matrix(match=5, mismatch=-3)
        # AG vs AT: match, mismatch -> 5-3 = 2
        self.assertEqual(score_msa(["AG", "AT"], sm, self.gap), 2)


class TestScoreMsaProtein(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sm = load_substitution_matrix(os.path.join(ROOT, "data", "blosum62.mat"))
        cls.gap = -4

    def test_identical_alanines(self) -> None:
        # BLOSUM62 AA = 4; two columns, one pair -> 8
        self.assertEqual(self.sm.sm["AA"], 4)
        self.assertEqual(score_msa(["AA", "AA"], self.sm, self.gap), 8)

    def test_substitution_and_gap(self) -> None:
        # A-W vs AAW: AA=4, -/A=-4, WW=11 -> 11
        self.assertEqual(self.sm.sm["WW"], 11)
        self.assertEqual(score_msa(["A-W", "AAW"], self.sm, self.gap), 11)

    def test_known_mismatch(self) -> None:
        # AW vs AA: AW=-3, AA=4 -> 1
        self.assertEqual(self.sm.sm["AW"], -3)
        self.assertEqual(score_msa(["AW", "AA"], self.sm, self.gap), 1)


if __name__ == "__main__":
    unittest.main()
