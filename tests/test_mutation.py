"""Tests for msa_ga/mutation.py (Phase 5, gap-block swap).  Run from the project root:
    python3 -m unittest tests.test_mutation -v

Conventions these tests assume:
  * A gap run is a maximal contiguous stretch of '-' in one row, stored as
    (row, start, end) with end exclusive.
  * gap_runs(msa) returns every run, row-major then left-to-right. Leading and
    trailing gaps count.
  * grow_block(msa, row, col) grows the SAGA gap block of the seed cell
    msa[row][col]: the run containing that cell, plus every other run whose
    columns overlap, closed under overlap. Returns a list of (row, start, end).
    Raises ValueError if the seed is out of bounds or is not a gap.
  * swap_runs(msa, run_a, run_b) slides two equal-length, non-overlapping runs
    on different rows to each other's columns (gaps move inside each chosen
    row; residues never change sequence). Other rows are unchanged. The parent
    is not modified. Raises ValueError if the runs overlap, have different
    lengths, share a row, or are out of range.
  * mutate(msa, rng) picks two eligible runs (different rows, same length,
    disjoint columns) with rng and applies swap_runs, then clean_alignment.
    If fewer than two eligible runs exist, or the result is invalid, it returns
    a new list that equals the parent. The parent is never modified.
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

from msa_ga import initialization as ini
from msa_ga import mutation as mut
from msa_ga.io_utils import read_fasta
from msa_ga.validity import clean_alignment, is_valid_msa

PROT_SEQS = read_fasta(os.path.join(ROOT, "data", "cytochromes.fa"))[1]
DNA_SEQS = ["ATATCCG", "TCCG", "ATGTACTG", "ATGTCTG"]

# Two equal-length, non-overlapping runs on different rows.
SWAP_PARENT = ["AA--CCDD", "AACCDD--"]
SWAP_CHILD = ["AACCDD--", "AA--CCDD"]
SWAP_SEQS = ["AACCDD", "AACCDD"]
RUN_A = (0, 2, 4)
RUN_B = (1, 6, 8)

# Seed (0, 2) overlaps row 2; the trailing run on row 1 does not.
BLOCK_MSA = [
    "AA--CCDD",
    "AACCDD--",
    "AA-CCD-D",
]


def sequences_of(msa):
    return [row.replace("-", "") for row in msa]


class TestGapRuns(unittest.TestCase):
    def test_no_gaps(self):
        self.assertEqual(mut.gap_runs(["AC", "GT"]), [])

    def test_empty_msa(self):
        self.assertEqual(mut.gap_runs([]), [])

    def test_single_internal_run(self):
        self.assertEqual(mut.gap_runs(["A--C"]), [(0, 1, 3)])

    def test_leading_and_trailing_gaps_count(self):
        self.assertEqual(mut.gap_runs(["--AC--"]), [(0, 0, 2), (0, 4, 6)])

    def test_row_major_left_to_right(self):
        self.assertEqual(
            mut.gap_runs(BLOCK_MSA),
            [(0, 2, 4), (1, 6, 8), (2, 2, 3), (2, 6, 7)],
        )

    def test_entire_row_of_gaps(self):
        self.assertEqual(mut.gap_runs(["----", "A--C"]), [(0, 0, 4), (1, 1, 3)])


class TestGrowBlock(unittest.TestCase):
    def _block(self, msa, row, col):
        return sorted(mut.grow_block(msa, row, col))

    def test_isolated_run_is_its_own_block(self):
        self.assertEqual(self._block(["A--C", "ACGT"], 0, 1), [(0, 1, 3)])

    def test_overlapping_run_on_another_row_is_included(self):
        self.assertEqual(self._block(BLOCK_MSA, 0, 2), [(0, 2, 4), (2, 2, 3)])

    def test_non_overlapping_run_is_excluded(self):
        block = self._block(BLOCK_MSA, 0, 2)
        self.assertNotIn((1, 6, 8), block)

    def test_seed_anywhere_in_the_run_gives_the_same_block(self):
        self.assertEqual(self._block(BLOCK_MSA, 0, 2), self._block(BLOCK_MSA, 0, 3))

    def test_overlap_is_closed_transitively(self):
        # row0 [1,3) overlaps row1 [2,4); row1 overlaps row2 [3,5)
        msa = [
            "A--AAA",
            "AA--AA",
            "AAA--A",
        ]
        self.assertEqual(
            self._block(msa, 0, 1),
            [(0, 1, 3), (1, 2, 4), (2, 3, 5)],
        )

    def test_seed_on_residue_raises(self):
        with self.assertRaises(ValueError):
            mut.grow_block(BLOCK_MSA, 0, 0)

    def test_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            mut.grow_block(BLOCK_MSA, -1, 0)
        with self.assertRaises(ValueError):
            mut.grow_block(BLOCK_MSA, 0, 99)
        with self.assertRaises(ValueError):
            mut.grow_block(BLOCK_MSA, 9, 0)


class TestSwapRuns(unittest.TestCase):
    def test_worked_example(self):
        self.assertEqual(mut.swap_runs(SWAP_PARENT, RUN_A, RUN_B), SWAP_CHILD)

    def test_order_of_runs_does_not_matter(self):
        self.assertEqual(mut.swap_runs(SWAP_PARENT, RUN_B, RUN_A), SWAP_CHILD)

    def test_residue_order_preserved(self):
        child = mut.swap_runs(SWAP_PARENT, RUN_A, RUN_B)
        self.assertEqual(sequences_of(child), SWAP_SEQS)

    def test_other_rows_unchanged(self):
        parent = ["AA--CCDD", "AACCDD--", "AABBCCDD"]
        child = mut.swap_runs(parent, RUN_A, RUN_B)
        self.assertEqual(child[2], "AABBCCDD")
        self.assertEqual(child[:2], SWAP_CHILD)

    def test_does_not_modify_parent(self):
        parent = list(SWAP_PARENT)
        mut.swap_runs(parent, RUN_A, RUN_B)
        self.assertEqual(parent, SWAP_PARENT)

    def test_returns_a_new_list(self):
        parent = list(SWAP_PARENT)
        child = mut.swap_runs(parent, RUN_A, RUN_B)
        child[0] = "X"
        self.assertEqual(parent[0], "AA--CCDD")
        self.assertIsNot(child, parent)

    def test_overlapping_runs_raise(self):
        with self.assertRaises(ValueError):
            mut.swap_runs(["A--C", "-A-C"], (0, 1, 3), (1, 0, 1))

    def test_different_lengths_raise(self):
        with self.assertRaises(ValueError):
            mut.swap_runs(["A---C", "A-C--"], (0, 1, 4), (1, 1, 2))

    def test_same_row_raises(self):
        with self.assertRaises(ValueError):
            mut.swap_runs(["A--C--"], (0, 1, 3), (0, 4, 6))


class TestMutate(unittest.TestCase):
    def test_single_eligible_pair_is_swapped(self):
        child = mut.mutate(SWAP_PARENT, random.Random(0))
        self.assertEqual(child, clean_alignment(SWAP_CHILD))

    def test_terminal_gaps_are_eligible(self):
        parent = ["--AA", "AA--"]
        child = mut.mutate(parent, random.Random(0))
        self.assertEqual(child, ["AA--", "--AA"])

    def test_no_gaps_returns_equal_copy(self):
        parent = ["ACGT", "TGCA"]
        child = mut.mutate(parent, random.Random(1))
        self.assertEqual(child, parent)
        self.assertIsNot(child, parent)

    def test_single_row_returns_equal_copy(self):
        parent = ["A--C--GT"]
        child = mut.mutate(parent, random.Random(1))
        self.assertEqual(child, parent)
        self.assertIsNot(child, parent)

    def test_overlapping_equal_runs_are_not_swapped(self):
        parent = ["AA--CC", "AA--CC"]
        child = mut.mutate(parent, random.Random(1))
        self.assertEqual(child, parent)

    def test_does_not_modify_parent(self):
        parent = list(SWAP_PARENT)
        mut.mutate(parent, random.Random(0))
        self.assertEqual(parent, SWAP_PARENT)

    def test_same_seed_same_child(self):
        parent = ini.random_msa(DNA_SEQS, 4, random.Random(1))
        a = mut.mutate(parent, random.Random(11))
        b = mut.mutate(parent, random.Random(11))
        self.assertEqual(a, b)

    def test_does_not_use_global_random(self):
        parent = ini.random_msa(DNA_SEQS, 4, random.Random(2))
        random.seed(0)
        a = mut.mutate(parent, random.Random(5))
        random.seed(999)
        b = mut.mutate(parent, random.Random(5))
        self.assertEqual(a, b)

    def test_result_is_valid_on_random_dna(self):
        rng = random.Random(3)
        for _ in range(40):
            parent = ini.random_msa(DNA_SEQS, 5, rng)
            child = mut.mutate(parent, random.Random(rng.randint(0, 10**6)))
            self.assertTrue(is_valid_msa(child, DNA_SEQS), msg=f"{parent} -> {child}")

    def test_result_is_valid_on_random_protein(self):
        rng = random.Random(4)
        for _ in range(15):
            parent = ini.random_msa(PROT_SEQS, 8, rng)
            child = mut.mutate(parent, random.Random(rng.randint(0, 10**6)))
            self.assertTrue(is_valid_msa(child, PROT_SEQS), msg=f"len={len(parent[0])}")

    def test_can_change_the_alignment_when_a_swap_exists(self):
        children = {tuple(mut.mutate(SWAP_PARENT, random.Random(s))) for s in range(10)}
        self.assertIn(tuple(SWAP_CHILD), children)


if __name__ == "__main__":
    unittest.main()
