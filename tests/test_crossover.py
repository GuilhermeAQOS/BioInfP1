"""Tests for msa_ga/crossover.py (Phase 4).  Run from the project root:
    python -m unittest tests.test_crossover -v
Conventions these tests assume:
  * index_at_residue(line, 0) == 0                    (everything, including leading gaps, goes right)
  * index_at_residue raises ValueError if n < 0 or n > number of residues of the line
  * split_at / residues_to_indexes / merge raise ValueError on inconsistent arguments
  * pad_alignment(msa, "r") pads on the right (left parts), pad_alignment(msa, "l") on the left (right parts)
  * generate_offspring cuts parent1 at column col, valid range 1 <= col <= len(parent1) - 1,
    anything else raises ValueError
  * crossover(...) draws the cut from parent1's length and returns ONE valid MSA (the better offspring);
    if parent1 has no valid cut (a single column) it returns a copy of parent1
"""
import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msa_ga import crossover as cx
from msa_ga import initialization as ini
from msa_ga import scoring
from msa_ga.io_utils import build_submat, read_fasta
from msa_ga.params import Params
from msa_ga.validity import is_valid_msa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = Params(matrix=os.path.join(ROOT, "data", "blosum62.mat"))
DNA_SM = build_submat("DNA", PARAMS)
PROT_SM = build_submat("PROTEIN", PARAMS)
DNA_SEQS = read_fasta(os.path.join(ROOT, "data", "cytochromes.fa"))[1]
PROT_SEQS = read_fasta(os.path.join(ROOT, "data", "cytochromes.fa"))[1]

# Parents from the crossover tutorial
ALIGN1 = ["ATC-G-G-TT", "A-CCGC-ATC", "A-CCG--ATC"]
ALIGN2 = ["ATCGG---TT", "AC-CGCA-TC", "A---CCGATC"]
TUT_SEQS = [r.replace("-", "") for r in ALIGN1]      # the sequences both tutorial parents contain


class TestCountResidues(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(cx.count_residues("--AC--"), 2)
        self.assertEqual(cx.count_residues("ABC"), 3)
        self.assertEqual(cx.count_residues("----"), 0)
        self.assertEqual(cx.count_residues(""), 0)


class TestIndexAtResidue(unittest.TestCase):
    def test_tutorial_example(self):
        self.assertEqual(cx.index_at_residue("--AC--GT", 2), 4)

    def test_gaps_after_last_residue_go_right(self):
        self.assertEqual(cx.index_at_residue("AC-CGCA-TC", 2), 2)
        self.assertEqual(cx.index_at_residue("ATCGG---TT", 3), 3)
        self.assertEqual(cx.index_at_residue("A---CCGATC", 2), 5)

    def test_zero_residues(self):
        self.assertEqual(cx.index_at_residue("--AC--GT", 0), 0)
        self.assertEqual(cx.index_at_residue("ACGT", 0), 0)

    def test_all_residues(self):
        self.assertEqual(cx.index_at_residue("--AC--GT", 4), 8)
        self.assertEqual(cx.index_at_residue("--AC--GT--", 4), 8)   # trailing gaps stay right
        self.assertEqual(cx.index_at_residue("AC", 2), 2)

    def test_middle(self):
        self.assertEqual(cx.index_at_residue("--AC--GT", 3), 7)

    def test_too_many_or_negative_residues_raise(self):
        with self.assertRaises(ValueError):
            cx.index_at_residue("--AC", 3)
        with self.assertRaises(ValueError):
            cx.index_at_residue("--AC", -1)

    def test_property_on_random_lines(self):
        rng = random.Random(1)
        for _ in range(300):
            line = "".join(rng.choice("AC-") for _ in range(rng.randint(0, 15)))
            total = cx.count_residues(line)
            for n in range(total + 1):
                idx = cx.index_at_residue(line, n)
                self.assertEqual(cx.count_residues(line[:idx]), n, msg=f"{line!r} n={n}")
                if n == 0:
                    self.assertEqual(idx, 0)
                else:
                    self.assertNotEqual(line[idx - 1], "-", msg=f"{line!r} n={n}")


class TestSplitAt(unittest.TestCase):
    def test_tutorial_first_parent(self):
        left, right, residues = cx.split_at(ALIGN1, [3] * 3)
        self.assertEqual(left, ["ATC", "A-C", "A-C"])
        self.assertEqual(right, ["-G-G-TT", "CGC-ATC", "CG--ATC"])
        self.assertEqual(residues, [3, 2, 2])

    def test_tutorial_second_parent_jagged(self):
        left, right, residues = cx.split_at(ALIGN2, [3, 2, 5])
        self.assertEqual(left, ["ATC", "AC", "A---C"])
        self.assertEqual(right, ["GG---TT", "-CGCA-TC", "CGATC"])
        self.assertEqual(residues, [3, 2, 2])

    def test_parts_reassemble(self):
        left, right, _ = cx.split_at(ALIGN2, [3, 2, 5])
        self.assertEqual([l + r for l, r in zip(left, right)], ALIGN2)

    def test_extreme_indexes(self):
        left, right, residues = cx.split_at(["A-C", "-GT"], [0, 3])
        self.assertEqual(left, ["", "-GT"])
        self.assertEqual(right, ["A-C", ""])
        self.assertEqual(residues, [0, 2])

    def test_wrong_arguments_raise(self):
        with self.assertRaises(ValueError):
            cx.split_at(ALIGN1, [3, 3])               # wrong number of indexes
        with self.assertRaises(ValueError):
            cx.split_at(ALIGN1, [3, 3, 99])           # beyond the row
        with self.assertRaises(ValueError):
            cx.split_at(ALIGN1, [3, -1, 3])

    def test_does_not_modify_input(self):
        copy_ = list(ALIGN1)
        cx.split_at(ALIGN1, [3] * 3)
        self.assertEqual(ALIGN1, copy_)


class TestResiduesToIndexes(unittest.TestCase):
    def test_tutorial_example(self):
        self.assertEqual(cx.residues_to_indexes(ALIGN2, [3, 2, 2]), [3, 2, 5])

    def test_zero_and_full(self):
        self.assertEqual(cx.residues_to_indexes(["--AC", "G-T-"], [0, 2]), [0, 3])

    def test_wrong_arguments_raise(self):
        with self.assertRaises(ValueError):
            cx.residues_to_indexes(ALIGN2, [3, 2])
        with self.assertRaises(ValueError):
            cx.residues_to_indexes(ALIGN2, [3, 2, 99])


class TestPadAlignment(unittest.TestCase):
    def test_pad_right_for_left_parts(self):
        self.assertEqual(cx.pad_alignment(["ATC", "AC", "A---C"], "r"), ["ATC--", "AC---", "A---C"])

    def test_pad_left_for_right_parts(self):
        self.assertEqual(cx.pad_alignment(["GG---TT", "-CGCA-TC", "CGATC"], "l"),
                         ["-GG---TT", "-CGCA-TC", "---CGATC"])

    def test_already_rectangular_is_unchanged(self):
        self.assertEqual(cx.pad_alignment(["ATC", "A-C"], "r"), ["ATC", "A-C"])
        self.assertEqual(cx.pad_alignment(["ATC", "A-C"], "l"), ["ATC", "A-C"])

    def test_empty_strings(self):
        self.assertEqual(cx.pad_alignment(["", "A"], "r"), ["-", "A"])
        self.assertEqual(cx.pad_alignment(["", ""], "l"), ["", ""])

    def test_invalid_side_raises(self):
        with self.assertRaises(ValueError):
            cx.pad_alignment(["A", "AC"], "x")

    def test_does_not_modify_input(self):
        part = ["A", "ACG"]
        cx.pad_alignment(part, "r")
        self.assertEqual(part, ["A", "ACG"])


class TestMerge(unittest.TestCase):
    def test_tutorial_example(self):
        left = ["ATC", "A-C", "A-C"]
        right = cx.pad_alignment(["GG---TT", "-CGCA-TC", "CGATC"], "l")
        self.assertEqual(cx.merge(left, right), ["ATC-GG---TT", "A-C-CGCA-TC", "A-C---CGATC"])

    def test_different_number_of_rows_raises(self):
        with self.assertRaises(ValueError):
            cx.merge(["A", "C"], ["G"])


class TestGenerateOffspring(unittest.TestCase):
    def test_tutorial_example(self):
        of1, of2 = cx.generate_offspring(ALIGN1, ALIGN2, 3)
        self.assertEqual(of1, ["ATCGG---TT", "A-CCGCA-TC", "A-C--CGATC"])
        self.assertEqual(of2, ["ATC--G-G-TT", "AC--CGC-ATC", "A--CCG--ATC"])

    def test_row_with_no_residue_in_left_part(self):
        # hand-computed: parent1 cut at column 1 leaves row 0 without residues on the left
        p1 = ["--AC", "GGTT"]
        p2 = ["AC--", "GGTT"]
        of1, of2 = cx.generate_offspring(p1, p2, 1)
        self.assertEqual(of1, ["-AC--", "G-GTT"])
        self.assertEqual(of2, ["--AC", "GGTT"])

    def test_invalid_columns_raise(self):
        L = len(ALIGN1[0])
        for col in (-1, 0, L, L + 5):
            with self.assertRaises(ValueError, msg=f"col={col}"):
                cx.generate_offspring(ALIGN1, ALIGN2, col)

    def test_different_number_of_rows_raises(self):
        with self.assertRaises(ValueError):
            cx.generate_offspring(ALIGN1, ALIGN2[:2], 3)

    def test_does_not_modify_parents(self):
        a, b = copy.deepcopy(ALIGN1), copy.deepcopy(ALIGN2)
        cx.generate_offspring(a, b, 4)
        self.assertEqual((a, b), (ALIGN1, ALIGN2))

    def test_tutorial_offspring_valid_for_every_column(self):
        for col in range(1, len(ALIGN1[0])):
            for child in cx.generate_offspring(ALIGN1, ALIGN2, col):
                self.assertTrue(is_valid_msa(child, TUT_SEQS), msg=f"col={col}: {child}")

    def _random_parent_pairs(self, seqs, n_pairs, seed):
        rng = random.Random(seed)
        for _ in range(n_pairs):
            yield (ini.random_msa(seqs, max(1, len(seqs[0]) // 3), rng),
                   ini.random_msa(seqs, max(1, len(seqs[0]) // 3), rng))

    def test_all_offspring_valid_dna_every_column(self):
        for p1, p2 in self._random_parent_pairs(DNA_SEQS, 25, 1):
            for col in range(1, len(p1[0])):
                for child in cx.generate_offspring(p1, p2, col):
                    self.assertTrue(is_valid_msa(child, DNA_SEQS), msg=f"{p1} {p2} col={col}")

    def test_all_offspring_valid_protein(self):
        rng = random.Random(2)
        for p1, p2 in self._random_parent_pairs(PROT_SEQS, 10, 3):
            for _ in range(10):
                col = rng.randint(1, len(p1[0]) - 1)
                for child in cx.generate_offspring(p1, p2, col):
                    self.assertTrue(is_valid_msa(child, PROT_SEQS))

    def test_left_of_child1_and_right_of_child2_come_from_parent1(self):
        """Cleaning only removes columns of the padded foreign half, so parent1's own half stays intact."""
        for p1, p2 in self._random_parent_pairs(DNA_SEQS, 25, 4):
            L = len(p1[0])
            for col in range(1, L):
                of1, of2 = cx.generate_offspring(p1, p2, col)
                self.assertEqual([r[:col] for r in of1], [r[:col] for r in p1])
                self.assertEqual([r[-(L - col):] for r in of2], [r[col:] for r in p1])

    def test_length_bounded_by_sum_of_parent_lengths(self):
        for p1, p2 in self._random_parent_pairs(PROT_SEQS, 10, 5):
            for col in (1, len(p1[0]) // 2, len(p1[0]) - 1):
                for child in cx.generate_offspring(p1, p2, col):
                    self.assertLessEqual(len(child[0]), len(p1[0]) + len(p2[0]))

    def test_children_are_new_lists(self):
        of1, of2 = cx.generate_offspring(ALIGN1, ALIGN2, 3)
        of1[0] = "X"
        self.assertEqual(ALIGN1[0], "ATC-G-G-TT")


class TestCrossover(unittest.TestCase):
    def _parents(self, seqs, seed):
        rng = random.Random(seed)
        return (ini.random_msa(seqs, max(1, len(seqs[0]) // 3), rng),
                ini.random_msa(seqs, max(1, len(seqs[0]) // 3), rng))

    def test_returns_valid_msa(self):
        for seed in range(40):
            p1, p2 = self._parents(DNA_SEQS, seed)
            child = cx.crossover(p1, p2, DNA_SM, -2, random.Random(seed))
            self.assertTrue(is_valid_msa(child, DNA_SEQS))

    def test_returns_valid_msa_protein(self):
        for seed in range(10):
            p1, p2 = self._parents(PROT_SEQS, seed)
            child = cx.crossover(p1, p2, PROT_SM, -4, random.Random(seed))
            self.assertTrue(is_valid_msa(child, PROT_SEQS))

    def test_child_is_the_better_offspring_of_some_cut(self):
        for seed in range(15):
            p1, p2 = self._parents(DNA_SEQS, seed)
            child = cx.crossover(p1, p2, DNA_SM, -2, random.Random(seed))
            score = scoring.score_msa(child, DNA_SM, -2)
            ok = False
            for col in range(1, len(p1[0])):
                o1, o2 = cx.generate_offspring(p1, p2, col)
                if child in (o1, o2) and score >= max(scoring.score_msa(o1, DNA_SM, -2),
                                                      scoring.score_msa(o2, DNA_SM, -2)):
                    ok = True
                    break
            self.assertTrue(ok, msg=f"seed {seed}: child is not the best offspring of any cut")

    def test_same_seed_same_child(self):
        p1, p2 = self._parents(DNA_SEQS, 1)
        a = cx.crossover(p1, p2, DNA_SM, -2, random.Random(11))
        b = cx.crossover(p1, p2, DNA_SM, -2, random.Random(11))
        self.assertEqual(a, b)

    def test_different_seeds_explore_different_cuts(self):
        p1, p2 = self._parents(PROT_SEQS, 2)
        children = {tuple(cx.crossover(p1, p2, PROT_SM, -4, random.Random(s))) for s in range(30)}
        self.assertGreater(len(children), 1)

    def test_does_not_use_global_random(self):
        p1, p2 = self._parents(DNA_SEQS, 3)
        random.seed(0)
        a = cx.crossover(p1, p2, DNA_SM, -2, random.Random(5))
        random.seed(999)
        b = cx.crossover(p1, p2, DNA_SM, -2, random.Random(5))
        self.assertEqual(a, b)

    def test_does_not_modify_parents(self):
        p1, p2 = self._parents(DNA_SEQS, 4)
        c1, c2 = list(p1), list(p2)
        cx.crossover(p1, p2, DNA_SM, -2, random.Random(1))
        self.assertEqual((p1, p2), (c1, c2))

    def test_parent_with_a_single_column(self):
        child = cx.crossover(["A", "C"], ["A", "C"], DNA_SM, -2, random.Random(1))
        self.assertEqual(child, ["A", "C"])

    def test_identical_parents_still_valid(self):
        p1, _ = self._parents(DNA_SEQS, 6)
        for seed in range(20):
            child = cx.crossover(p1, list(p1), DNA_SM, -2, random.Random(seed))
            self.assertTrue(is_valid_msa(child, DNA_SEQS))


if __name__ == "__main__":
    unittest.main()