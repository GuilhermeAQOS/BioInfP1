"""Tests for msa_ga/selection.py (Phase 3).  Run from the project root:
    python -m unittest tests.test_selection -v

Expected API:
    select_elites(scores, frac) -> list[int]          indices, best first
    select_parents(scores, n, rng) -> list[int]       n indices (repetitions allowed), shuffled
"""
import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msa_ga import selection as sel


class TestSelectElites(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(sel.select_elites(list(range(10)), 0.5)), 5)
        self.assertEqual(len(sel.select_elites(list(range(11)), 0.5)), 5)   # int() floors

    def test_best_first_and_correct(self):
        scores = [3, -5, 10, 0, 7, -1]
        self.assertEqual(sel.select_elites(scores, 0.5), [2, 4, 0])

    def test_negative_scores(self):
        self.assertEqual(sel.select_elites([-10, -3, -7, -1], 0.5), [3, 1])

    def test_at_least_one_even_with_zero_fraction(self):
        self.assertEqual(sel.select_elites([1, 9, 5], 0.0), [1])

    def test_fraction_above_one_is_capped(self):
        out = sel.select_elites([1, 2, 3], 5.0)
        self.assertEqual(sorted(out), [0, 1, 2])

    def test_ties_keep_original_order(self):
        self.assertEqual(sel.select_elites([5, 5, 5, 1], 0.75), [0, 1, 2])

    def test_single_individual(self):
        self.assertEqual(sel.select_elites([42.0], 0.5), [0])

    def test_indices_unique_and_in_range(self):
        scores = [random.Random(1).random() for _ in range(30)]
        out = sel.select_elites(scores, 0.4)
        self.assertEqual(len(set(out)), len(out))
        self.assertTrue(all(0 <= i < 30 for i in out))

    def test_does_not_modify_input(self):
        scores = [3, 1, 2]
        sel.select_elites(scores, 0.5)
        self.assertEqual(scores, [3, 1, 2])

    def test_empty_population_raises(self):
        with self.assertRaises(ValueError):
            sel.select_elites([], 0.5)


class TestSelectParents(unittest.TestCase):
    def test_returns_n_valid_indices(self):
        scores = [1.0, 4.0, 2.5, -3.0, 0.0]
        for n in (1, 2, 7, 100, 1000):                 # n may exceed the population size
            out = sel.select_parents(scores, n, random.Random(n))
            self.assertEqual(len(out), n)
            self.assertTrue(all(0 <= i < len(scores) for i in out))

    def test_zero_parents(self):
        self.assertEqual(sel.select_parents([1, 2, 3], 0, random.Random(1)), [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            sel.select_parents([1, 2, 3], -1, random.Random(1))
        with self.assertRaises(ValueError):
            sel.select_parents([], 2, random.Random(1))

    def test_same_seed_same_result(self):
        scores = [random.Random(3).random() for _ in range(20)]
        a = sel.select_parents(scores, 15, random.Random(9))
        b = sel.select_parents(scores, 15, random.Random(9))
        self.assertEqual(a, b)

    def test_does_not_use_global_random(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        random.seed(0)
        a = sel.select_parents(scores, 10, random.Random(5))
        random.seed(12345)
        b = sel.select_parents(scores, 10, random.Random(5))
        self.assertEqual(a, b)

    def test_integer_part_is_guaranteed(self):
        """fitness = [0, 10, 30], n = 8 -> expected offspring [0, 2, 6], all integers: no randomness left."""
        for seed in range(20):
            counts = Counter(sel.select_parents([0, 10, 30], 8, random.Random(seed)))
            self.assertEqual(counts[0], 0)
            self.assertEqual(counts[1], 2)
            self.assertEqual(counts[2], 6)

    def test_dominant_individual(self):
        out = sel.select_parents([0, 0, 0, 1000], 10, random.Random(1))
        self.assertEqual(out, [3] * 10)

    def test_expected_counts_match_fitness(self):
        """Remainder sampling: mean count of each individual per call ~ its expected offspring."""
        scores = [1, 2, 3, 4, 5]                       # fitness after shift: [0, 1, 2, 3, 4]
        n, reps = 7, 5000
        rng = random.Random(2)
        totals = Counter()
        for _ in range(reps):
            totals.update(sel.select_parents(scores, n, rng))
        expected = [n * f / 10 for f in (0, 1, 2, 3, 4)]
        for i, e in enumerate(expected):
            self.assertAlmostEqual(totals[i] / reps, e, delta=0.06, msg=f"individual {i}")

    def test_better_individuals_chosen_more_often(self):
        rng = random.Random(4)
        totals = Counter()
        for _ in range(500):
            totals.update(sel.select_parents([10, 20, 30, 40], 6, rng))
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])
        self.assertLess(totals[2], totals[3])

    def test_negative_scores(self):
        out = sel.select_parents([-100.0, -50.0, -10.0], 20, random.Random(6))
        self.assertEqual(len(out), 20)
        self.assertNotIn(0, out)                        # the worst has shifted fitness 0
        self.assertGreater(out.count(2), out.count(1))

    def test_all_equal_scores(self):
        seen = Counter()
        rng = random.Random(7)
        for _ in range(200):
            out = sel.select_parents([5.0, 5.0, 5.0, 5.0], 4, rng)
            self.assertEqual(len(out), 4)
            seen.update(out)
        self.assertEqual(set(seen), {0, 1, 2, 3})       # everybody has a chance

    def test_single_individual(self):
        self.assertEqual(sel.select_parents([5.0], 4, random.Random(1)), [0, 0, 0, 0])

    def test_output_is_shuffled(self):
        """Copies of the same individual must not be adjacent systematically (crossover pairs them up)."""
        results = [sel.select_parents([0, 10, 30], 8, random.Random(s)) for s in range(30)]
        self.assertTrue(any(r != sorted(r) for r in results))
        self.assertTrue(len({tuple(r) for r in results}) > 5)

    def test_terminates_with_awkward_floats(self):
        scores = [0.1, 0.2, 0.30000000000000004, 1 / 3, 1e-9, 1e9, -2.5, 7.0]
        rng = random.Random(8)
        for n in (1, 2, 3, 10, 101):
            for _ in range(100):
                self.assertEqual(len(sel.select_parents(scores, n, rng)), n)

    def test_many_tiny_fractions_terminates(self):
        rng = random.Random(10)
        scores = [rng.random() for _ in range(500)]
        for _ in range(50):
            self.assertEqual(len(sel.select_parents(scores, 1, rng)), 1)
        self.assertEqual(len(sel.select_parents(scores, 5000, rng)), 5000)

    def test_does_not_modify_input(self):
        scores = [3.0, 1.0, 2.0]
        sel.select_parents(scores, 5, random.Random(1))
        self.assertEqual(scores, [3.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()