"""Residue-preserving crossover operators for multiple sequence alignments."""
import random
from typing import List, Tuple

from msa_ga.scoring import score_msa
from msa_ga.validity import clean_alignment


def count_residues(line: str) -> int:
    """Return the number of non-gap characters in a row."""
    return sum(char != "-" for char in line)


def index_at_residue(line: str, n: int) -> int:
    """Return the column index after the first `n` residues, before following gaps."""
    if n < 0 or n > count_residues(line):
        raise ValueError("residue count is outside the row")
    if n == 0:
        return 0

    residues = 0
    for index, char in enumerate(line):
        if char != "-":
            residues += 1
            if residues == n:
                return index + 1
    raise ValueError("residue count is outside the row")


def split_at(msa: List[str], indexes: List[int]) -> Tuple[List[str], List[str], List[int]]:
    """Split each row at its corresponding column index and count left-side residues."""
    if len(msa) != len(indexes):
        raise ValueError("one split index is required for each row")

    left = []
    right = []
    residues = []
    for row, index in zip(msa, indexes):
        if index < 0 or index > len(row):
            raise ValueError("split index is outside the row")
        left_row, right_row = row[:index], row[index:]
        left.append(left_row)
        right.append(right_row)
        residues.append(count_residues(left_row))
    return left, right, residues


def residues_to_indexes(msa: List[str], residues: List[int]) -> List[int]:
    """Convert per-row residue counts to split indexes."""
    if len(msa) != len(residues):
        raise ValueError("one residue count is required for each row")
    return [index_at_residue(row, n) for row, n in zip(msa, residues)]


def pad_alignment(msa: List[str], side: str) -> List[str]:
    """Pad rows to equal length on the requested side ('r' or 'l')."""
    if side not in ("r", "l"):
        raise ValueError("side must be 'r' or 'l'")
    width = max((len(row) for row in msa), default=0)
    if side == "r":
        return [row.ljust(width, "-") for row in msa]
    return [row.rjust(width, "-") for row in msa]


def merge(left: List[str], right: List[str]) -> List[str]:
    """Join corresponding rows from two alignment fragments."""
    if len(left) != len(right):
        raise ValueError("alignment fragments must have the same number of rows")
    return [left_row + right_row for left_row, right_row in zip(left, right)]


def generate_offspring(parent1: List[str], parent2: List[str], col: int) -> Tuple[List[str], List[str]]:
    """Generate reciprocal children by cutting at the same residue counts."""
    if len(parent1) != len(parent2):
        raise ValueError("parents must have the same number of rows")
    if not parent1:
        raise ValueError("parents must contain at least one row")
    parent_length = len(parent1[0])
    if col < 1 or col >= parent_length:
        raise ValueError("split column must be between 1 and the last column")

    left1, right1, residue_counts = split_at(parent1, [col] * len(parent1))
    indexes2 = residues_to_indexes(parent2, residue_counts)
    left2, right2, _ = split_at(parent2, indexes2)

    offspring1 = merge(left1, pad_alignment(right2, "l"))
    offspring2 = merge(pad_alignment(left2, "r"), right1)
    return clean_alignment(offspring1), clean_alignment(offspring2)


def crossover(parent1: List[str], parent2: List[str], sm, gap: int,
              rng: random.Random) -> List[str]:
    """Return the better child from a random valid split."""
    if len(parent1[0]) <= 1:
        return list(parent1)

    col = rng.randint(1, len(parent1[0]) - 1)
    offspring1, offspring2 = generate_offspring(parent1, parent2, col)
    if score_msa(offspring1, sm, gap) >= score_msa(offspring2, sm, gap):
        return offspring1
    return offspring2