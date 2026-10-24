"""Phase 1 - population initialization."""
import random
from typing import List

from msa_ga.validity import clean_alignment


def random_msa(seqs: List[str], max_offset: int, rng: random.Random) -> List[str]:
    rows = ["-" * rng.randint(0, max_offset) + seq for seq in seqs]
    if not rows:
        return []
    longest = max(len(row) for row in rows)
    rows = [row.ljust(longest, "-") for row in rows]
    return clean_alignment(rows)


def init_population(seqs: List[str], pop_size: int, max_offset_frac: float, rng: random.Random) -> List[List[str]]:
    max_seq_length = max((len(seq) for seq in seqs), default=0)
    max_offset = int(max_offset_frac * max_seq_length)
    population: List[List[str]] = []
    seen = set()
    max_attempts = max(100, pop_size * 10)

    for _ in range(max_attempts):
        if len(population) >= pop_size:
            break
        individual = random_msa(seqs, max_offset, rng)
        key = tuple(individual)
        if key not in seen:
            seen.add(key)
            population.append(individual)

    return population
