"""Selection and elitism for the genetic algorithm."""
import random
from typing import List


def select_elites(scores: List[float], frac: float) -> List[int]:
    """Return indices of the highest-scoring individuals, preserving tie order."""
    if not scores:
        raise ValueError("cannot select elites from an empty population")

    n_elite = min(len(scores), max(1, int(frac * len(scores))))
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:n_elite]


def select_parents(scores: List[float], n: int, rng: random.Random) -> List[int]:
    """Select parent indices using remainder stochastic sampling."""
    if n < 0:
        raise ValueError("number of parents cannot be negative")
    if n == 0:
        return []
    if not scores:
        raise ValueError("cannot select parents from an empty population")

    minimum = min(scores)
    fitness = [score - minimum for score in scores]
    total_fitness = sum(fitness)
    if total_fitness == 0:
        fitness = [1.0] * len(scores)
        total_fitness = float(len(scores))

    expected = [n * value / total_fitness for value in fitness]
    pool: List[int] = []
    fractions: List[float] = []
    for index, expected_offspring in enumerate(expected):
        guaranteed = int(expected_offspring)
        pool.extend([index] * guaranteed)
        fractions.append(expected_offspring - guaranteed)

    remaining = n - len(pool)
    while remaining:
        if not any(fraction > 0 for fraction in fractions):
            pool.extend(rng.choices(range(len(scores)), weights=fitness, k=remaining))
            break

        order = list(range(len(scores)))
        rng.shuffle(order)
        for index in order:
            if remaining == 0:
                break
            if fractions[index] > 0 and rng.random() < fractions[index]:
                pool.append(index)
                fractions[index] = 0
                remaining -= 1

    rng.shuffle(pool)
    return pool