import random
import time
from typing import List

from msa_ga.params import GAResult, Params
from msa_ga.initialization import init_population
from msa_ga.scoring import score_msa
from msa_ga.selection import select_elites, select_parents
from msa_ga.crossover import crossover
from msa_ga.mutation import mutate


def run_ga(seqs, sm, gap, params: Params, rng: random.Random) -> GAResult:
    t0 = time.perf_counter()
    pop = init_population(seqs, params.pop_size, params.max_offset_frac, rng)
    scores: List[float] = [score_msa(indiv, sm, gap) for indiv in pop]
    prev_best = max(scores)
    idx = scores.index(prev_best)
    best_indiv = pop[idx]
    score_mean = sum(scores) / len(scores)
    history = [(0, prev_best, score_mean)]
    stop_reason = ""
    stall = 0
    final_gens = 0
    for gen in range(1, params.max_generations + 1):
        elite_idx = select_elites(scores, params.elite_frac)
        next_pop = [list(pop[i]) for i in elite_idx]
        seen = {tuple(indiv) for indiv in next_pop}
        attempts = 0
        max_attempts = params.pop_size * 10

        while len(next_pop) < params.pop_size and attempts < max_attempts:
            attempts += 1
            if rng.random() < params.crossover_prob:
                i, j = select_parents(scores, 2, rng)
                child = crossover(pop[i], pop[j], sm, gap, rng)
            else:
                [i] = select_parents(scores, 1, rng)
                child = mutate(pop[i], rng)
            key = tuple(child)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(child)

        while len(next_pop) < params.pop_size:
            i = select_parents(scores, 1, rng)[0]
            next_pop.append(mutate(pop[i], rng))

        pop = next_pop
        scores: List[float] = [score_msa(indiv, sm, gap) for indiv in pop]

        cur_best = max(scores)
        score_mean = sum(scores) / len(scores)
        history.append((gen, cur_best, score_mean))
        if cur_best > prev_best:
            stall = 0
            prev_best = cur_best
            idx = scores.index(prev_best)
            best_indiv = pop[idx]
        else:
            stall += 1
        
        if stall >= params.patience:
            stop_reason = "no_improvement"
            final_gens = gen
            break
    else:
        stop_reason = "max_generations"
        final_gens = params.max_generations
    runtime = time.perf_counter() - t0
    return GAResult(best_indiv, prev_best, final_gens, history, runtime, stop_reason)