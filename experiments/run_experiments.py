"""Experiment runner for report question 9.  Run from the project root:

    python experiments/run_experiments.py --n-runs 10 --n-random 1000
    python experiments/run_experiments.py --n-runs 5 --sweep max_offset_frac=0.1,0.25,0.5

Experiments
  1. random baseline  : score of `n_random` randomly initialised MSAs (no evolution)
  2. ablation         : full GA vs crossover only vs mutation only (n_runs seeds each)
  3. optional sweep   : one parameter varied over a list of values (full GA)
Results are written to experiments/results/ and plotted by plot_results.py.
Any Params flag (--pop-size, --max-generations, --fasta, ...) sets the shared base configuration."""
import argparse
import csv
import os
import random
import statistics
import sys
import time
from dataclasses import fields, replace
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msa_ga.ga import run_ga
from msa_ga.initialization import init_population
from msa_ga.io_utils import load_problem
from msa_ga.params import Params, add_param_arguments, params_from_args
from msa_ga.scoring import score_msa

# label -> overrides applied to the base Params
ABLATION: Dict[str, dict] = {
    "full": {},
    "crossover_only": {"crossover_prob": 1.0},
    "mutation_only": {"crossover_prob": 0.0},
}


def random_baseline(seqs, sm, gap, params: Params, n: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    pop = init_population(seqs, n, params.max_offset_frac, rng)
    return [score_msa(m, sm, gap) for m in pop]


def run_config(label: str, base: Params, seqs, sm, gap, seeds: List[int], summary_rows: list, history_rows: list):
    for seed in seeds:
        p = replace(base, seed=seed, verbose=False)
        res = run_ga(seqs, sm, gap, p, random.Random(seed))
        summary_rows.append([label, seed, res.best_score, res.generations, round(res.runtime, 3), res.stop_reason])
        for gen, best, mean in res.history:
            history_rows.append([label, seed, gen, best, mean])
        print(f"  {label:<18} seed={seed:<5} best={res.best_score:<10} gens={res.generations:<5} "
              f"time={res.runtime:.1f}s ({res.stop_reason})", flush=True)


def write_csv(path: str, header: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def parse_sweep(text: str, base: Params):
    """'name=v1,v2,v3' -> (name, [values cast to the type of Params.name])."""
    name, values = text.split("=", 1)
    ftype = type(getattr(base, name))
    return name, [ftype(v) for v in values.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_param_arguments(parser)
    parser.add_argument("--n-runs", type=int, default=10, help="GA runs (seeds) per configuration")
    parser.add_argument("--n-random", type=int, default=1000, help="random MSAs for the baseline")
    parser.add_argument("--base-seed", type=int, default=1000, help="first seed; runs use base, base+1, ...")
    parser.add_argument("--out-dir", default="experiments/results")
    parser.add_argument("--sweep", default="", help="e.g. max_offset_frac=0.1,0.25,0.5")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    base = params_from_args(args)
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    names, seqs, seq_type, sm, gap = load_problem(base)
    print(f"{len(seqs)} {seq_type} sequences from {base.fasta}; {len(seeds)} seeds per configuration")

    # 1. random baseline
    print("Random baseline...")
    scores = random_baseline(seqs, sm, gap, base, args.n_random, args.base_seed)
    write_csv(os.path.join(args.out_dir, "random_baseline.csv"), ["score"], [[s] for s in scores])
    print(f"  random MSAs: mean={statistics.mean(scores):.1f} max={max(scores)} min={min(scores)}")

    summary_rows: list = []
    history_rows: list = []

    # 2. ablation
    if not args.skip_ablation:
        print("Ablation (full / crossover only / mutation only)...")
        for label, overrides in ABLATION.items():
            run_config(label, replace(base, **overrides), seqs, sm, gap, seeds, summary_rows, history_rows)

    # 3. optional sweep
    if args.sweep:
        name, values = parse_sweep(args.sweep, base)
        print(f"Sweep over {name}...")
        for v in values:
            run_config(f"{name}={v}", replace(base, **{name: v}), seqs, sm, gap, seeds, summary_rows, history_rows)

    write_csv(os.path.join(args.out_dir, "summary.csv"),
              ["config", "seed", "best_score", "generations", "runtime_s", "stop_reason"], summary_rows)
    write_csv(os.path.join(args.out_dir, "history.csv"),
              ["config", "seed", "generation", "best", "mean"], history_rows)

    print("\nconfig               mean best   std    mean gens  mean time(s)")
    for label in dict.fromkeys(r[0] for r in summary_rows):
        rows = [r for r in summary_rows if r[0] == label]
        b = [r[2] for r in rows]
        sd = statistics.stdev(b) if len(b) > 1 else 0.0
        print(f"{label:<20} {statistics.mean(b):>9.1f} {sd:>6.1f} {statistics.mean(r[3] for r in rows):>10.1f} "
              f"{statistics.mean(r[4] for r in rows):>12.1f}")
    print(f"\nCSV files written to {args.out_dir}/")


if __name__ == "__main__":
    main()
