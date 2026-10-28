"""Experiment runner for report questions 8 and 9.  Run from the project root:

    python experiments/run_experiments.py --n-runs 10 --n-random 1000 --progressive
    python experiments/run_experiments.py --fasta data/small_dna.fa --out-dir experiments/results_dna --n-runs 10
    python experiments/run_experiments.py --skip-ablation --n-runs 5 --max-generations 150 \
        --sweep max_offset_frac=0.1,0.25,0.5 --sweep pop_size=20,50,100

Experiments
  1. random baseline   : SP score of `n_random` randomly initialised MSAs (no evolution)
  2. progressive       : (--progressive) the lab's progressive alignment, scored with our SP function
  3. ablation          : full GA vs crossover only vs mutation only (n_runs seeds each)
  4. sweeps            : (--sweep name=v1,v2,...) one parameter varied, full GA; can be repeated
All configurations use the same seeds, so runs are paired.  Results go to --out-dir:
random_baseline.csv, progressive.csv, summary.csv (one row per run), history.csv (one row per
generation), aggregate.csv (one row per configuration = table for the report).
Any Params flag (--pop-size, --max-generations, --fasta, ...) sets the shared base configuration."""
import argparse
import csv
import os
import random
import statistics
import sys
from dataclasses import replace
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import msa_ga  # noqa: F401,E402  (adds lib/ to sys.path)
from msa_ga.ga import run_ga  # noqa: E402
from msa_ga.initialization import init_population  # noqa: E402
from msa_ga.io_utils import load_problem  # noqa: E402
from msa_ga.params import Params, add_param_arguments, params_from_args  # noqa: E402
from msa_ga.scoring import score_msa  # noqa: E402
from msa_ga.validity import clean_alignment, is_valid_msa  # noqa: E402

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


def progressive_baseline(seqs, seq_type, sm, gap) -> List[str]:
    """Progressive alignment with the professor's lab code (lib/multiplealign.py, consensus-based).
    That file calls testMSA() at import time, so we run its source without that one line."""
    from myseq import MySeq
    from pairwisealignment import PairwiseAlignment

    with open(os.path.join(ROOT, "lib", "multiplealign.py")) as f:
        source = "\n".join(line for line in f.read().splitlines() if line.strip() != "testMSA()")
    namespace = {"__name__": "multiplealign_lab"}
    exec(compile(source, "multiplealign.py", "exec"), namespace)
    ma = namespace["MultipleAlignment"]([MySeq(s, seq_type) for s in seqs], PairwiseAlignment(sm, gap))
    return clean_alignment(ma.align_consensus().listseqs)


def run_config(label: str, base: Params, seqs, sm, gap, seeds: List[int], summary_rows: list, history_rows: list):
    for seed in seeds:
        p = replace(base, seed=seed, verbose=False)
        res = run_ga(seqs, sm, gap, p, random.Random(seed))
        if not is_valid_msa(res.best_msa, seqs):
            raise SystemExit(f"{label} seed {seed}: the GA returned an invalid MSA")
        summary_rows.append([label, seed, res.best_score, res.generations, round(res.runtime, 3),
                             len(res.best_msa[0]), res.stop_reason])
        for gen, best, mean in res.history:
            history_rows.append([label, seed, gen, best, mean])
        print(f"  {label:<24} seed={seed:<5} best={res.best_score:<10} gens={res.generations:<5} "
              f"len={len(res.best_msa[0]):<4} time={res.runtime:.1f}s ({res.stop_reason})", flush=True)


def write_csv(path: str, header: list, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def parse_sweep(text: str, base: Params):
    """'name=v1,v2,v3' -> (name, [values cast to the type of Params.name])."""
    name, values = text.split("=", 1)
    if not hasattr(base, name):
        raise SystemExit(f"--sweep: unknown parameter {name!r}")
    ftype = type(getattr(base, name))
    return name, [ftype(v) for v in values.split(",")]


def aggregate(summary_rows: list) -> list:
    """One row per configuration: n, mean/std/min/max best score, mean generations, time, length."""
    rows = []
    for label in dict.fromkeys(r[0] for r in summary_rows):
        group = [r for r in summary_rows if r[0] == label]
        best = [r[2] for r in group]
        rows.append([label, len(group), round(statistics.mean(best), 2),
                     round(statistics.stdev(best), 2) if len(best) > 1 else 0.0, min(best), max(best),
                     round(statistics.mean(r[3] for r in group), 1),
                     round(statistics.mean(r[4] for r in group), 2),
                     round(statistics.mean(r[5] for r in group), 1)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_param_arguments(parser)
    parser.add_argument("--n-runs", type=int, default=10, help="GA runs (seeds) per configuration")
    parser.add_argument("--n-random", type=int, default=1000, help="random MSAs for the baseline")
    parser.add_argument("--base-seed", type=int, default=1000, help="first seed; runs use base, base+1, ...")
    parser.add_argument("--out-dir", default="experiments/results")
    parser.add_argument("--sweep", action="append", default=[], help="e.g. max_offset_frac=0.1,0.25,0.5 (repeatable)")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--progressive", action="store_true", help="also score the lab's progressive alignment")
    args = parser.parse_args()
    base = params_from_args(args)
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    names, seqs, seq_type, sm, gap = load_problem(base)
    print(f"{len(seqs)} {seq_type} sequences from {base.fasta}; {len(seeds)} seeds per configuration")

    # 1. random baseline
    print("Random baseline...")
    scores = random_baseline(seqs, sm, gap, base, args.n_random, args.base_seed)
    write_csv(os.path.join(args.out_dir, "random_baseline.csv"), ["score"], [[s] for s in scores])
    print(f"  {len(scores)} random MSAs: mean={statistics.mean(scores):.1f} max={max(scores)} min={min(scores)}")

    # 2. progressive alignment (reference)
    if args.progressive:
        print("Progressive alignment (lab code)...")
        prog = progressive_baseline(seqs, seq_type, sm, gap)
        if not is_valid_msa(prog, seqs):
            raise SystemExit("progressive alignment is not a valid MSA")
        prog_score = score_msa(prog, sm, gap)
        write_csv(os.path.join(args.out_dir, "progressive.csv"), ["score", "length"], [[prog_score, len(prog[0])]])
        print(f"  progressive SP score = {prog_score} (length {len(prog[0])})")

    summary_rows: list = []
    history_rows: list = []

    # 3. ablation
    if not args.skip_ablation:
        print("Ablation (full / crossover only / mutation only)...")
        for label, overrides in ABLATION.items():
            run_config(label, replace(base, **overrides), seqs, sm, gap, seeds, summary_rows, history_rows)

    # 4. sweeps
    for text in args.sweep:
        name, values = parse_sweep(text, base)
        print(f"Sweep over {name}...")
        for v in values:
            run_config(f"{name}={v}", replace(base, **{name: v}), seqs, sm, gap, seeds, summary_rows, history_rows)

    write_csv(os.path.join(args.out_dir, "summary.csv"),
              ["config", "seed", "best_score", "generations", "runtime_s", "final_length", "stop_reason"], summary_rows)
    write_csv(os.path.join(args.out_dir, "history.csv"),
              ["config", "seed", "generation", "best", "mean"], history_rows)
    agg = aggregate(summary_rows)
    write_csv(os.path.join(args.out_dir, "aggregate.csv"),
              ["config", "n_runs", "mean_best", "std_best", "min_best", "max_best",
               "mean_generations", "mean_runtime_s", "mean_length"], agg)

    print("\nconfig                    n   mean best      std      min      max   gens   time(s)  length")
    for r in agg:
        print(f"{r[0]:<24} {r[1]:>3} {r[2]:>11} {r[3]:>8} {r[4]:>8} {r[5]:>8} {r[6]:>6} {r[7]:>9} {r[8]:>7}")
    print(f"\nCSV files written to {args.out_dir}/")


if __name__ == "__main__":
    main()