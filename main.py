"""Command-line entry point.  Run from the project root:

    python main.py                                   # cytochromes.fa with default parameters
    python main.py --fasta data/cytochromes.fa --pop-size 30 --seed 1
    python main.py --help                            # every parameter and its default
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msa_ga.ga import run_ga
from msa_ga.io_utils import load_problem, write_alignment, write_run_info
from msa_ga.params import add_param_arguments, params_from_args
from msa_ga.validity import is_valid_msa


def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic algorithm for multiple sequence alignment")
    add_param_arguments(parser)
    params = params_from_args(parser.parse_args())

    names, seqs, seq_type, sm, gap = load_problem(params)
    print(f"Loaded {len(seqs)} {seq_type} sequences from {params.fasta} "
          f"(lengths {min(map(len, seqs))}-{max(map(len, seqs))}), gap penalty {gap}, seed {params.seed}")

    rng = random.Random(params.seed)
    result = run_ga(seqs, sm, gap, params, rng)

    if not is_valid_msa(result.best_msa, seqs):
        raise SystemExit("ERROR: the GA returned an invalid MSA")

    print("\n=== Result ===")
    print(f"Best score:       {result.best_score}")
    print(f"Alignment length: {len(result.best_msa[0])}")
    print(f"Generations:      {result.generations} ({result.stop_reason})")
    print(f"Runtime:          {result.runtime:.2f} s")

    write_alignment(params.output, names, result.best_msa)
    info_path = os.path.splitext(params.output)[0] + "_info.txt"
    write_run_info(info_path, params, result, seq_type, len(seqs))
    print(f"Alignment written to {params.output} (run info: {info_path})")


if __name__ == "__main__":
    main()
