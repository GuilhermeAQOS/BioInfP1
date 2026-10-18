"""All parameters of the project in one place (defaults = report question 11), plus the
result container returned by the GA and helpers to expose every parameter on the command line."""
import argparse
from dataclasses import dataclass, field, fields
from typing import List, Tuple


@dataclass
class Params:
    # --- input / output ---------------------------------------------------------------
    fasta: str = "data/cytochromes.fa"
    matrix: str = "data/blosum62.mat"          # used for protein only
    output: str = "output/best_alignment.fasta"
    seq_type: str = "auto"                     # auto | dna | rna | protein

    # --- scoring (sum of pairs) -------------------------------------------------------
    dna_match: int = 1
    dna_mismatch: int = -1
    gap_protein: int = -4                      # linear gap penalty, protein
    gap_dna: int = -2                          # linear gap penalty, DNA/RNA

    # --- initialization ---------------------------------------------------------------
    pop_size: int = 100
    max_offset_frac: float = 0.25              # max leading-gap offset, as a fraction of sequence length

    # --- selection --------------------------------------------------------------------
    elite_frac: float = 0.5                    # best fraction copied unchanged to next generation

    # --- operators --------------------------------------------------------------------
    # Probability that a new child is produced by crossover (otherwise by mutation).
    # 1.0 -> crossover only, 0.0 -> mutation only (useful for the ablation experiment).
    crossover_prob: float = 0.5

    # --- termination ------------------------------------------------------------------
    max_generations: int = 500
    patience: int = 50                         # stop after this many generations without improvement

    # --- misc -------------------------------------------------------------------------
    seed: int = 42
    verbose: bool = True

    def gap_for(self, seq_type: str) -> int:
        return self.gap_protein if seq_type.upper() == "PROTEIN" else self.gap_dna


@dataclass
class GAResult:
    """What ga.run_ga must return."""
    best_msa: List[str]
    best_score: float
    generations: int                                   # generations actually run
    history: List[Tuple[int, float, float]] = field(default_factory=list)  # (generation, best, mean)
    runtime: float = 0.0                               # seconds
    stop_reason: str = ""                              # e.g. "max_generations" or "no_improvement"


def add_param_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one --flag per Params field (underscores become dashes). New fields appear automatically."""
    for f in fields(Params):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default,
                                help=f"(default: {default})")
        else:
            kw = dict(type=type(default), default=default, help=f"(default: {default})")
            if f.name == "seq_type":
                kw["choices"] = ["auto", "dna", "rna", "protein"]
            parser.add_argument(flag, **kw)


def params_from_args(args: argparse.Namespace) -> Params:
    return Params(**{f.name: getattr(args, f.name) for f in fields(Params)})
