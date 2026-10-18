"""Input/output helpers: FASTA, substitution matrices, sequence-type detection, output files."""
import os
from typing import Dict, List, Tuple

import msa_ga  # noqa: F401  (makes ../lib importable)
from substmatrix import SubstMatrix

from msa_ga.params import GAResult, Params


# ---------------------------------------------------------------- FASTA -----------------
def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """Return (names, sequences). Sequences are upper-cased and stripped of gap characters
    and whitespace (the input file may use Windows line endings)."""
    names: List[str] = []
    seqs: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()
                names.append(header[0] if header else f"seq{len(names) + 1}")
                seqs.append("")
            else:
                if not seqs:
                    raise ValueError(f"{path}: sequence data before first '>' header")
                seqs[-1] += line.upper().replace("-", "").replace(" ", "")
    if any(len(s) == 0 for s in seqs):
        raise ValueError(f"{path}: contains an empty sequence")
    return names, seqs


def write_alignment(path: str, names: List[str], msa: List[str], width: int = 60) -> None:
    """Write an alignment as aligned FASTA (one record per sequence, lines wrapped at `width`)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for name, row in zip(names, msa):
            f.write(f">{name}\n")
            for i in range(0, len(row), width):
                f.write(row[i:i + width] + "\n")


def write_run_info(path: str, params: Params, result: GAResult, seq_type: str, n_seqs: int) -> None:
    """Write a small text file with the parameters and statistics of a run."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"sequences: {n_seqs} ({seq_type})\n")
        f.write(f"alignment length: {len(result.best_msa[0])}\n")
        f.write(f"best score: {result.best_score}\n")
        f.write(f"generations: {result.generations}\n")
        f.write(f"stop reason: {result.stop_reason}\n")
        f.write(f"runtime (s): {result.runtime:.2f}\n")
        f.write("parameters:\n")
        for k, v in vars(params).items():
            f.write(f"  {k} = {v}\n")


# ------------------------------------------------------- sequence type ------------------
def detect_seq_type(seqs: List[str]) -> str:
    """'DNA', 'RNA' or 'PROTEIN' (same names as MySeq.seq_type)."""
    chars = set("".join(seqs))
    if chars <= set("ACGTN"):
        return "DNA"
    if chars <= set("ACGUN"):
        return "RNA"
    return "PROTEIN"


def check_alphabet(seqs: List[str], alphabet: str) -> Dict[str, int]:
    """Characters of the sequences that are missing from the substitution matrix alphabet,
    with their counts. Empty dict means everything can be scored."""
    unknown: Dict[str, int] = {}
    for s in seqs:
        for c in s:
            if c not in alphabet:
                unknown[c] = unknown.get(c, 0) + 1
    return unknown


# ------------------------------------------------ substitution matrices ----------------
def load_substitution_matrix(path: str) -> SubstMatrix:
    """Load a matrix file into the professor's SubstMatrix.

    Accepted format: a header line with the symbols, then one line of integers per symbol
    (optionally preceded by a row label). Any whitespace separator and Windows line endings
    are accepted; lines starting with '#' are ignored."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    alphabet = lines[0].split()
    if not all(len(a) == 1 for a in alphabet):
        raise ValueError(f"{path}: first line must list single-character symbols")
    sm = SubstMatrix()
    sm.alphabet = "".join(alphabet)
    sm.sm = {}
    rows = lines[1:]
    if len(rows) != len(alphabet):
        raise ValueError(f"{path}: expected {len(alphabet)} rows, found {len(rows)}")
    for i, line in enumerate(rows):
        tokens = line.split()
        if len(tokens) == len(alphabet) + 1:        # drop the row label
            tokens = tokens[1:]
        if len(tokens) != len(alphabet):
            raise ValueError(f"{path}: row {i + 1} has {len(tokens)} values")
        for j, tok in enumerate(tokens):
            sm.sm[alphabet[i] + alphabet[j]] = int(tok)
    return sm


def build_submat(seq_type: str, params: Params) -> SubstMatrix:
    """BLOSUM62 file for proteins; match/mismatch values for DNA/RNA."""
    seq_type = seq_type.upper()
    if seq_type == "PROTEIN":
        return load_substitution_matrix(params.matrix)
    sm = SubstMatrix()
    sm.create_submat(params.dna_match, params.dna_mismatch, "ACGU" if seq_type == "RNA" else "ACGT")
    return sm


# ----------------------------------------------------------- everything at once --------
def load_problem(params: Params) -> Tuple[List[str], List[str], str, SubstMatrix, int]:
    """Read the FASTA and build the scoring setup. Returns (names, seqs, seq_type, sm, gap).
    Exits with a clear message if some residues cannot be scored."""
    names, seqs = read_fasta(params.fasta)
    seq_type = detect_seq_type(seqs) if params.seq_type == "auto" else params.seq_type.upper()
    sm = build_submat(seq_type, params)
    unknown = check_alphabet(seqs, sm.alphabet)
    if unknown:
        raise SystemExit(f"Residues not in the substitution matrix alphabet: {unknown}")
    return names, seqs, seq_type, sm, params.gap_for(seq_type)
