"""Validity of an MSA (list of gapped strings). Shared by initialization, crossover and mutation."""
from typing import List


def is_valid_msa(msa: List[str], seqs: List[str]) -> bool:
    """True if: same number of rows as sequences, all rows same length, removing the gaps of row i
    gives exactly sequence i, and no column is made only of gaps."""
    if len(msa) != len(seqs):
        return False
    if len({len(r) for r in msa}) != 1:
        return False
    if any(r.replace("-", "") != s for r, s in zip(msa, seqs)):
        return False
    L = len(msa[0])
    return all(any(r[c] != "-" for r in msa) for c in range(L))


def is_all_gaps(msa: List[str], col: int) -> bool:
    return all(row[col] == "-" for row in msa)


def clean_alignment(msa: List[str]) -> List[str]:
    if not msa:
        return []
    columns = [col for col in range(len(msa[0])) if not is_all_gaps(msa, col)]
    return ["".join(row[col] for col in columns) for row in msa]
