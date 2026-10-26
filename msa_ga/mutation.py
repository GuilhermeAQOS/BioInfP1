import random
from msa_ga.validity import clean_alignment, is_valid_msa

def mutate(msa: list[str], rng: random.Random) -> list[str]:
    seqs = [row.replace("-", "") for row in msa]
    runs = gap_runs(msa)
    tmp = list(msa)
    pairs = []
    for i in range(len(runs)):
        for j in range(i+1, len(runs)):
            a, b = runs[i], runs[j]
            if a[0] != b[0] and (a[2] - a[1]) == (b[2] - b[1]) and not overlaps(a, b):
                pairs.append((a, b))

    if not pairs:
        return tmp

    run_a, run_b = rng.choice(pairs)
    mutated = swap_runs(msa, run_a, run_b)
    mutated = clean_alignment(mutated)
    if not is_valid_msa(mutated, seqs):
        return tmp
    return mutated

def grow_block(msa: list[str], row: int, col: int) -> list[tuple[int, int, int]]:

    if row < 0 or row >= len(msa):
        raise ValueError("Invalid row")
    if col < 0 or col >= len(msa[row]):
        raise ValueError("Invalid column")
    if msa[row][col] != "-":
        raise ValueError("Initial column should be a gap")
    
    gap_list = gap_runs(msa)
    start_tuple = None

    for r, start, end in gap_list:
        if r == row and start <= col < end:
            start_tuple = (r, start, end)
            break

    if start_tuple is None:
        return []

    tuples = [start_tuple]
    updated = True

    while updated:
        updated = False
        for run in gap_list:
            if run in tuples:
                continue
            if any(overlaps(run, t) for t in tuples):
                tuples.append(run)
                updated = True
    return tuples


def overlaps(a, b):
    return not (a[2] <= b[1] or b[2] <= a[1])

def gap_runs(msa: list[str]) -> list[tuple[int, int, int]]:
    gap_tuples = []
    for i, row in enumerate(msa):
        j = 0
        while j < len(row):
            if(row[j] == "-"):
                start_gap = j
                j = j + 1
                while j < len(row) and row[j] == "-":
                    j += 1
                gap_tuples.append((i, start_gap, j))
            else: 
                j += 1
    return gap_tuples

def swap_runs(msa: list[str], run_a: tuple[int, int, int], run_b: tuple[int, int, int]) -> list[str]:
    row_a, row_b = run_a[0], run_b[0]
    start_a, start_b = run_a[1], run_b[1]
    end_a, end_b = run_a[2], run_b[2]
    seq_a, seq_b = msa[row_a], msa[row_b]
    swap_length = end_a - start_a

    if end_a - start_a != end_b - start_b:
        raise ValueError("Swaps should be made with the same length")
    if row_a == row_b:
        raise ValueError("Rows should be different")
    if start_a >= end_a or start_b >= end_b:
        raise ValueError("Invalid swapping interval")
    if overlaps(run_a, run_b):
        raise ValueError("Intervals should not collide")
    if start_a < 0 or start_b < 0 or end_a > len(seq_a) or end_b > len(seq_b):
        raise ValueError("Index out of range")

    new_seq_a = ""
    new_seq_b = ""

    if start_a < start_b:
        new_seq_a = (seq_a[:start_a] + seq_a[start_a + swap_length:start_b + swap_length] 
            + seq_a[start_a:start_a + swap_length] + seq_a[start_b + swap_length:])
        new_seq_b = (seq_b[:start_a] + seq_b[start_b:start_b + swap_length]
            + seq_b[start_a:start_b] + seq_b[start_b + swap_length:])
    else:
        new_seq_b = (seq_b[:start_b] + seq_b[start_b + swap_length:start_a + swap_length] 
            + seq_b[start_b:start_b + swap_length] + seq_b[start_a + swap_length:])
        new_seq_a = (seq_a[:start_b] + seq_a[start_a:start_a + swap_length]
            + seq_a[start_b:start_a] + seq_a[start_a + swap_length:])
    
    tmp = list(msa)
    tmp[row_a] = new_seq_a
    tmp[row_b] = new_seq_b    
    return tmp