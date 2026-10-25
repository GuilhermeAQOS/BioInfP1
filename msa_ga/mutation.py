import random

def mutate(msa: list[str], rng: random.Random) -> list[str]:
    return msa

def grow_block(msa: list[str], row: int, col: int) -> list[str]:
    return msa

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
    if not (end_a <= start_b or end_b <= start_a):
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