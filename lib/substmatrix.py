from typing import Dict, Optional, Tuple


class SubstMatrix:

    def __init__(self) -> None:
        self.alphabet: str = ""
        self.sm: Dict[str, int] = {}

    def __getitem__(self, ij: Tuple[str, str]) -> Optional[int]:
        i, j = ij
        return self.score_pair(i, j)

    def score_pair(self, c1: str, c2: str) -> Optional[int]:
        if c1 not in self.alphabet or c2 not in self.alphabet:
            return None
        return self.sm[c1+c2]

    def create_submat(self, match: int, mismatch: int, alphabet: str) -> None:
        self.sm = {}
        self.alphabet = alphabet
        for c1 in alphabet:
            for c2 in alphabet:
                if (c1 == c2):
                    self.sm[c1+c2] = match
                else:
                    self.sm[c1+c2] = mismatch

    def read_submat_file(self, filename: str, sep: str = "\t") -> None:
        self.sm = {}
        with open(filename ,"r") as f:
            line = f.readline()
            tokens = line.split(sep)
            ns = len(tokens)
            self.alphabet = ""
            for i in range(0,ns):
                self.alphabet += tokens[i][0]
            for i in range(0,ns):
                line = f.readline()
                tokens = line.split(sep)
                for j in range(0, len(tokens)):
                    k = self.alphabet[i]+self.alphabet[j]
                    self.sm[k] = int(tokens[j])

