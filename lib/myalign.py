from typing import Dict, List, Optional, Tuple, Union


class MyAlign:
    def __init__(self, lseqs: List[str], al_type: str = "protein") -> None:
        self.listseqs: List[str] = lseqs
        self.al_type: str = al_type

    def __len__(self) -> int: # number of columns
        return len(self.listseqs[0])

    def __getitem__(self, n: Union[int, Tuple[int, int]]) -> Optional[str]:
        if type(n) is tuple and len(n) == 2:
            i, j = n
            return self.listseqs[i][j]
        elif type(n) is int: return self.listseqs[n]
        return None

    def __str__(self) -> str:
        res = ""
        for seq in self.listseqs:
            res += "\n" + seq
        return res

    def num_seqs(self) -> int:
        return len(self.listseqs)

    def column(self, indice: int) -> List[str]:
        res: List[str] = []
        for k in range( len (self.listseqs)):
            res.append(self.listseqs[k][indice])
        return res

    def consensus(self) -> str:
        cons = ""
        for i in range(len(self)):
            cont: Dict[str, int] = {}
            for k in range(len(self.listseqs)):
                c = self.listseqs[k][i]
                if c in cont: cont[c] = cont[c] + 1
                else: cont[c] = 1
            maximum = 0; cmax: Optional[str] = None
            for ke in cont.keys():
                if ke != "-" and cont[ke] > maximum:
                    maximum = cont[ke]
                    cmax = ke
            if cmax is None: cmax = "-" # column with only gaps
            cons = cons + cmax
        return cons
