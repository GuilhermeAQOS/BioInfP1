from typing import List, Optional, Tuple

if __package__:
    from .myalign import MyAlign
    from .myseq import MySeq
    from .substmatrix import SubstMatrix
else:
    from myalign import MyAlign
    from myseq import MySeq
    from substmatrix import SubstMatrix

class PairwiseAlignment:
    def __init__(self, sm: SubstMatrix, g: int) -> None:
        self.g: int = g
        self.sm: SubstMatrix = sm
        self.S: List[List[int]] = []
        self.T: List[List[int]] = []
        self.seq1: MySeq = MySeq("")
        self.seq2: MySeq = MySeq("")

    def score_pos(self, c1: str, c2: str) -> int:
        if c1 == "-" and c2=="-":
            return 0
        elif c1 == "-" or c2=="-":
            return self.g
        else:
            return self.sm.sm[c1+c2]

    def score_align(self) -> int:
        res = 0
        for i in range(len(self.seq1)):
            res += self.score_pos(self.seq1[i], self.seq2[i])
        return res

    def max3t(self, v1: int, v2: int, v3: int) -> int:
        if v1 > v2:
            if v1 > v3: return 1
            else: return 3
        else:
            if v2 > v3: return 2
            else: return 3

    def needleman_Wunsch(self, seq1: MySeq, seq2: MySeq) -> Optional[int]:
        if (seq1.seq_type != seq2.seq_type): return None
        self.seq1 = seq1
        self.seq2 = seq2
        self.S = [[0]]
        self.T = [[0]]
        ## initialize first row
        for j in range(1,len(self.seq2)+1):
            self.S[0].append(self.g * j)
            self.T[0].append(3)
        ## initialize first column
        for i in range(1,len(self.seq1)+1):
            self.S.append([self.g * i])
            self.T.append([2])
        ## apply the recurrence relation to fill the remaining of the matrix
        for i in range(0,len(self.seq1)):
            for j in range(len(self.seq2)):
                s1 = self.S[i][j] + self.score_pos(self.seq1[i], self.seq2[j])
                s2 = self.S[i][j+1] + self.g
                s3 = self.S[i+1][j] + self.g
                self.S[i+1].append(max(s1, s2, s3))
                self.T[i+1].append(self.max3t(s1, s2, s3))
        return self.S[len(self.seq1)][len(self.seq2)]

    def recover_align(self) -> MyAlign:
        res = ["", ""]
        i = len(self.seq1); j = len(self.seq2)
        while i>0 or j>0:
            if self.T[i][j] == 1:
                res[0] = self.seq1[i-1] + res[0]
                res[1] = self.seq2[j-1] + res[1]
                i -= 1
                j -= 1
            elif self.T[i][j] == 3:
                res[0] = "-" + res[0]
                res[1] = self.seq2[j-1] + res[1]
                j -= 1
            else:
                res[0] = self.seq1[i-1] + res[0]
                res[1] = "-" + res[1]
                i -= 1
        return MyAlign(res, self.seq1.seq_type)

    def smith_Waterman(self, seq1: MySeq, seq2: MySeq) -> Optional[Tuple[int, int]]:
        if (seq1.seq_type != seq2.seq_type): return None
        self.seq1 = seq1
        self.seq2 = seq2
        self.S = [[0]]; self.T = [[0]]; maxscore = 0
        maxrow = 0; maxcol = 0
        for j in range(1,len(self.seq2)+1):
            self.S[0].append(0)
            self.T[0].append(0)
        for i in range(1,len(self.seq1)+1):
            self.S.append([0])
            self.T.append([0])
        for i in range(0,len(self.seq1)):
            for j in range(len(self.seq2)):
                s1 = self.S[i][j] + self.score_pos(self.seq1[i], self.seq2[j])
                s2 = self.S[i][j+1] + self.g
                s3 = self.S[i+1][j] + self.g
                b = max(s1, s2, s3)
                if b <= 0:
                    self.S[i+1].append(0)
                    self.T[i+1].append(0)
                else:
                    self.S[i+1].append(b)
                    self.T[i+1].append(self.max3t(s1, s2, s3))
                    if b > maxscore:
                        maxscore = b; maxrow = i + 1; maxcol = j + 1
        return (maxrow, maxcol)

    def recover_align_local(self, i: int, j: int) -> MyAlign:
        res = ["", ""]
        while self.T[i][j]>0:
            if self.T[i][j] == 1:
                res[0] = self.seq1[i-1] + res[0]
                res[1] = self.seq2[j-1] + res[1]
                i -= 1
                j -= 1
            elif self.T[i][j] == 3:
                res[0] = "-" + res[0]
                res[1] = self.seq2[j-1] + res[1]
                j -= 1
            elif self.T[i][j] == 2:
                res[0] = self.seq1[i-1] + res[0]
                res[1] = "-" + res[1]
                i -= 1
        return MyAlign(res, self.seq1.seq_type)

def testGlobal() -> None:
    submat = SubstMatrix()
    submat.read_submat_file("blosum62.mat")
    seq1 = MySeq("PHSWG","PROTEIN")
    seq2 = MySeq("HGWAG","PROTEIN")
    pa = PairwiseAlignment(submat,-8)
    score = pa.needleman_Wunsch(seq1, seq2)
    align = pa.recover_align()
    print("Sequences to align:", seq1, seq2)
    print("Score of optimal alignment:", score)
    print("Optimal alignment:",align)

def testLocal() -> None:
    submat = SubstMatrix()
    submat.read_submat_file("blosum62.mat")
    seq1 = MySeq("PHSWG","PROTEIN")
    seq2 = MySeq("HGWAG","PROTEIN")
    pa = PairwiseAlignment(submat,-8)
    res = pa.smith_Waterman(seq1, seq2)
    if res is None: return
    (i,j) = res
    align = pa.recover_align_local(i,j)
    print("Sequences to align:", seq1, seq2)
    print("Score of optimal alignment:", pa.S[i][j])
    print("Optimal alignment:",align)
