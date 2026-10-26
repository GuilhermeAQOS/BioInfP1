from lib.pairwisealignment import PairwiseAlignment
from lib.myseq import MySeq

def score_msa(msa, sm, gap):
    pa = PairwiseAlignment(sm, gap)
    score = 0
    for i in range(len(msa)):
        seq1 = MySeq(msa[i])
        for j in range(i+1, len(msa)):
            seq2 = MySeq(msa[j])
            score += score_pairwise(seq1, seq2, pa)
    return score

def score_pairwise(seq1, seq2, pa: PairwiseAlignment):
    score = 0
    for i in range(0,len(seq1)):
        score += pa.score_pos(seq1[i], seq2[i])
    return score