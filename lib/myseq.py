from typing import Optional, Union


class MySeq:
    """ Class for Biological sequences. """
    def __init__(self, seq: str, seq_type: str = "DNA") -> None:
        self.seq: str = seq
        self.seq_type: str = seq_type

    def __len__(self) -> int:
        return len(self.seq)

    def __str__(self) -> str:
        return self.seq

    def __getitem__(self, n: Union[int, slice]) -> str:
        return self.seq[n]

    def print_sequence(self) -> None:
        print("Sequence: " + self.seq)

    def get_seq_biotype(self) -> str:
        return self.seq_type

    def show_info_seq(self) -> None:
        print("Sequence: " + self.seq + " biotype: " + self.seq_type)

    def count_occurrences(self, seq_search: str) -> int:
        return self.seq.count(seq_search)

    # class method to validate biotype update
    def set_seq_biotype(self, bt: str) -> None:
        biotype = bt.upper()
        if biotype == "DNA" or biotype == "RNA" or biotype == "PROTEIN":
            self.seq_type = biotype
        else:
            print("Non biological sequence type!")

    def alphabet(self) -> Optional[str]:
        if (self.seq_type=="DNA"): return "ACGT"
        elif (self.seq_type=="RNA"): return "ACGU"
        elif (self.seq_type=="PROTEIN"): return "ACDEFGHIKLMNPQRSTVWY"
        else: return None

    def validate(self) -> bool:
        alp = self.alphabet()
        if alp is None: return False
        res = True
        i = 0
        while i < len(self.seq) and res:
            if self.seq[i] not in alp: res = False
            else: i += 1
        return res

    def transcription(self) -> Optional["MySeq"]:
        if (self.seq_type == "DNA"):
            return MySeq(self.seq.replace("T","U"), "RNA")
        else:
            return None


