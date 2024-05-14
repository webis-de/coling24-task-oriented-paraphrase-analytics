from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text


class MSRPCParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def get_next(self):
        for line in self.in_file:
            if line.startswith("﻿Quality"):
                continue

            comp = line.split("\t")
            paraphrase_set = ParaphraseSet(f"{comp[1]}-{comp[2]}",
                                           quality="1" == comp[0])

            paraphrase_set.add_text(Text(comp[3], _id=comp[1]))
            paraphrase_set.add_text(Text(comp[4], _id=comp[2]))

            yield paraphrase_set
