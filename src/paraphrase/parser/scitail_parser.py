from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import os


class SciTailParser(Parser):
    def get_next(self):
        for line in self.in_file:
            comp = line.strip().split("\t")

            if comp[2] != "entails":
                continue

            paraphrase_set = ParaphraseSet(str(hash(comp[0] + comp[1])))
            paraphrase_set.add_text(Text(comp[0]))
            paraphrase_set.add_text(Text(comp[1]))

            yield paraphrase_set
