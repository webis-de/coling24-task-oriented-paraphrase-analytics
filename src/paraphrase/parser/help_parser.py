from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import os


class HelpParser(Parser):
    def get_next(self):
        for line in self.in_file:
            comp = line.split("\t")

            if comp[4] != "entailment":
                continue

            paraphrase_set = ParaphraseSet(comp[0])
            paraphrase_set.add_text(Text(comp[8]))
            paraphrase_set.add_text(Text(comp[9]))

            yield paraphrase_set
