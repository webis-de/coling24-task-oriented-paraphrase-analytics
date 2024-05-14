from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import os


class MSRParser(Parser):
    def get_next(self):
        for line in self.in_file:
            comp = line.split(" ||| ")

            source = comp[0]
            source_comp = source.split("\t")

            for i in range(1, len(comp)):
                com_comp = comp[i].split("\t")

                paraphrase_set = ParaphraseSet(source_comp[0] + "-" + str(hash(com_comp[0])))

                paraphrase_set.add_text(Text(source_comp[2]))
                paraphrase_set.add_text(Text(com_comp[0]))

                yield paraphrase_set
