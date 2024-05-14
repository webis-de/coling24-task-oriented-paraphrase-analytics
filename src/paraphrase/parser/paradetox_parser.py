from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import os


class ParadetoxParser(Parser):
    def get_next(self):
        first_line = True

        for line in self.in_file:
            if first_line:
                first_line = False
                continue

            comp = line.split("\t")

            for i in range(1, len(comp)):
                paraphrase_set = ParaphraseSet(str(hash(str(self.in_file.filelineno()) + "-" + str(i))))

                paraphrase_set.add_text(Text(comp[0]))
                paraphrase_set.add_text(Text(comp[i]))

                yield paraphrase_set
