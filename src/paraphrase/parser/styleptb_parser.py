from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import os


class StylePTBParser(Parser):
    def get_next(self):
        for line in self.in_file:
            comp = line.split("\t")

            type = os.path.basename(os.path.dirname(self.in_file.filename()))
            paraphrase_set = ParaphraseSet(str(hash(type + "-" + str(self.in_file.filelineno()))), type=type)
            paraphrase_set.add_text(Text(comp[0]))
            paraphrase_set.add_text(Text(comp[1]))

            yield paraphrase_set
