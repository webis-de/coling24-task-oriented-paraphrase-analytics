from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text


class TurkCorpusParser(Parser):
    def get_next(self):
        for line in self.in_file:
            comp = line.split("\t")

            text_1 = Text(comp[1])
            for i in range(1, len(comp)):
                if " ".join(comp[i].split()) == " ".join(comp[1].split()):
                    continue

                paraphrase_set = ParaphraseSet(comp[0])
                paraphrase_set.add_text(text_1)
                paraphrase_set.add_text(Text(comp[i]))

                yield paraphrase_set
