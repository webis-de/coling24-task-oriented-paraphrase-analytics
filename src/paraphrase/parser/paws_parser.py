from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text


class PAWSParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def get_next(self):
        for line in self.in_file:

            if line.startswith("id"):
                continue

            comp = line.split("\t")
            paraphrase_set = ParaphraseSet(self.in_file.filename() + "-" + comp[0],
                                           quality="1" == comp[3].strip())

            paraphrase_set.add_text(Text(comp[1]))
            paraphrase_set.add_text(Text(comp[2]))

            yield paraphrase_set
