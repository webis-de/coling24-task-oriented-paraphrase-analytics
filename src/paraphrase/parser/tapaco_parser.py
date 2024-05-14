import os

from paraphrase.paraphrase_set import ParaphraseSet, Text
from paraphrase.parser.parser import Parser


class TaPaCoParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)
        self.c_id = "1"

    def get_next(self):
        tmp = []

        for line in self.in_file:
            values = line.replace("\n", "").split("\t")

            if values[0] != self.c_id:
                for i in range(len(tmp)):
                    for j in range(i + 1, len(tmp)):
                        paraphrase_set = ParaphraseSet(tmp[i][1] + "-" + tmp[j][1])
                        paraphrase_set.add_text(Text(tmp[i][2]))
                        paraphrase_set.add_text(Text(tmp[j][2]))

                        yield paraphrase_set

                tmp = []

            tmp.append(values)
            self.c_id = values[0]

        for i in range(len(tmp)):
            for j in range(i + 1, len(tmp)):
                paraphrase_set = ParaphraseSet(tmp[i][1] + "-" + tmp[j][1])
                paraphrase_set.add_text(Text(tmp[i][2]))
                paraphrase_set.add_text(Text(tmp[j][2]))

                yield paraphrase_set
