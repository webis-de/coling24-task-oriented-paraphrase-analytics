import csv

from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text


class QuoraParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def get_next(self):
        reader = csv.reader(self.in_file, delimiter=",", doublequote=True)
        for row in reader:
            if row[0].startswith("id") or row[0].startswith("test_id"):
                continue

            if self.in_file.filename().startswith("train"):
                paraphrase_set = ParaphraseSet(f"train-{row[0]}",
                                               quality="1" == row[5])

                paraphrase_set.add_text(Text(row[1], row[3]))
                paraphrase_set.add_text(Text(row[2], row[4]))

            else:
                paraphrase_set = ParaphraseSet(f"test-{row[0]}")

                paraphrase_set.add_text(Text(row[1]))
                paraphrase_set.add_text(Text(row[2]))

            yield paraphrase_set
