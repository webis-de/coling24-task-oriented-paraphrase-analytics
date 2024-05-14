from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import re


class BibleStyleParser(Parser):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        super().__init__(None)

    def get_next(self):
        for file_set in self.file_paths:
            file_set_done = False
            file_streams = []

            for file in file_set:
                file_streams.append(open(file, "r"))

            while not file_set_done:
                lines = []
                for stream in file_streams:
                    line = stream.readline()
                    if line == "":
                        for stream in file_streams:
                            stream.close()

                        file_set_done = True
                        break

                    lines.append(re.sub("^<[A-Z]+> ", "", line))

                if len(lines) > 0:
                    for i in range(1, len(lines)):
                        paraphrase_set = ParaphraseSet(str(hash(lines[i])))

                        if len(lines[0].split()) == 0 or len(lines[i].split()) == 0:
                            continue

                        paraphrase_set.add_text(Text(lines[0]))
                        paraphrase_set.add_text(Text(lines[i]))

                        yield paraphrase_set
