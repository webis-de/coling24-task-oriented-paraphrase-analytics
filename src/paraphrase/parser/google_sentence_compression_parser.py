import json

from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text


class GoogleSentenceCompressionParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def get_next(self):
        obj_string = ""
        for line in self.in_file:
            if line.strip() == "":
                data = json.loads(obj_string)
                content = data["graph"]["sentence"]
                paraphrase_set = ParaphraseSet(str(hash(content)))

                paraphrase_set.add_text(Text(content))
                paraphrase_set.add_text(Text(data["compression"]["text"]))
                yield paraphrase_set

                paraphrase_set = ParaphraseSet(str(hash(content)))
                paraphrase_set.add_text(Text(content))
                paraphrase_set.add_text(Text(data["compression_untransformed"]["text"]))

                yield paraphrase_set
                obj_string = ""

            obj_string += line

