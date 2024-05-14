import json
import os

from paraphrase.paraphrase_set import ParaphraseSet, Text
from paraphrase.parser.parser import Parser


class WikipediaIPCParser(Parser):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def get_next(self):
        for line in self.in_file:
            example = json.loads(line)

            paraphrase_set = ParaphraseSet(
                example["id"],
                cluster_id=example["cluster_id"],
                image_url=example["image_url"]
            )

            for text in example["paraphrase"]:
                text_obj = Text(text["content"], reference=text["reference"])

                paraphrase_set.add_text(
                    text_obj
                )

            yield paraphrase_set
