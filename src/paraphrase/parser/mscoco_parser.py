from paraphrase.parser.parser import Parser
from paraphrase.paraphrase_set import ParaphraseSet, Text
import json


class MSCocoParser(Parser):
    def get_next(self):
        for line in self.in_file:

            data = json.loads(line)

            image_groups = {}
            for annotation in data["annotations"]:
                if annotation["image_id"] not in image_groups:
                    image_groups[annotation["image_id"]] = []

                image_groups[annotation["image_id"]].append(annotation)

            for image_id in image_groups:
                for i in range(len(image_groups[image_id])):
                    for j in range(i+ 1, len(image_groups[image_id])):
                        first = image_groups[image_id][i]
                        second = image_groups[image_id][j]

                        paraphrase_set = ParaphraseSet(str(first["id"]) + "-" + str(second["id"]))

                        paraphrase_set.add_text(Text(first["caption"]))
                        paraphrase_set.add_text(Text(second["caption"]))
                        yield paraphrase_set
