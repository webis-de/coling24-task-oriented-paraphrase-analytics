import logging
import os

import config
from compute_features import get_class
from paraphrase.generator.parrot_generator import ParrotGenerator
from paraphrase.paraphrase_set import GeneratedText

def main():
    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

    dataset_conf = config.load_config(config.DATASET_CONF_PATH)
    generator = ParrotGenerator()

    for dataset in dataset_conf["datasets"]:
        logging.info(f"Process dataset \"{dataset['name']}\"")
        out_path = os.path.join(os.path.dirname(__file__), "../out/" + dataset["name"] + "-parrot.jsonl")
        files = [os.path.join(dataset["path"], file_name) for file_name in dataset["files"]]
        parser_class = get_class(dataset["parser"])
        parser_instance = parser_class(files)

        with open(out_path, "w+") as out_file:
            for paraphrase_set in parser_instance.get_next():
                orig_length = len(paraphrase_set.texts)
                for i in range(orig_length):
                    paraphrases = generator.generate(paraphrase_set.texts[i].content)
                    for paraphrase in paraphrases:
                        paraphrase_set.add_texts(GeneratedText())

                out_file.write(paraphrase_set.to_json() + "\n")


if __name__ == '__main__':
    main()
