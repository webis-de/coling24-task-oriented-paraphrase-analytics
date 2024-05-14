import os.path
import pandas as pd

import config


def main():
    dataset_conf = config.load_config(config.DATASET_CONF_PATH)

    for dataset in dataset_conf["datasets"]:
        if dataset["name"] == "wikipedia-ipc-gold" or dataset["name"] == "wikipedia-ipc-silver":
            file_paths = [os.path.join(dataset["path"], file_name) for file_name in dataset["files"]]

            for file_path in file_paths:
                with open(file_path) as in_file:
                    df = pd.read_json(in_file, lines=True, orient="records")
                    df["cluster_id"] = df.apply(lambda x: hash(x["image_url"]), axis=1)

                    df = df.sort_values(by="cluster_id")
                    with open(os.path.join(dataset["path"], f"{dataset['name']}-sorted.jsonl"), "w+") as out_file:
                        df.to_json(out_file, lines=True, orient="records")


if __name__ == '__main__':
    main()
