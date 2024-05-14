import json
import os
import random

import config
import csv


def main():
    data_array = []

    for file_name in os.listdir(config.SAMPLE_DIR):
        in_path = os.path.join(config.SAMPLE_DIR, file_name)
        if os.path.isdir(in_path):
            continue

        with open(in_path, "r") as in_file:
            for line in in_file:
                data = json.loads(line)
                data_array.append(data)

    random.shuffle(data_array)

    out_annotations = os.path.join(config.OUT_DIR, "annotations/annotations.csv")
    out_labels = os.path.join(config.OUT_DIR, "annotations/labels.csv")
    with open(out_annotations, "w+", newline="") as annotation_out_file:
        with open(out_labels, "w+", newline="") as label_out_file:
            anno_writer = csv.writer(annotation_out_file, quoting=csv.QUOTE_ALL, dialect="excel")
            label_writer = csv.writer(label_out_file, quoting=csv.QUOTE_ALL, dialect="excel")

            for data in data_array:
                anno_writer.writerow([data["texts"][0]["content"].lower(), data["texts"][1]["content"].lower()])
                label_writer.writerow([data["_id"], data["meta"]["task"]])


if __name__ == '__main__':
    main()
