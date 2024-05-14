import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt

import config


def main():
    anno_path = os.path.join(config.ANNOTATION_DIR, "annotations-run-23-06-21.csv")
    label_path = os.path.join(config.ANNOTATION_DIR, "labels-run-23-06-21.csv")

    confusion = {}
    des_labels = set()
    anno_labels = set()

    with open(anno_path, "r") as anno_file:
        with open(label_path, "r") as label_file:
            anno_reader = csv.reader(anno_file)
            label_reader = csv.reader(label_file)

            for anno, label in zip(anno_reader, label_reader):
                des_label = label[1].replace("-", "\n")
                des_labels.add(des_label)
                if des_label not in confusion:
                    confusion[des_label] = {}

                anno_label = anno[2]
                if anno_label == "identity":
                    continue

                anno_label = anno_label.replace("-", "\n")
                anno_labels.add(anno_label)
                if anno_label not in confusion[des_label]:
                    confusion[des_label][anno_label] = 0

                confusion[des_label][anno_label] += 1

    heat_map = []
    for des_label in sorted(des_labels):
        row = []
        for anno_label in sorted(anno_labels):
            if anno_label not in confusion[des_label]:
                confusion[des_label][anno_label] = 0

            row.append(confusion[des_label][anno_label])

        heat_map.append(row)

    heat_map = np.array(heat_map)

    font = {
            'weight': 'normal',
            'size': 12}
    plt.rc("font", **font)
    fig, ax = plt.subplots()
    im = ax.imshow(heat_map, cmap="bone_r")

    ax.set_xticks(np.arange(len(anno_labels)), labels=sorted(anno_labels))
    ax.set_yticks(np.arange(len(des_labels)), labels=sorted(des_labels))

    plt.xlabel("Annotated task", fontdict={"weight": "bold"})
    plt.ylabel("Actual task", fontdict={"weight": "bold"})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", font={"size": 10})
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", font={"size": 10})

    for i in range(len(des_labels)):
        for j in range(len(anno_labels)):
            if heat_map[i, j] >= 50:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, f"{heat_map[i, j] / 100.0:.2f}",
                           ha="center", va="center", color=color)
    fig.tight_layout()
    plt.rcParams["pdf.fonttype"] = "truetype"
    plt.savefig(os.path.join(config.FIG_DIR, "task-annotation-confusion.pdf"))
    plt.show()


if __name__ == '__main__':
    main()
