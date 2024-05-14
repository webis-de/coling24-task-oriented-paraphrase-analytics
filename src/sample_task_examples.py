import os
import json
import random

import matplotlib.pyplot as plt
import numpy as np

import config

from sklearn.model_selection import train_test_split

from paraphrase.paraphrase_set import ParaphraseSet


def sample(x, y, n, stratify_dist):
    return train_test_split(x, train_size=n, stratify=stratify_dist)[0]


def main():
    min_length = 0
    max_length = 250
    n_bins = 10
    n_samples = 10000
    task_examples = {}

    for file_name in os.listdir(config.OUT_DIR):
        in_path = os.path.join(config.OUT_DIR, file_name)
        if os.path.isdir(in_path):
            continue

        with open(in_path) as in_file:
            corpus_examples = []
            for line in in_file:
                paraphrase_set = ParaphraseSet.from_json(line)
                task_name = paraphrase_set.meta["task"]
                corpus_name = file_name.replace(".jsonl", "")
                paraphrase_set.meta["corpus"] = corpus_name

                if task_name == "general":
                    break

                if task_name not in task_examples:
                    task_examples[task_name] = []

                length = sum(paraphrase_set.metrics["length(chars)"])
                first_content = paraphrase_set.texts[0]["content"].lower().replace(" ", "")
                second_content = paraphrase_set.texts[1]["content"].lower().replace(" ", "")
                if min_length <= length <= max_length and first_content != second_content:
                    paraphrase_set.metrics["length(sum)"] = length
                    corpus_examples.append(paraphrase_set)

            if task_name not in task_examples:
                continue

            if len(corpus_examples) == 0:
                continue

            task_examples[task_name].append(corpus_examples)

    bins = np.linspace(min_length, max_length, n_bins + 1)
    data = []
    labels = []

    task_example_sample = {}

    for task in task_examples:
        if task not in task_example_sample:
            task_example_sample[task] = []

        for corpus_i in range(len(task_examples[task])):
            print(f"Task \"{task}\": {len(task_examples[task][corpus_i])}")
            if len(task_examples[task][corpus_i]) < n_samples / len(task_examples[task]):
                raise KeyError(f"Not enough samples matching conditions for task \"{task}\"")

            len_distribution = [x.metrics["length(sum)"] for x in task_examples[task][corpus_i]]
            indices = np.digitize(len_distribution, bins)

            binned_indices = {}
            for i in range(n_bins + 1):
                n = np.where(indices == i)[0]
                if len(n) > 0:
                    binned_indices[i] = list(n)

            bin_indices = list(binned_indices.keys())
            while len(task_example_sample[task]) < (corpus_i + 1) * (n_samples // len(task_examples[task])):
                bin = np.random.choice(bin_indices)

                bin_index = random.choice(range(len(binned_indices[bin])))
                example_index = binned_indices[bin][bin_index]

                task_example_sample[task].append(task_examples[task][corpus_i][example_index])
                del binned_indices[bin][bin_index]
                if len(binned_indices[bin]) == 0:
                    del binned_indices[bin]
                    bin_indices.remove(bin)




        sample_len_distribution = [x.metrics["length(sum)"] for x in task_example_sample[task]]
        data.append(sample_len_distribution)
        labels.append(task)

        with open(os.path.join(config.OUT_DIR, f"samples/{task}.jsonl"), "w+") as out_file:
            random.shuffle(task_example_sample[task])
            for paraphrase_set in task_example_sample[task]:
                out_file.write(ParaphraseSet.to_json(paraphrase_set))
                out_file.write("\n")

    plt.figure()
    plt.hist(data, bins, histtype="bar", stacked=False, label=labels, rwidth=0.7)
    plt.xticks(bins)
    plt.xlabel("Length (chars)")
    plt.ylabel("Number of paraphrase pairs")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
