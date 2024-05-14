import os
import joblib
import numpy

import numpy as np

import config
from paraphrase.paraphrase_set import ParaphraseSet
from train_classifier import get_feature_vector, construct_pos_string


def main():
    batch_size = 20000

    corpus_counts = {}

    with open(os.path.join(config.MODEL_DIR, "task-classifier.joblib"), "rb") as in_file:
        cls = joblib.load(in_file)

    with open(os.path.join(config.MODEL_DIR, "pos-encoder.joblib"), "rb") as in_file:
        pos_encoder = joblib.load(in_file)

    for file_name in os.listdir(config.CORPORA_DIR):
        in_path = os.path.join(config.CORPORA_DIR, file_name)
        if os.path.isdir(in_path):
            continue

        corpus_name = os.path.basename(file_name).replace(".jsonl", "")
        features = []
        pos_strs = []
        predicted_tasks = numpy.array([])
        examples = []
        with open(in_path) as in_file:
            for line in in_file:
                paraphrase_set = ParaphraseSet.from_json(line)
                task_name = paraphrase_set.meta["task"]

                if task_name != "general":
                    break

                examples.append(paraphrase_set)
                features.append(get_feature_vector(paraphrase_set))
                pos_strs.append(construct_pos_string(paraphrase_set))

                if len(features) >= batch_size:
                    features = np.array(features)
                    pos_counts = pos_encoder.transform(np.array(pos_strs)).toarray()

                    features = np.concatenate((features, pos_counts), axis=1)

                    predicted_tasks = np.concatenate((predicted_tasks, cls.predict(features)))

                    features = []
                    pos_strs = []

        if len(features) == 0:
            continue

        features = np.array(features)
        pos_counts = pos_encoder.transform(np.array(pos_strs)).toarray()

        features = np.concatenate((features, pos_counts), axis=1)

        predicted_tasks = np.concatenate((predicted_tasks, cls.predict(features)))

        unique, counts = np.unique(predicted_tasks, return_counts=True)
        task_counts = dict(zip(unique, counts))

        corpus_counts[corpus_name] = task_counts

    print_header = True
    for corpus_name in sorted(corpus_counts.keys()):
        tasks = sorted(corpus_counts[corpus_name].keys())
        if print_header:
            header = f"{'Dataset':>20s}"
            for task in tasks:
                header += f" {task:14s}"

            print(header)
            print_header = False

        corpus_counts[corpus_name]["total"] = sum(corpus_counts[corpus_name].values())
        out_str = f"{corpus_name:>20s}"

        for task in tasks:
            out_str = (f"{out_str} {corpus_counts[corpus_name][task] / corpus_counts[corpus_name]['total'] * 100:>5.1f}% "
                       f"{corpus_counts[corpus_name][task]:>6d}")

        print(out_str)


if __name__ == '__main__':
    main()
