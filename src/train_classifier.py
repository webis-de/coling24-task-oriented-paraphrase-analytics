import os
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

import config
from paraphrase.paraphrase_set import ParaphraseSet
import matplotlib.pyplot as plt
import joblib


def construct_pos_string(paraphrase_set):
    pos = paraphrase_set.texts[0]["pos"]
    pos_str = f"<bos> {' '.join(pos)} <eos>"
    pos = paraphrase_set.texts[1]["pos"]
    pos_str += " <bos> " + " ".join(pos) + " <eos>"

    return pos_str


def construct_pos_vectorizer():
    pos_sequences_str = []
    samples_dir = config.SAMPLE_DIR
    count = TfidfVectorizer(use_idf=False, norm="l1", ngram_range=(1, 4))

    for file_name in os.listdir(samples_dir):
        in_path = os.path.join(samples_dir, file_name)
        if os.path.isdir(in_path):
            continue

        with open(in_path) as in_file:
            for line in in_file:
                paraphrase_set = ParaphraseSet.from_json(line)
                pos_str = construct_pos_string(paraphrase_set)
                pos_sequences_str.append(pos_str)

    count.fit(pos_sequences_str)
    return count


def get_feature_vector(paraphrase_set):
    feature_set = []
    feature_set.extend(paraphrase_set.metrics["compression_ratio"])
    feature_set.extend([paraphrase_set.metrics["rouge1_prec"][0], paraphrase_set.metrics["rouge1_rec"][0],
                        paraphrase_set.metrics["rouge1_f1"][0]])
    feature_set.extend(paraphrase_set.metrics["bleu"])
    feature_set.extend(paraphrase_set.metrics["sentence_bert"])

    return feature_set


def main():
    samples_dir = config.SAMPLE_DIR

    examples_train = []
    examples_test = []

    features_train = []
    labels_train = []

    features_test = []
    labels_test = []

    unique_labels = []

    pos_vectorizer = construct_pos_vectorizer()

    for file_name in os.listdir(samples_dir):
        in_path = os.path.join(samples_dir, file_name)
        if os.path.isdir(in_path):
            continue

        pos_sequences_str = []
        examples = []
        features = []
        labels = []

        with open(in_path) as in_file:
            for line in in_file:
                paraphrase_set = ParaphraseSet.from_json(line)
                labels.append(paraphrase_set.meta["task"])

                if paraphrase_set.meta["task"] not in unique_labels:
                    unique_labels.append(paraphrase_set.meta["task"])

                pos_str = construct_pos_string(paraphrase_set)
                pos_sequences_str.append(pos_str)

                examples.append(paraphrase_set)
                features.append(get_feature_vector(paraphrase_set))

        features = np.array(features)
        pos_counts = pos_vectorizer.transform(pos_sequences_str).toarray()
        features = np.concatenate((features, pos_counts), axis=1)
        x_train, x_test, y_train, y_test, ex_train, ex_test = train_test_split(features, labels, examples,
                                                                               test_size=0.2)

        features_train.extend(x_train)
        features_test.extend(x_test)
        labels_train.extend(y_train)
        labels_test.extend(y_test)
        examples_train.extend(ex_train)
        examples_test.extend(ex_test)

    shuffle_collection = list(zip(features_train, labels_train, examples_train))
    random.shuffle(shuffle_collection)
    features_train, labels_train, examples_train = zip(*shuffle_collection)

    models_dir = config.MODEL_DIR
    with open(os.path.join(models_dir, "pos-encoder.joblib"), "wb+") as out_file:
        joblib.dump(pos_vectorizer, out_file)

    cls = RandomForestClassifier(max_depth=15)

    cls.fit(features_train, labels_train)

    with open(os.path.join(models_dir, "task-classifier.joblib"), "wb+") as out_file:
        joblib.dump(cls, out_file)

    shuffle_collection = list(zip(features_test, labels_test, examples_test))
    random.shuffle(shuffle_collection)
    features_test, labels_test, examples_test = zip(*shuffle_collection)

    label_predict = cls.predict(features_test)

    prec_micro = precision_score(labels_test, label_predict, average="micro")
    rec_micro = recall_score(labels_test, label_predict, average="micro")
    f1_micro = f1_score(labels_test, label_predict, average="micro")

    print(f"F1(micro): {f1_micro:.4f} Prec(micro): {prec_micro:.4f} Rec(micro): {rec_micro:.4f}")

    cm = ConfusionMatrixDisplay.from_predictions(labels_test, label_predict, labels=sorted(unique_labels),
                                                 cmap="bone_r", colorbar=False, normalize="true", values_format=".2f")
    ax = cm.ax_
    ax.set_xticks(np.arange(len(ax.get_xticklabels())), labels=sorted([x.replace("-", "\n") for x in sorted(unique_labels)]))
    ax.set_yticks(np.arange(len(ax.get_yticklabels())),
                  labels=sorted([x.replace("-", "\n") for x in sorted(unique_labels)]))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", font={"size": 10})
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", font={"size": 10})
    plt.ylabel("Actual task", fontdict={"weight": "bold"})
    plt.xlabel("Predicted task", fontdict={"weight": "bold"})
    plt.tight_layout()
    plt.rcParams["pdf.fonttype"] = "truetype"
    plt.savefig(os.path.join(config.FIG_DIR, "task-prediction-confusion.pdf"), format="pdf", transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
