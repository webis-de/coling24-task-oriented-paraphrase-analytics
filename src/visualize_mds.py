import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder

import config
from paraphrase.paraphrase_set import ParaphraseSet


def main():
    samples_dir = os.path.join(config.OUT_DIR, "samples")
    sample_size = 1000
    features = []
    examples = []
    labels = []
    unique_labels = []
    pos_sequences_first = []
    max_pos_length = 0
    pos_sequences_second = []
    pos_sequences_str = []

    max_word_freq_length = 0
    word_freq_seq = []

    for file_name in os.listdir(samples_dir):
        in_path = os.path.join(samples_dir, file_name)
        if os.path.isdir(in_path):
            continue

        with open(in_path) as in_file:
            for line in in_file:
                paraphrase_set = ParaphraseSet.from_json(line)
                labels.append(paraphrase_set.meta["task"])

                if paraphrase_set.meta["task"] not in unique_labels:
                    unique_labels.append(paraphrase_set.meta["task"])

                word_freq = [x for x in paraphrase_set.metrics["word_frequency_classes"] if x is not None]
                word_freq_seq.append(word_freq)

                if len(word_freq) > max_word_freq_length:
                    max_word_freq_length = len(word_freq)
                # num_oov = len(word_freq[word_freq == np.array(None)])
                # word_freq = word_freq[word_freq != np.array(None)]

                pos = paraphrase_set.texts[0]["pos"]
                pos_str = "<bos> " + " ".join(pos) + " <eos>"
                if len(pos) > max_pos_length:
                    max_pos_length = len(pos)
                pos = paraphrase_set.texts[1]["pos"]
                pos_str += " <bos> " + " ".join(pos) + " <eos>"
                if len(pos) > max_pos_length:
                    max_pos_length = len(pos)

                pos_sequences_first.append(paraphrase_set.texts[0]["pos"])
                pos_sequences_second.append(paraphrase_set.texts[1]["pos"])
                pos_sequences_str.append(pos_str)

                feature_set = []
                feature_set.extend(paraphrase_set.metrics["compression_ratio"])
                feature_set.extend([paraphrase_set.metrics["rouge1_prec"][0], paraphrase_set.metrics["rouge1_rec"][0],
                                    paraphrase_set.metrics["rouge1_f1"][0]])

                feature_set.extend(paraphrase_set.metrics["bleu"])
                feature_set.extend(paraphrase_set.metrics["sentence_bert"])

                examples.append(paraphrase_set)
                features.append(feature_set)

    for pos_sequence in pos_sequences_first:
        pos_sequence += [''] * (max_pos_length - len(pos_sequence))

    for pos_sequence in pos_sequences_second:
        pos_sequence += [''] * (max_pos_length - len(pos_sequence))

    for freq in word_freq_seq:
        freq += [-1] * (max_word_freq_length - len(freq))

    # pos_sequences_first = np.array(pos_sequences_first)
    # pos_sequences_second = np.array(pos_sequences_second)
    # stack = np.concatenate((pos_sequences_first, pos_sequences_second), axis=0)

    count = TfidfVectorizer(use_idf=False, norm="l1", ngram_range=(1, 4))
    pos_count = count.fit_transform(pos_sequences_str).toarray()

    features = np.array(features)

    features = np.concatenate((features, pos_count), axis=1)

    embedding = MDS(normalized_stress="auto", metric=True)
    transformed = embedding.fit_transform(features)

    plt.figure()

    total_dist = 0
    dis_matrix = embedding.dissimilarity_matrix_

    for i in range(len(unique_labels)):
        for j in range(i, len(labels)):
            cluster_dis_matrix = dis_matrix[i * sample_size:(i + 1) * sample_size,
                                 j * sample_size:(j + 1) * sample_size]
            distances = cluster_dis_matrix[np.tril_indices(sample_size, k=-1)]
            avg_dist = np.mean(distances)

            if i == j:
                total_dist -= avg_dist
            else:
                total_dist += avg_dist
            print(f"Avg. distance:  {labels[i]} -> {labels[j]}: {avg_dist:.4f}")

        plt.plot(transformed[i * sample_size:(i + 1) * sample_size, 0],
                 transformed[i * sample_size:(i + 1) * sample_size, 1],
                 marker="o", linestyle="", label=unique_labels[i], alpha=0.7)

    print(f"\nTotal score: {total_dist}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
