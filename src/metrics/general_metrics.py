from abc import ABCMeta
from nltk.tokenize import word_tokenize

from metrics.metric import Metric


class GeneralMetric(Metric, metaclass=ABCMeta):
    pass


class Length(GeneralMetric):
    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["length(chars)", "length(words)", "length(distinct words)"])

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)

        for text in paraphrase_set.texts:
            paraphrase_set.metrics["length(chars)"].append(len(text.content))
            paraphrase_set.metrics["length(words)"].append(len(text.tokens))
            paraphrase_set.metrics["length(distinct words)"].append(len(set(text.tokens)))


class SetSize(GeneralMetric):
    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["set_size"])

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)

        paraphrase_set.metrics["set_size"].append(len(paraphrase_set.texts))


class CompressionRatio(GeneralMetric):
    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["compression_ratio"])

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)

        for i in range(len(paraphrase_set.texts)):
            for j in range(i + 1, len(paraphrase_set.texts)):
                len_i = len(paraphrase_set.texts[i].content)
                len_j = len(paraphrase_set.texts[j].content)
                paraphrase_set.metrics["compression_ratio"].append(min(len_i, len_j) / max(len_i, len_j))

