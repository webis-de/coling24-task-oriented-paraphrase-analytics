from abc import ABC, ABCMeta
from metrics.metric import Metric

import textstat


class ReadabilityStat(Metric, metaclass=ABCMeta):

    def __init__(self, metric_conf):
        super().__init__(metric_conf, [metric_conf["name"]])

    def _compute(self, paraphrase_set, method):
        metric = [method(text.content) for text in paraphrase_set.texts]

        paraphrase_set.metrics[self.metric_conf["name"]] = metric


class FleschReadingEase(ReadabilityStat):
    def compute(self, paraphrase_set):
        super()._compute(paraphrase_set, textstat.flesch_reading_ease)


class FleschKincaidGradeLevel(ReadabilityStat):
    def compute(self, paraphrase_set):
        super()._compute(paraphrase_set, textstat.flesch_kincaid_grade)


class FogScale(ReadabilityStat):
    def compute(self, paraphrase_set):
        super()._compute(paraphrase_set, textstat.gunning_fog)


class ARI(ReadabilityStat):
    def compute(self, paraphrase_set):
        super()._compute(paraphrase_set, textstat.automated_readability_index)
