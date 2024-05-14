from metrics.metric import Metric
from util.nlp import BerkleyConstituencyParser, SpacyParser
import fkassim.FastKassim as fkassim


class FastKASSIM(Metric):

    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["fastkassim"])

        self.fastkassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
        self.parser = BerkleyConstituencyParser()

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)
        for i in range(len(paraphrase_set.texts)):
            for j in range(i + 1, len(paraphrase_set.texts)):
                sim = self.fastkassim.compute_similarity_preparsed(
                    self.parser.parse(paraphrase_set.texts[i].content),
                    self.parser.parse(paraphrase_set.texts[j].content))

                paraphrase_set.metrics["fastkassim"].append(sim)


class POSFeature(Metric):

    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["pos"])

        self.parser = SpacyParser()

    def compute(self, paraphrase_set):
        for i in range(len(paraphrase_set.texts)):
            paraphrase_set.texts[i].pos = self.parser.pos(paraphrase_set.texts[i].content)

