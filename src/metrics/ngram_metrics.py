import abc
from abc import ABCMeta
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer

from metrics.metric import Metric
from util.nlp import Tokenizer


class NGramMetric(Metric, metaclass=ABCMeta):
    def __init__(self, metric_conf):
        super().__init__(metric_conf, [metric_conf["name"]])

    def compute(self, paraphrase_set):
        scores = []
        for i in range(len(paraphrase_set.texts)):
            for j in range(i + 1, len(paraphrase_set.texts)):
                scores.append(
                    self._compute(paraphrase_set.texts[i].tokens,
                                  [paraphrase_set.texts[j].tokens])
                )

        paraphrase_set.metrics[self.metric_conf["name"]] = scores

    @abc.abstractmethod
    def _compute(self, tokens, reference_tokens):
        pass


class Bleu(NGramMetric):
    def __init__(self, metric_conf):
        super().__init__(metric_conf)
        self.smoothing = SmoothingFunction()

    def _compute(self, tokens, reference_tokens):
        try:
            return sentence_bleu(reference_tokens, tokens, smoothing_function=self.smoothing.method7)
        except KeyError as e:
            print(reference_tokens, tokens)
            raise e


class NIST(NGramMetric):
    def _compute(self, tokens, reference_tokens):
        try:
            return sentence_nist(reference_tokens, tokens)
        except ZeroDivisionError:
            return 0


class Rouge(Metric):

    def __init__(self, metric_conf):
        super().__init__(metric_conf, ["rouge1_prec", "rouge1_rec", "rouge1_f1", "rouge2_prec", "rouge2_rec", "rouge2_f1", "rougeL_prec", "rougeL_rec", "rougeL_f1",])
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=Tokenizer)

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)

        for i in range(len(paraphrase_set.texts)):
            for j in range(i + 1, len(paraphrase_set.texts)):
                rouge_scores = self.rouge.score(paraphrase_set.texts[j].content.lower(), paraphrase_set.texts[i].content.lower())

                paraphrase_set.metrics["rouge1_prec"].append(rouge_scores["rouge1"].precision)
                paraphrase_set.metrics["rouge1_rec"].append(rouge_scores["rouge1"].recall)
                paraphrase_set.metrics["rouge1_f1"].append(rouge_scores["rouge1"].fmeasure)

                paraphrase_set.metrics["rouge2_prec"].append(rouge_scores["rouge2"].precision)
                paraphrase_set.metrics["rouge2_rec"].append(rouge_scores["rouge2"].recall)
                paraphrase_set.metrics["rouge2_f1"].append(rouge_scores["rouge2"].fmeasure)

                paraphrase_set.metrics["rougeL_prec"].append(rouge_scores["rougeL"].precision)
                paraphrase_set.metrics["rougeL_rec"].append(rouge_scores["rougeL"].recall)
                paraphrase_set.metrics["rougeL_f1"].append(rouge_scores["rougeL"].fmeasure)
