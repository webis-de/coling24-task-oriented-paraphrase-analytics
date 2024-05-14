from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from metrics.metric import Metric


class SentenceBERT(Metric):
    def __init__(self, metric_conf):
        super().__init__(metric_conf)
        self.sentence_transformer = SentenceTransformer("paraphrase-mpnet-base-v2")

    def compute(self, paraphrase_set):
        super().compute(paraphrase_set)

        for i in range(len(paraphrase_set.texts)):
            for j in range(i + 1, len(paraphrase_set.texts)):
                text_i = paraphrase_set.texts[i].content
                text_j = paraphrase_set.texts[j].content
                sentences = [text_i, text_j]
                embeddings = self.sentence_transformer.encode(sentences, show_progress_bar=False, convert_to_tensor=True)

                paraphrase_set.metrics["sentence_bert"].append(cos_sim(embeddings[0], embeddings[1])[0][0].item())


