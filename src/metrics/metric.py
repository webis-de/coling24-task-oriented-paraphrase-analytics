import abc


class Metric(metaclass=abc.ABCMeta):
    def __init__(self, metric_conf, metric_names=None):
        self.metric_conf = metric_conf
        if metric_names is None:
            metric_names = [metric_conf["name"]]

        self.metric_names = metric_names

    @abc.abstractmethod
    def compute(self, paraphrase_set):
        for metric in self.metric_names:
            if metric not in paraphrase_set.metrics:
                paraphrase_set.metrics[metric] = []
