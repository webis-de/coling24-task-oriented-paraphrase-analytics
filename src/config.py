import os
import yaml

DATASET_CONF_PATH = os.path.join(os.path.dirname(__file__), "../datasets.yml")

METRIC_CONF_PATH = os.path.join(os.path.dirname(__file__), "../metrics.yml")

OUT_DIR = os.path.join(os.path.dirname(__file__), "../out/")

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "../data/annotations/")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "../data/corpora/samples/")

CORPORA_DIR = os.path.join(os.path.dirname(__file__), "../data/corpora/")

FIG_DIR = os.path.join(os.path.dirname(__file__), "../data/figures/")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/cache/")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../data/models/")


def load_config(file_path):
    with open(file_path) as conf_file:
        conf = yaml.safe_load(conf_file)

    if conf is None:
        raise IOError(f"Error parsing config \"{file_path}\"")

    return conf
