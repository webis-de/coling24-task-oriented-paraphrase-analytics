import logging
import os
from nltk.tree import Tree
from contextlib import redirect_stdout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("h5py").setLevel(logging.INFO)

from keras.preprocessing.text import text_to_word_sequence


def tokenize(text):
    return text_to_word_sequence(text)


class Tokenizer:
    @staticmethod
    def tokenize(text):
        return tokenize(text)


class StanzaConstituencyParser:
    def __init__(self):
        import stanza
        self.nlp_c = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency",
                                     verbose=False, logging_level="info",
                                     download_method=stanza.DownloadMethod.REUSE_RESOURCES, use_gpu=False)

        self.nlp_p = stanza.Pipeline(lang="en", processors="tokenize,pos",
                                     verbose=False, logging_level="info",
                                     download_method=stanza.DownloadMethod.REUSE_RESOURCES, use_gpu=True)

    def parse(self, text):
        doc = self.nlp_c(text)
        parse_trees = []
        for sentence in doc.sentences:
            parse_trees.append(Tree.fromstring(str(sentence.constituency)))
        return parse_trees

    def pos(self, text):
        doc = self.nlp_p(text)
        pos = []
        for sentence in doc.sentences:
            for word in sentence.words:
                pos.append(word.xpos)
        return pos


class BerkleyConstituencyParser:
    def __init__(self):
        import benepar
        import spacy
        with redirect_stdout(open(os.devnull, "w")):
            self.nlp = spacy.load("en_core_web_md")
            benepar.download('benepar_en3')
            self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    def parse(self, text):
        doc = self.nlp(text)
        parse_trees = []
        for sentence in doc.sents:
            parse_trees.append(Tree.fromstring(sentence._.parse_string))
        return parse_trees

    def pos(self, text):
        doc = self.nlp(text)
        pos_tags = []
        for token in doc:
            pos_tags.append(token.tag_)

        return pos_tags


class SpacyParser:
    def __init__(self):
        import spacy
        with redirect_stdout(open(os.devnull, "w")):
            self.nlp = spacy.load("en_core_web_md")

    def pos(self, text):
        doc = self.nlp(text)
        pos_tags = []
        for token in doc:
            pos_tags.append(token.tag_)

        return pos_tags
