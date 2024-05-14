import json
from util.nlp import tokenize


class Text:
    def __init__(self, content: str, _id: str = None, **kwargs):
        if _id is None:
            self._id = str(hash(content))
        self.content = content
        self.tokens = tokenize(content.lower())
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return f"\"{self.content}\""

    def __eq__(self, other):
        return self.content == other.content

    def __hash__(self):
        return hash(self.content)


class GeneratedText(Text):
    def __init__(self, content: str, source, _id: str = None, **kwargs):
        super().__init__(content, _id=_id, **kwargs)
        self.source = source


class ParaphraseSet:

    def __init__(self, set_id: str, **kwargs):
        self._id = set_id
        self.texts = []
        self.metrics = {}
        self.meta = kwargs

    def add_text(self, text: Text) -> None:
        self.texts.append(text)

    def __str__(self) -> str:
        return "[" + ",".join([str(x) for x in self.texts]) + "]\n" + str(self.metrics) + "\n"

    def to_json(self):
        return json.dumps(self, default=self.default)

    def to_pairs(self):
        for i in range(len(self.texts)):
            for j in range(i + 1, len(self.texts)):
                pair = ParaphraseSet(self._id, **self.meta)
                pair.texts = [self.texts[i], self.texts[j]]
                yield pair

    @staticmethod
    def default(o):
        return o.__dict__

    @staticmethod
    def from_json(text):
        paraphrase_set = ParaphraseSet("")
        paraphrase_set.__dict__ = json.loads(text)
        return paraphrase_set
