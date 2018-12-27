from sentiment.classifier import SentimentClassifier
from hate.tokenizer import Tokenizer


class HateClassifier(SentimentClassifier):

    def __init__(self, lang='en', **kwargs):
        self._lang = lang
        super().__init__(**kwargs)

    def build_bow_tokenizer(self):
        return Tokenizer(lang=self._lang, rdup=True)

    def build_emb_tokenizer(self):
        return Tokenizer(lang=self._lang, lem=False)
