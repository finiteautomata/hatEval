from sentiment.classifier import SentimentClassifier
from hate.tokenizer import Tokenizer


class HateClassifier(SentimentClassifier):

    def __init__(self, lang='en', **kwargs):
        self._lang = lang
        super().__init__(**kwargs)

    def build_bow_tokenizer(self):
        neg = self._lang == 'es'  # only handle negations in spanish
        return Tokenizer(lang=self._lang, rdup=True, neg=neg)

    def build_emb_tokenizer(self):
        neg = self._lang == 'es'  # only handle negations in spanish
        return Tokenizer(lang=self._lang, lem=False, neg=neg)
