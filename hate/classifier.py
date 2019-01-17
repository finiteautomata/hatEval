from sentiment.classifier import SentimentClassifier
from hate.tokenizer import Tokenizer

from sklearn import metrics


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

    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        print('accuracy\t{:2.2f}\n'.format(acc))
        print(metrics.classification_report(y_test, y_pred))
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)
