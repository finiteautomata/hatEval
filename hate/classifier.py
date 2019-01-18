from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import metrics

from hate.tokenizer import Tokenizer
from hate.embeddings import SentenceVectorizer, WeightedSentenceVectorizer


classifiers = {
    'dt': DecisionTreeClassifier,
    'mnb': MultinomialNB,
    'maxent': LogisticRegression,
    'lrcv': LogisticRegressionCV,
    'svm': LinearSVC,
    'svm2': SVC,
    'rf': RandomForestClassifier,
    'erf': ExtraTreesClassifier,
    'ada': AdaBoostClassifier,
}

default_clf_params = {
    'dt': {'random_state': 0},
    'maxent': {
        'penalty': 'l2',  # 'l1' or 'l2'
        'C': 1.0,  # decrease for more regularization
        'class_weight': 'balanced',
        'random_state': 0,
    },
    'svm': {
        # 'penalty': 'l1',  # 'l1' or 'l2'
        # 'loss': 'squared_hinge',  # 'hinge' or 'squared_hinge'
        # 'dual': False,  # required for l1 penalty
        'C': 0.1,  # decrease for more regularization
        'class_weight': 'balanced',
        'random_state': 0,
    },
    'svm2': {
        # 'penalty': 'l1',  # 'l1' or 'l2'
        # 'loss': 'squared_hinge',  # 'hinge' or 'squared_hinge'
        # 'dual': False,  # required for l1 penalty
        #'C': 0.1,  # decrease for more regularization
        'class_weight': 'balanced',
        'random_state': 0,
    },
    'rf': {
        'random_state': 0,
        'n_estimators': 100,  # [10, 50, 100, 200]
        # 'class_weight': 'balanced',  # [None, 'balanced', 'balanced_subsample']
    },
    'erf': {
        'random_state': 0,
        'n_estimators': 100,  # [10, 50, 100, 200]
        # 'class_weight': 'balanced',  # [None, 'balanced', 'balanced_subsample']
        'bootstrap': True,
        # 'oob_score': True,
    },
    'ada': {
        'base_estimator': LogisticRegression(class_weight='balanced'),
        'n_estimators': 400,
        'random_state': 0,
    },
}

default_bow_params = {
    'binary': True,
    # 'sublinear_tf': True,  # only makes sense with binary=False
    'ngram_range': (1, 2),
}

default_boc_params = {
    'analyzer': 'char',
    'binary': True,
    'ngram_range': (1, 3),
}

default_emb_params = {
    'a': 0.1,
    # 'binarize': True,
    # 'normalize': True,
}


class HateClassifier(object):

    def __init__(self, lang='en', clf='svm', bow=True, bow_params=None, boc=False, boc_params=None,
                 emb=False, emb_params=None, clf_params=None,
                 test_binarize=True):
        """
        lang -- language ('en' or 'es') (default: 'en').
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        bow -- whether to use bag-of-words (default: True).
        bow_params -- bag-of-words vectorizer parameters.
        boc --  whether to use bag-of-characters (default: False).
        boc_params -- bag-of-characters vectorizer parameters.
        emb -- whether to use embeddings (default: False).
        emb_params -- embedding vectorizer parameters.
        clf_params -- classifier parameters.
        test_binarize -- binarize embeddings only on test (not on train).
        """
        self._lang = lang
        self._clf = clf

        vects = []
        transformer_weights = {}
        if bow:
            bow_params = bow_params or default_bow_params
            self._bow_vect = bow_vect = TfidfVectorizer(
                tokenizer=self.build_bow_tokenizer(),
                **bow_params,
            )
            vects.append(('bow_vect', bow_vect))
            transformer_weights['bow_vect'] = 1.0

        if boc:
            boc_params = boc_params or default_boc_params
            self._boc_vect = boc_vect = TfidfVectorizer(
                **boc_params,
            )
            vects.append(('boc_vect', boc_vect))
            transformer_weights['boc_vect'] = 1.0

        if emb:
            self._test_binarize = test_binarize

            emb_params = emb_params or default_emb_params
            self._e_vect = e_vect = WeightedSentenceVectorizer(
                tokenizer=self.build_emb_tokenizer(),
                **emb_params,
            )
            vects.append(('e_vect', e_vect))
            transformer_weights['e_vect'] = 1.0
        else:
            self._e_vect = None

        if len(vects) == 1:
            vect = vects[0][1]
        else:
            vect = FeatureUnion(vects, transformer_weights=transformer_weights)

        clf_params = clf_params or self.clf_params()
        clf = classifiers[clf](**clf_params)
        self._pipeline = Pipeline([
            ('vect', vect),
            ('clf', clf),
        ])

    def build_bow_tokenizer(self):
        neg = self._lang == 'es'  # only handle negations in spanish
        return Tokenizer(lang=self._lang, rdup=True, neg=neg)

    def build_emb_tokenizer(self):
        neg = self._lang == 'es'  # only handle negations in spanish
        return Tokenizer(lang=self._lang, lem=False, neg=neg)

    def clf_params(self):
        return default_clf_params.get(self._clf, {})

    def fit(self, X, y, sample_weight=None):
        if self._e_vect and self._test_binarize:
            # turn off binarization on train
            bin = self._e_vect._binarize
            self._e_vect._binarize = False
        self._pipeline.fit(X, y, clf__sample_weight=sample_weight)
        if self._e_vect and self._test_binarize:
            # restore binarization value
            self._e_vect._binarize = bin

    def predict(self, X):
        return self._pipeline.predict(X)

    def predict_proba(self, X):
        return self._pipeline.predict_proba(X)

    def decision_function(self, X):
        return self._pipeline.decision_function(X)

    def vect(self):
        return self._pipeline.named_steps['vect']

    def clf(self):
        return self._pipeline.named_steps['clf']

    def stats(self):
        vect = self.vect()
        if isinstance(vect, FeatureUnion):
            name, vect = vect.transformer_list[0]
            assert name == 'bow_vect'
        fns = vect.get_feature_names()
        return {'features': len(fns)}

    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        print('accuracy\t{:2.2f}\n'.format(acc))
        print(metrics.classification_report(y_test, y_pred))
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)
