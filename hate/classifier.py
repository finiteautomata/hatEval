from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD

from hate.tokenizer import Tokenizer
from hate.embeddings import SentenceVectorizer, WeightedSentenceVectorizer
from hate.elmo import CachedElmoVectorizer


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
    'xgb': XGBClassifier,
}

embeddings = {
    'fasttext': SentenceVectorizer,
    'wfasttext': WeightedSentenceVectorizer,
    'elmo': CachedElmoVectorizer,
}

default_clf_params = {
    'dt': {'random_state': 0},
    'maxent': {
        'penalty': 'l2',  # 'l1' or 'l2'
        'C': 1.0,  # decrease for more regularization
        'class_weight': 'balanced',
        'random_state': 0,
    },
    'lrcv': {
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

default_svd_params = {
    'algorithm': 'randomized',
    'n_components': 300,
    'random_state': 0,
}

default_emb_params = {
    'a': 0.1,
    # 'binarize': True,
    # 'normalize': True,
}


class HateClassifier(object):

    def __init__(self, lang='en',
                 bow=True, bow_params=None,
                 boc=False, boc_params=None,
                 svd=False, svd_params=None,
                 emb=False, emb_params=None,
                 clf='svm', clf_params=None):
        """
        lang -- language ('en' or 'es') (default: 'en').
        bow -- whether to use bag-of-words (default: True).
        bow_params -- bag-of-words vectorizer parameters.
        boc --  whether to use bag-of-characters (default: False).
        boc_params -- bag-of-characters vectorizer parameters.
        svd -- whether to use svd to reduce bag-of-* dimensionality (default: False).
        svd_params -- svd parameters.
        emb -- embeddings model, False if none (default: False).
        emb_params -- embedding vectorizer parameters.
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        clf_params -- classifier parameters.
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

        if svd:
            svd_params = svd_params or default_svd_params
            if len(vects) == 1:
                vect = vects[0][1]
            else:
                vect = FeatureUnion(vects)
            vects = [('bag_vect', Pipeline([
                ('vect', vect),
                ('svd', TruncatedSVD(**svd_params)),
            ]))]

        if emb is None:
            embs, emb_paramss = [], []
        elif not isinstance(emb, list):
            embs, emb_paramss = [emb], [emb_params]
        else:
            embs, emb_paramss = emb, emb_params
        for emb, emb_params in zip(embs, emb_paramss):
            emb_params = emb_params or default_emb_params
            if 'tokenizer' not in emb_params:
                emb_params['tokenizer'] = self.build_emb_tokenizer()
            e_vect = embeddings[emb](**emb_params)
            vects.append(('{}_vect'.format(emb), e_vect))
            transformer_weights['{}_vect'.format(emb)] = 1.0

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
        return Tokenizer(lang=self._lang, lem=False, neg=False, rdup=True)

    def clf_params(self):
        return default_clf_params.get(self._clf, {})

    def fit(self, X, y, sample_weight=None):
        self._pipeline.fit(X, y, clf__sample_weight=sample_weight)

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
