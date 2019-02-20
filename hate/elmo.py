from collections import defaultdict
import pickle
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import TweetTokenizer


class CachedElmoVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, train_file=None, test_file=None, tokenizer=None):
        """
        file -- filenames with stored embeddings
        """
        with open(train_file, 'rb') as f:
            self.emb_train = pickle.load(f)
        with open(test_file, 'rb') as f:
            self.emb_test = pickle.load(f)
        self._binarize = False
        self._mode = 'train'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self._mode == 'train':
            emb = self.emb_train
            self._mode = 'test'
        else:
            emb = self.emb_test
        assert len(X) == len(emb)
        return np.array([np.average(e, axis=0) for e in emb])
        # VERY BAD RESULTS:
        #return np.array([e[0] for e in emb])
        #return np.array([e[-1] for e in emb])


url = re.compile(r'https?://[\w./\-?=&+]+')
mention = re.compile(r'((?<=\W)|^)@\w+')
hashtag = re.compile(r'((?<=\W)|^)#\w+')
email = re.compile(r'[\w.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+')
# (based on http://emailregex.com/)


class ElmoTokenizer(object):

    def __init__(self, **kwargs):
        self._rdup = True
        self._tokenizer = TweetTokenizer(**kwargs)

    def __call__(self, doc):
        doc = url.sub('URL', doc)
        doc = mention.sub('@USER', doc)
        #doc = hashtag.sub('#HTAG', doc)
        doc = email.sub('user@mail.com', doc)
        doc = doc.lower()
        tokens = self._tokenizer.tokenize(doc)

        if self._rdup:
            # remove consecutive duplicate user mentions
            new_tokens = []
            prev = None
            for t in tokens:
                if prev != t or t not in {'@USER', 'URL'}:
                    new_tokens.append(t)
                prev = t
            tokens = new_tokens
            # remove ANY consecutive duplicates:
            # tokens = [t for t, _ in groupby(tokens)]

        return tokens
