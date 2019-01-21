from collections import defaultdict
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
