from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from fastText import load_model
except:
    pass


class BaseVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, file=None, tokenizer=None, binarize=False, normalize=False):
        """
        file -- filename with stored embeddings
        tokenizer -- sentence tokenizer
        binarize -- don't count token repetitions
        normalize -- return vectors with norm 1
        """
        self._filename = file
        self._tokenizer = tokenizer
        self._binarize = binarize
        self._normalize = normalize
        self._model = load_model(file)

    def fit(self, X, y=None):
        return self

    def get_tweet_representation(self, tweet):
        raise NotImplementedError(
            "Use WordVectorizer or SentenceVectorizer"
        )

    @property
    def dimension(self):
        return self._model.get_dimension()

    @property
    def model(self):
        if self._model is None:
            # TODO: do this in __setstate__
            self._model = load_model(self._filename)
        return self._model

    def transform(self, X):
        return np.array([self.get_tweet_representation(tweet) for tweet in X])

    def __getstate__(self):
        """Return internal state for pickling, omitting unneeded objects.
        """
        # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        state = self.__dict__.copy()
        state['_model'] = None
        return state


class SentenceVectorizer(BaseVectorizer):
    """Converts one tweet to a single vector."""
    def preprocess(self, tweet):
        if self._tokenizer:
            tokens = self._tokenizer(tweet)
            if self._binarize:
                tokens = sorted(set(tokens))
            tweet = ' '.join(tokens)
        else:
            tweet = x.replace('\n', ' ')

        return tweet

    def get_tweet_representation(self, tweet):
        preprocessed_tweet = self.preprocess(tweet)
        vec = self.model.get_sentence_vector(preprocessed_tweet)
        if self._normalize and preprocessed_tweet:
            # (if preprocessed_tweet is empty, vec = 0)
            vec = vec / np.linalg.norm(vec)
        return vec


class WordVectorizer(BaseVectorizer):
    """Converts one tweet to an array of vectors."""
    def get_token_representation(self, token):
        return self.model.get_word_vector(token)

    def get_tweet_representation(self, tweet):
        tokens = self._tokenizer(tweet)

        return np.array([
            self.get_token_representation(token) for token in tokens
        ])


class WeightedSentenceVectorizer(BaseVectorizer):

    def __init__(self, file=None, a=1.0, tokenizer=None, binarize=False, normalize=False):
        """
        Sentence embeddings with simple Smooth Inverse Frequency (SIF) weighting.
        
        file -- filename with stored embeddings
        a -- regularization term for the weights (higher is more regular).
        tokenizer -- sentence tokenizer
        binarize -- don't count token repetitions
        normalize -- return vectors with norm 1
        """
        self._a = a
        super().__init__(file, tokenizer, binarize, normalize)

    def fit(self, X, y=None):
        count = defaultdict(int)
        total = 0
        for x in X:
            tokens = self._tokenizer(x) + ['\n']
            for token in tokens:
                count[token] += 1
            total += 1
        self._count = dict(count)
        #self._total = float(sum(count.values()))
        self._total = float(total)

        return self

    def get_tweet_representation(self, tweet):
        tokens = self._tokenizer(tweet) + ['\n']

        if self._binarize:
            tokens = sorted(set(tokens))

        #weights = [1.0 for _ in tokens]
        a = self._a
        weights = [a / (a + (self._count.get(t, 0) / self._total)) for t in tokens]

        vec_sum = np.zeros(self.model.get_dimension())
        for weight, token in zip(weights, tokens):
            vec = self.model.get_word_vector(token)
            #norm = np.linalg.norm(vec)
            #if norm > 0.0:
            #    vec = vec / norm
            vec_sum += weight * vec
        vec = vec_sum / len(tokens)

        if self._normalize:
            vec = vec / np.linalg.norm(vec)

        return vec
