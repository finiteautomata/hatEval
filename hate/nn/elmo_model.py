from .preprocessing import Tokenizer
import numpy as np
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)
import keras
from elmoformanylangs import Embedder


class ElmoModel(keras.Model):
    def __init__(self, max_len, embedder, tokenize_args={},
                 recursive_class=LSTM, lstm_units=128, dropout=[0.75, 0.50],
                 dense_units=128):

        self._max_len = max_len
        self._embedder = embedder
        self._tokenizer = Tokenizer(**tokenize_args)
        self._elmo_dim = 1024
        # Build the graph
        input_elmo = Input(shape=(max_len, self._elmo_dim), name="Elmo_Input")
        y = Bidirectional(recursive_class(lstm_units))(input_elmo)
        y = Dropout(dropout[0])(y)
        y = Dense(dense_units, activation='relu', name='dense_elmo')(y)
        output = Dense(1, activation='sigmoid', name='output')(y)


        super().__init__(inputs=[input_elmo], outputs=[output])

    def preprocess_fit(self, X):
        return

    def preprocess_transform(self, X):
        list_of_tokens = [self._tokenizer.tokenize(t) for t in X]

        padded_tokens = []
        for tokens in list_of_tokens:
            if len(tokens) >= self._max_len:
                tokens = tokens[:self._max_len]
            else:
                tokens = tokens + [''] * (self._max_len - len(tokens))

            padded_tokens.append(tokens)

        elmo_embeddings = self._embedder.sents2elmo(padded_tokens)
        return np.array(elmo_embeddings)

    def fit(self, X, y, validation_data=None, **kwargs):
        self.preprocess_fit(X)

        X_train = self.preprocess_transform(X)

        val_data = None
        if validation_data:
            X_val = self.preprocess_transform(validation_data[0])
            y_val = validation_data[1]
            val_data = (X_val, y_val)

        super().fit(X_train, y, validation_data=val_data, **kwargs)

    def evaluate(self, X, y=None, **kwargs):
        X = self.preprocess_transform(X)

        return super().evaluate(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.preprocess_transform(X)

        return super().predict(X, **kwargs)
