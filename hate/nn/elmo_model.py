from .preprocessing import Tokenizer
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)
import keras
from elmoformanylangs import Embedder


class ElmoModel(keras.Model):
    def __init__(self, max_len, path_to_elmo_model, tokenize_args={},
                 recursive_class=LSTM, lstm_units=128, dropout=[0.75, 0.50],
                 dense_units=128):

        self._max_len = max_len
        self._tokenizer = Tokenizer(**tokenize_args)
        self._elmo_dim = 1024
        # Build the graph
        input_elmo = Input(shape=(max_len, self._elmo_dim), name="Elmo_Input")
        y = Bidirectional(recursive_class(lstm_units))(input_elmo)
        y = Dropout(dropout[0])(y)
        y = Dense(dense_units, activation='relu', name='dense_elmo')(y)
        output = Dense(1, activation='sigmoid', name='output')(y)

        self._embedder = Embedder(path_to_elmo_model)

        super().__init__(inputs=[input_elmo], outputs=[output])

    def _preprocess(self, X):
        tokens = map(self._tokenizer.tokenize, X)
        instances = [" ".join(seq_tokens) for seq_tokens in tokens]

        return instances

    def fit(self, X, y, validation_data=None, **kwargs):
        text_train = self._preprocess(X)

        self._char_tokenizer.fit_on_texts(text_train)

        X_train = self._char_tokenizer.texts_to_sequences(text_train)
        X_train = pad_sequences(X_train, self._max_charlen)

        val_data = None
        if validation_data:
            text_val = self._preprocess(validation_data[0])
            X_val = self._char_tokenizer.texts_to_sequences(text_val)
            X_val = pad_sequences(X_val, self._max_charlen)
            y_val = validation_data[1]
            val_data = (X_val, y_val)

        super().fit(X_train, y, validation_data=val_data, **kwargs)

    def evaluate(self, X, y=None, **kwargs):
        X = self._preprocess(X)
        X = self._char_tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, self._max_charlen)

        return super().evaluate(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self._preprocess(X)
        X = self._char_tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, self._max_charlen)

        return super().predict(X, **kwargs)
