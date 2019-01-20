from .preprocessing import Tokenizer
from .base_model import BaseModel
import numpy as np
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)
import keras
from elmoformanylangs import Embedder


class ElmoModel(BaseModel):
    def __init__(self, max_len, embedder, tokenize_args={},
                 recursive_class=LSTM, lstm_units=128, dropout=[0.75, 0.50],
                 dense_units=128, **kwargs):

        self._max_len = max_len
        self._embedder = embedder
        self._elmo_dim = 1024
        # Build the graph
        input_elmo = Input(shape=(max_len, self._elmo_dim), name="Elmo_Input")
        y = Bidirectional(recursive_class(lstm_units))(input_elmo)
        y = Dropout(dropout[0])(y)
        y = Dense(dense_units, activation='relu', name='dense_elmo')(y)
        y = Dropout(dropout[1])(y)
        
        output = Dense(1, activation='sigmoid', name='output')(y)

        tok_args = {
            "preserve_case": False,
            "deaccent": False,
            "reduce_len": True,
            "strip_handles": False,
            "alpha_only": True,
            "stem": False
        }

        tok_args.update(tokenize_args)

        super().__init__(
            inputs=[input_elmo], outputs=[output],
            tokenize_args=tok_args, **kwargs
        )

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
