from .base_model import BaseModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)


class BowModel(BaseModel):
    def __init__(self, num_words, tokenize_args={}, vectorize_args={},
                 dense_units=[512, 128], dropout=[0.75, 0.50],
                 **kwargs):

        # Build the graph
        input_bow = Input(shape=(num_words, ), name="BoW_Input")
        z = Dense(dense_units[0], activation='relu')(input_bow)
        z = Dropout(dropout[0])(z)
        z = Dense(dense_units[1], activation='relu')(z)
        z = Dropout(dropout[1])(z)
        output = Dense(1, activation='sigmoid')(z)

        tok_args = {
            "preserve_case": False,
            "deaccent": False,
            "reduce_len": True,
            "strip_handles": False,
            "alpha_only": False,
            "stem": True
        }

        tok_args.update(tokenize_args)

        super().__init__(
            inputs=[input_bow], outputs=[output],
            tokenize_args=tok_args, **kwargs
        )
        vec_args ={
            "max_features": num_words,
            "max_df": 0.65,
            "min_df": 0.001,
            "ngram_range": (1, 2),
            "binary": True,
        }

        vec_args.update(vectorize_args)

        self._count_vectorizer = CountVectorizer(
            tokenizer=self._tokenizer.tokenize,
            **vec_args
        )


    def preprocess_fit(self, X):
        self._count_vectorizer.fit(X)

    def preprocess_transform(self, X):
        return self._count_vectorizer.transform(X)
