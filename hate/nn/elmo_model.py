from .preprocessing import Tokenizer
from .base_model import BaseModel
import numpy as np
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, CuDNNGRU, Concatenate
)
import keras
from keras import regularizers
from elmoformanylangs import Embedder


class ElmoModel(BaseModel):
    def __init__(self, max_len, fasttext_model, elmo_embedder,
                 tokenize_args={},
                 recursive_class=CuDNNGRU, rnn_units=256, dropout=0.75, l1_regularization=0.00,
                 pooling='max', bidirectional=False, **kwargs):

        self._max_len = max_len
        self._embedder = fasttext_model
        self._elmo_embedder = elmo_embedder
        self._elmo_dim = 1024
        # Build the graph

        inputs = []

        if elmo_embedder:
            elmo_input = Input(shape=(max_len, self._elmo_dim), name="Elmo")
            inputs.append(elmo_input)
        if fasttext_model:
            embedding_size = fasttext_model.get_word_vector("pepe").shape[0]
            emb_input = Input(shape=(max_len, embedding_size), name="Fasttext")
            inputs.append(emb_input)

        if len(inputs) > 1:
            x = Concatenate()(inputs)
        else:
            x = inputs[0]

        recursive_args = {
            "return_sequences": True
        }

        if l1_regularization > .0:
            print("Using L1 regularization")
            recursive_args["kernel_regularizer"]= regularizers.l2(l1_regularization)

        rec_layer = recursive_class(
            rnn_units, **recursive_args)

        if bidirectional:
            rec_layer = Bidirectional(rec_layer)
        x = self.recursive_layer = rec_layer(x)


        if pooling == 'max':
            x = GlobalMaxPooling1D()(x)
        elif pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        else:
            raise ValueError("pooling should be 'max' or 'avg'")

        if dropout > 0:
            x = Dropout(dropout)(x)

        output = Dense(1, activation='sigmoid')(x)

        tok_args = {
            "preserve_case": False,
            "deaccent": False,
            "reduce_len": True,
            "strip_handles": True,
            "alpha_only": True,
            "stem": False
        }

        tok_args.update(tokenize_args)

        self.display_name = "{}{} with {} pooling consuming {}".format(
            'bi-' if bidirectional else '',
            type(recursive_class(50)).__name__,
            pooling,
            "+".join([i.name for i in inputs]),
        )
        super().__init__(
            inputs=inputs, outputs=[output],
            tokenize_args=tok_args, **kwargs
        )

    def preprocess_fit(self, X):
        return

    def _preprocess_tweet(self, tweet):
        tokens = self._tokenizer.tokenize(tweet)

        if len(tokens) >= self._max_len:
            tokens = tokens[:self._max_len]
        else:
            tokens = tokens + [''] * (self._max_len - len(tokens))
        return tokens

    def _get_embeddings(self, toks):
        return [self._embedder.get_word_vector(tok) for tok in toks]


    def preprocess_transform(self, X):
        X_tokenized = [self._preprocess_tweet(tweet) for tweet in X]

        ret = []

        if self._elmo_embedder:
            ret.append(np.array(self._elmo_embedder.sents2elmo(X_tokenized)))
        if self._embedder:
            fasttext_embeddings = np.array([
                self._get_embeddings(tweet) for tweet in X_tokenized
            ])
            ret.append(fasttext_embeddings)
        return ret
