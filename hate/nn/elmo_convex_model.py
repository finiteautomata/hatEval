from .preprocessing import Tokenizer
from .base_model import BaseModel
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, CuDNNGRU, Concatenate
)
import keras
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import numpy as np
from elmoformanylangs import Embedder

class ConvexCombination(Layer):
    def __init__(self, **kwargs):
        super(ConvexCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, _, _ = input_shape[0]
        self.len_inputs = len(input_shape)
        self.l = self.add_weight(name='l0',
                                     shape=(self.len_inputs, 1),  # Adding one dimension for broadcasting
                                     initializer='uniform',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        self.gamma =  self.add_weight(name='elmo_gamma',
                                     shape=(1, 1),  # Adding one dimension for broadcasting
                                     initializer='uniform',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        super(ConvexCombination, self).build(input_shape)

    def call(self, x):
        # x is a list of two tensors with shape=(batch_size, H, T)
        weights = K.softmax(self.l)
        ret = 0
        for i in range(self.len_inputs):
            ret += weights[i] * x[i]
        return self.gamma * ret

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class ElmoConvexModel(BaseModel):
    def __init__(self, max_len, fasttext_model, tokenize_args={},
                 elmo_embedder=None, elmo_range=[0, 1, 2],
                 recursive_class=CuDNNGRU, rnn_units=256, dropout=0.75,
                 l1_regularization=0.00, pooling='max', bidirectional=False,
                 num_classes=1, **kwargs):

        self._max_len = max_len
        self._embedder = fasttext_model
        self._elmo_embedder = elmo_embedder
        self._elmo_dim = 1024
        # Build the graph


        elmo_input = None
        emb_input = None
        inputs = []
        concatenate = []
        self.elmo_range = elmo_range

        if elmo_embedder:
            elmo_inputs = []
            for i in elmo_range:
                inp = Input(shape=(max_len, self._elmo_dim), name="Elmo{}".format(i))
                elmo_inputs.append(inp)

            inputs.extend(elmo_inputs)
            concatenate.append(ConvexCombination()(elmo_inputs))
        if fasttext_model:
            embedding_size = fasttext_model.get_word_vector("pepe").shape[0]
            emb_input = Input(shape=(max_len, embedding_size), name="Fasttext")
            inputs.append(emb_input)
            concatenate.append(emb_input)

        if len(concatenate) > 1:
            x = Concatenate()(concatenate)
        else:
            x = concatenate[0]

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

        if dropout:
            x = Dropout(dropout)(x)

        if num_classes == 1:
            output = Dense(1, activation='sigmoid')(x)
        else:
            output = Dense(1, activation='softmax')(x)

        tok_args = {
            "preserve_case": False,
            "deaccent": False,
            "reduce_len": True,
            "strip_handles": True,
            "alpha_only": True,
            "stem": False
        }

        tok_args.update(tokenize_args)

        self.display_name = "{}{} with {} pooling consuming ELMo + FastText".format(
            'bi-' if bidirectional else '',
            type(recursive_class(50)).__name__,
            pooling,
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

        elmo_embeddings = None
        fasttext_embeddings = None

        ret = []
        if self._elmo_embedder:
            for i in self.elmo_range:
                elmo_inp = np.array(self._elmo_embedder.sents2elmo(X_tokenized, i))
                ret.append(elmo_inp)
        if self._embedder:
            ret.append(np.array([
                self._get_embeddings(tweet) for tweet in X_tokenized
            ]))

        return ret
