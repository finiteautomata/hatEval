from .preprocessing import Tokenizer
from .base_model import BaseModel
import numpy as np
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, CuDNNGRU, Concatenate
)
import keras
from elmoformanylangs import Embedder


class ElmoModel(BaseModel):
    def __init__(self, max_len, fasttext_model, elmo_embedder,
                 tokenize_args={}, 
                 recursive_class=CuDNNGRU, rnn_units=256, dropout=0.75,
                 pooling='max', **kwargs):

        self._max_len = max_len
        self._embedder = fasttext_model
        self._elmo_embedder = elmo_embedder
        self._elmo_dim = 1024
        # Build the graph
        
        embedding_size = fasttext_model.get_word_vector("pepe").shape[0]
        
        elmo_input = Input(shape=(max_len, self._elmo_dim))
        emb_input = Input(shape=(max_len, embedding_size))

        x = Concatenate()([elmo_input, emb_input])
        self.recursive_layer = Bidirectional(CuDNNGRU(rnn_units, return_sequences=True))(x)
        x = Dropout(dropout)(self.recursive_layer)
        if pooling == 'max':
            x = GlobalMaxPooling1D()(x)
        elif pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        else:
            raise ValueError("pooling should be 'max' or 'avg'")
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

        super().__init__(
            inputs=[elmo_input, emb_input], outputs=[output],
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

        elmo_embeddings = self._elmo_embedder.sents2elmo(X_tokenized)
        fasttext_embeddings = np.array([
            self._get_embeddings(tweet) for tweet in X_tokenized
        ])
        return [np.array(elmo_embeddings), fasttext_embeddings]
