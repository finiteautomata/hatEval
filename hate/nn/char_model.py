from .base_model import BaseModel
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)
import keras


class CharModel(BaseModel):
    def __init__(self, vocab_size=200, max_charlen=250,
                 tokenize_args={}, embedding_dim=64, filters=128,
                 kernel_size=7, pooling_size=3,
                 recursive_class=LSTM, recursive_units=128,
                 dense_units=64, dropout=[0.75, 0.50], **kwargs):

        self._max_charlen = max_charlen
        self._vocab_size = vocab_size
        self._char_tokenizer = KerasTokenizer(
            num_words=vocab_size, char_level=True
        )

        # Build the graph
        input_char = Input(shape=(max_charlen,), name="Char_Input")
        x = Embedding(vocab_size, embedding_dim)(input_char)
        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   padding='same', activation='relu')(x)

        x = MaxPooling1D(pool_size=pooling_size)(x)
        x = Bidirectional(recursive_class(recursive_units))(x)
        if dropout[0] > 0:
            x = Dropout(dropout[0])(x)
        x = Dense(dense_units, activation='relu')(x)
        if dropout[1] > 0:
            x = Dropout(dropout[1])(x)
        output = Dense(1, activation='sigmoid')(x)

        tok_args = {
            "preserve_case": False,
            "deaccent": True,
            "reduce_len": True,
            "strip_handles": False,
            "stem": True,
            "alpha_only": False
        }


        tok_args.update(tokenize_args)

        super().__init__(
            inputs=[input_char], outputs=[output],
            tokenize_args=tok_args, **kwargs)

    def _preprocess_text(self, X):
        tokens = map(self._tokenizer.tokenize, X)
        instances = [" ".join(seq_tokens) for seq_tokens in tokens]

        return instances

    def preprocess_fit(self, X):
        text_train = self._preprocess_text(X)

        self._char_tokenizer.fit_on_texts(text_train)

    def preprocess_transform(self, X):
        X_transf = self._preprocess_text(X)
        X_transf = self._char_tokenizer.texts_to_sequences(X_transf)

        return pad_sequences(X_transf, self._max_charlen)
