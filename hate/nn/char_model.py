from .preprocessing import Tokenizer
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input, Embedding, Dense, Dropout, Bidirectional, LSTM,
    MaxPooling1D, Conv1D,
)
import keras


class CharModel(keras.Model):
    def __init__(self, vocab_size, max_charlen,
                 tokenize_args={}, embedding_dim=64, filters=128,
                 kernel_size=7, pooling_size=3,
                 recursive_class=LSTM, recursive_units=128,
                 dense_units=64, dropout=[0.75, 0.50]):

        self._max_charlen = max_charlen
        self._vocab_size = vocab_size
        self._tokenizer = Tokenizer(**tokenize_args)
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
        x = Dropout(dropout[0])(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout[1])(x)
        output = Dense(1, activation='sigmoid')(x)

        super().__init__(inputs=[input_char], outputs=[output])

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
