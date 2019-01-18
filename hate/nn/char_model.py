from .preprocessing import Tokenizer
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences
import keras



class CharModel(keras.Model):
    def __init__(
        self, vocab_size, max_charlen,
        tokenize_args={}, embedding_dim=64, filters=128, kernel_size=7,
        pooling_size=3, lstm_units=128, dense_units=64, dropout=[0.75, 0.50]):

        self.max_charlen = max_charlen
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(**tokenize_args)
        char_tokenizer = Tokenizer(num_words=vocab_size, char_level=True)

        # Build the graph 
        input_char = Input(shape=(max_charlen,), name="Char_Input")
        x = Embedding(vocab_size, embedding_dim)(input_char)
        x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding='same', activation='relu')(x)

        x = MaxPooling1D(pool_size=pooling_size)(x)
        x = Bidirectional(recursive_class(lstm_units))(x)
        x = Dropout(dropout[0])(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout[1])(x)
        output = Dense(1, activation='sigmoid')(merge_layer)

        super().__init__(inputs=[input_char], outputs=[output])

    def __preprocess(self, X):
        tokens = map(self.tokenizer.tokenize, X)
        instances = [" ".join(seq_tokens) for seq_tokens in tokens]

        return instances


    def fit(self, X, y, validation_data=None, **kwargs):
        X_processed = self.preprocess(X)

        val_data = None
        if validation_data:
            X_val = self.preprocess(validation_data[0])
            y_val = validation_data[1]
            val_data = (X_val, y_val)
