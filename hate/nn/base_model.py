from .preprocessing import Tokenizer
import keras


class BaseModel(keras.Model):
    def __init__(self, tokenize_args={}, **kwargs):

        self._tokenizer = Tokenizer(**tokenize_args)
        super().__init__(**kwargs)

    def preprocess_fit(self, X):
        raise NotImplementedError()

    def preprocess_transform(self, X):
        raise NotImplementedError()

    def fit(self, X, y, validation_data=None, **kwargs):
        self.preprocess_fit(X)

        X_train = self.preprocess_transform(X)

        val_data = None
        if validation_data:
            X_val = self.preprocess_transform(validation_data[0])
            y_val = validation_data[1]
            val_data = (X_val, y_val)

        return super().fit(X_train, y, validation_data=val_data, **kwargs)

    def evaluate(self, X, y=None, **kwargs):
        X = self.preprocess_transform(X)

        return super().evaluate(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.preprocess_transform(X)

        return super().predict(X, **kwargs)
