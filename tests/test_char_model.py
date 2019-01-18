import unittest
import numpy as np
from hate.nn import CharModel


class CharModelTest(unittest.TestCase):
    def test_it_is_created_with_first_layer_of_max_charlen(self):
        model = CharModel(vocab_size=4, max_charlen=10)
        self.assertEqual(model.layers[0].input_shape, (None, 10))

    def test_it_creates_with_embedding_size(self):
        model = CharModel(vocab_size=4, max_charlen=20, embedding_dim=32)

        self.assertEqual(
            model.layers[1].output_shape,
            (None, 20, 32)
        )

    def test_it_can_be_fitted(self):
        model = CharModel(vocab_size=4, max_charlen=10)

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)

    def test_it_can_be_fitted_with_validation_data(self):
        model = CharModel(vocab_size=4, max_charlen=10)

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        X_val = ["Algo", "otra cosa"]
        y_val = np.array([1, 0]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, validation_data=(X_val, y_val), epochs=2)

    def test_it_can_predict(self):
        pass
