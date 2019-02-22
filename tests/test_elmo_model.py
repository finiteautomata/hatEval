import unittest
import numpy as np
from elmoformanylangs import Embedder
from unittest.mock import Mock
from hate.nn import ElmoModel

class ElmoModelTest(unittest.TestCase):
    def test_it_is_created_with_first_layer_of_max_charlen(self):
        model = ElmoModel(
            max_len=50,
            embedder=Mock())

        self.assertEqual(model.layers[0].input_shape, (None, 50, 1024))

    def test_it_creates_with_embedding_size(self):
        model = ElmoModel(
            max_len=50, embedder=Mock(),
            lstm_units=50
        )

        # As it is bidirectional, it is the double
        self.assertEqual(
            model.layers[2].output_shape,
            (None, 100)
        )

    def test_it_can_be_fitted(self):
        embedder = Mock()

        def mocked_sents2elmo(list_of_tokens):
            return [np.random.randn(len(l), 1024) for l in list_of_tokens]

        embedder.sents2elmo = mocked_sents2elmo

        model = ElmoModel(
            max_len=50,
            embedder=embedder,
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)

    def test_preprocess_transform_returns_vectors_of_given_max_len(self):
        embedder = Mock()

        def mocked_sents2elmo(list_of_tokens):
            return [np.random.randn(len(l), 1024) for l in list_of_tokens]

        embedder.sents2elmo = mocked_sents2elmo

        model = ElmoModel(
            max_len=50,
            embedder=embedder,
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]

        model.preprocess_fit(X)

        self.assertEqual(
            model.preprocess_transform(X).shape,
            (2, 50, 1024)
        )

    def test_with_real_embedder(self):
        model = ElmoModel(
            max_len=50,
            embedder=Embedder("models/elmo/es/"),
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)


if __name__ == '__main__':
    unittest.main()
