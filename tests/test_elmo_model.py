import unittest
import numpy as np
from keras.layers import LSTM
from elmoformanylangs import Embedder
from unittest.mock import Mock
from hate.nn import ElmoModel

class ElmoModelTest(unittest.TestCase):

    def setUp(self):

        self.fasttext_model = Mock()
        self.ft_dim = 300
        self.elmo_dim = 1024

        self.fasttext_model.get_word_vector()
        self.elmo_embedder = Mock()

        def mocked_sents2elmo(list_of_tokens):
            return [np.random.randn(len(l), self.elmo_dim) for l in list_of_tokens]

        self.elmo_embedder.sents2elmo = mocked_sents2elmo
        self.fasttext_model.get_word_vector = \
            lambda x: np.random.randn(self.ft_dim)

        self.model_args = {
            "recursive_class": LSTM,
        }

    def test_it_is_created_with_elmo(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=None,
            elmo_embedder=self.elmo_embedder,
            **self.model_args,
        )

        self.assertEqual(model.layers[1].input_shape, (None, 50, self.elmo_dim))



    def test_it_is_created_with_fasttext(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=self.fasttext_model,
            elmo_embedder=None,
            **self.model_args,
        )

        self.assertEqual(model.layers[1].input_shape, (None, 50, self.ft_dim))

    def test_it_is_created_with_both(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=self.fasttext_model,
            elmo_embedder=self.elmo_embedder,
            **self.model_args,
        )
        self.assertEqual(model.layers[3].input_shape,
            (None, 50, self.ft_dim + self.elmo_dim))


    def test_it_creates_with_embedding_size(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=self.fasttext_model,
            elmo_embedder=self.elmo_embedder,
            rnn_units = 100, bidirectional=False,
            **self.model_args,
        )

        self.assertEqual(
            model.layers[4].output_shape,
            (None, 100)
        )

    def test_it_can_be_fitted(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=self.fasttext_model,
            elmo_embedder=self.elmo_embedder,
            rnn_units = 100, bidirectional=False,
            **self.model_args,
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)

    def test_preprocess_transform_returns_vectors_of_given_max_len(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=self.fasttext_model,
            elmo_embedder=self.elmo_embedder,
            rnn_units = 100, bidirectional=False,
            **self.model_args,
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]

        model.preprocess_fit(X)
        ret = model.preprocess_transform(X)

        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0].shape, (2, 50, self.elmo_dim))
        self.assertEqual(ret[1].shape, (2, 50, self.ft_dim))

    def test_with_real_embedder(self):
        model = ElmoModel(
            max_len=50,
            fasttext_model=None,
            elmo_embedder=Embedder("models/elmo/es/"),
            **self.model_args,
        )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)


if __name__ == '__main__':
    unittest.main()
