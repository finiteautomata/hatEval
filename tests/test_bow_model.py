import unittest
import numpy as np
from hate.nn import BowModel


class BowModelTest(unittest.TestCase):
    def test_it_is_created_with_correct_input_size(self):
        model = BowModel(num_words=10)
        self.assertEqual(model.layers[0].input_shape, (None, 10))


    def test_it_creates_with_embedding_size(self):
        model = BowModel(num_words=10, dense_units=[16, 32])

        self.assertEqual(
            model.layers[1].output_shape,
            (None, 16)
        )

        self.assertEqual(
            model.layers[3].output_shape,
            (None, 32)
        )

    def test_it_can_be_fitted(self):
        model = BowModel(num_words=5,
            vectorize_args={"max_df": 1.0, "min_df": 0.0},
            )

        X = ["Esto no es agresivo", "Esto sí es agresivo"]
        y = np.array([0, 1]).reshape(-1, 1)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=2)
