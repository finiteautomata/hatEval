import unittest
import numpy as np
from hate.nn import ElmoModel


class ElmoModelTest(unittest.TestCase):
    def test_it_is_created_with_first_layer_of_max_charlen(self):
        model = ElmoModel(max_len=50, path_to_elmo_model="models/elmo/es")


if __name__ == '__main__':
    unittest.main()
