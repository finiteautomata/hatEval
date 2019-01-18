import unittest
from hate.nn.preprocessing import Tokenizer

class TokenizerTest(unittest.TestCase):
    def test_tokenizes_simple(self):
        tokenizer = Tokenizer()

        self.assertEqual(
            tokenizer.tokenize("hola mundo"),
            ["hola", "mundo"]
        )

    def test_does_not_deaccents_by_default(self):
        tokenizer = Tokenizer()

        self.assertEqual(
            tokenizer.tokenize("el mató a un policía motorizado"),
            ["el", "mató", "a", "un", "policía", "motorizado"]
        )

    def test_does_not_lowercase_by_default(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            tokenizer.tokenize("EL PERRO"),
            ["EL", "PERRO"]
        )

    def test_replaces_handles_by_user(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            tokenizer.tokenize("hola @pepe"),
            ["hola", "@user"]
        )


    def test_removes_hash_from_hashtags(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            tokenizer.tokenize("Hola #HashTag"),
            ["Hola", "HashTag"]
        )

    def test_removes_urls(self):
        tokenizer = Tokenizer()

        self.assertEqual(
            tokenizer.tokenize("@usuario http://t.co/123 jajaja"),
            ["@user", "jajaja"]
        )

    def test_keeps_nonalpha_by_default(self):
        tokenizer = Tokenizer()

        self.assertEqual(
            tokenizer.tokenize("1 2 3 $"),
            ["1", "2", "3", "$"]
        )

    def test_removes_nonalpha(self):
        tokenizer = Tokenizer(alpha_only=True)

        self.assertEqual(
            tokenizer.tokenize("hola a1 $"),
            ["hola"]
        )


    def test_stemming(self):
        tokenizer = Tokenizer(stem=True)

        self.assertEqual(
            tokenizer.tokenize("hola gatos"),
            ["hol", "gat"]
        )

    def test_deaccents(self):
        tokenizer = Tokenizer(deaccent=True)

        self.assertEqual(
            tokenizer.tokenize("qué onda möno"),
            ["que", "onda", "mono"]
        )

    def test_reduces_len_by_default(self):
        tokenizer = Tokenizer()

        self.assertEqual(
            tokenizer.tokenize("jajaaaaaa"),
            ["jajaaa"]
        )
