import unidecode
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer


class Tokenizer:
    """
    Tokenizer for tweets based on NLTK's Tokenizer + Stemming
    """
    def __init__(self, stem=False, deaccent=False, alpha_only=False, strip_hash=False,
                 language='spanish', **kwargs):
        self._deaccent = deaccent
        self._alpha_only = alpha_only
        if stem:
            self._stemmer = SnowballStemmer(language)
        else:
            self._stemmer = None

        tokenizer_args = {"reduce_len": True}
        tokenizer_args.update(**kwargs)
        
        self._strip_hash = strip_hash

        self._tokenizer = TweetTokenizer(**tokenizer_args)

    def stem(self, token):
        if self._stemmer:
            return self._stemmer.stem(token)
        else:
            return token

    def tokenize(self, text):
        tokens = self._tokenizer.tokenize(text)


        ret = []

        for token in tokens:
            tok = None
            if token[0] == "#" and self._strip_hash:
                tok = self.stem(token[1:])
            elif token[0] == "@":
                tok = "@user"
            elif "http" in token:
                continue
            else:
                tok = self.stem(token)

            if self._deaccent and tok.isalpha():
                tok = unidecode.unidecode(tok)
            if self._alpha_only and not tok.isalpha():
                continue

            ret.append(tok)
        return ret
