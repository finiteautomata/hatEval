import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import treetaggerwrapper
import spacy

from embeddings.tokenizer import TweetTokenizer
from sentiment.negation import handle_negations, negation_tokens
from .settings import tree_tagger_path, tree_tagger_params_path
from sentiment.emoji import EmojiDict


mentions = re.compile(r'(?:@[^\s]+)')
hashtags = re.compile(r'(?:#[^\s]+)')
urls = re.compile(r'(?:https?\://t.co/[\w]+)')
digits = re.compile(r'(?:\d+)')
punct = re.compile(r'(?:[!(),.…\-:;?¡¿"\'`´^*+=_%])')


resplit = re.compile(r'[- ]')


# http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
class Tokenizer(object):
    """
    """

    def __init__(self, lang='es', stem=False, lem=True, ht=True, emoji=False, rdup=False, neg=True):
        """
        es -- language ('es' or 'en')
        stem -- do stemming
        lem -- do lemmatization (overrides stem)
        ht -- filter hashtags
        emoji -- replace emojis with polarity markers
        rdup -- remove consecutive duplicate user mentions and URLs
        neg -- handle negations
        """
        self._lang = lang
        if stem and not lem:
            self._stemmer = SnowballStemmer('spanish' if lang == 'es' else 'english')
        else:
            self._stemmer = None
        if lem:
            # self._nlp = spacy.load('es', disable=['tagger', 'parser', 'ner'])
            tagger_config = {
                'TAGLANG': lang,
                'TAGDIR': tree_tagger_path,
                'TAGPARFILE': tree_tagger_params_path[lang],
            }
            self._nlp = treetaggerwrapper.TreeTagger(**tagger_config)
        else:
            self._nlp = None
        if emoji:
            filename = 'emoji/Emoji Sentiment Ranking 1.0/Emoji_Sentiment_Data_v1.0.csv'
            self._emoji = EmojiDict(filename)
        else:
            self._emoji = None
        self._tokenizer = TweetTokenizer()
        self._tokenizer._tokenizer.reduce_len = True  # FIXME
        self._stopwords = set(stopwords.words('spanish' if lang == 'es' else 'english'))
        if neg:
            assert lang == 'es', 'English negation handling not supported. Use neg=False.'
            self._stopwords = self._stopwords - negation_tokens
        if lem and lang == 'es':
            # avoid this with embeddings (only for reproducibility):
            self._stopwords.update({'haber', 'ser', 'hacer'})

        self._ht = ht
        self._rdup = rdup
        self._neg = neg

    def __call__(self, doc):
        tokens = self._tokenizer(doc)

        if not self._ht:
            tokens = [t for t in tokens if not hashtags.match(t)]
        tokens = [t for t in tokens if not digits.match(t)]
        tokens = [t.lower() for t in tokens]

        if self._nlp:
            # spacy:
            # doc = ' '.join(tokens)
            # tokens = [t.lemma_ for t in self._nlp(doc)]

            # treetagger:
            tags = self._nlp.tag_text(tokens,
                                      tagonly=True, notagurl=True, notagemail=True,
                                      notagip=True, notagdns=True, nosgmlsplit=True)
            lemmas = [t.split('\t')[-1] for t in tags]
            # alternatively:
            # tags2 = treetaggerwrapper.make_tags(tags)
            # lemmas = [t.lemma for t in tags2]

            fixed_lemmas = []
            for token, lemma in zip(tokens, lemmas):
                if lemma == '<unknown>':
                    # not sure if possible:
                    lemma = token
                elif lemma == '@card@':
                    # cardinals are processed elsewhere
                    lemma = token
                elif '_' in lemma:
                    # for strange lemmas such as 'caerse_la_cara_de_vergüenza'
                    lemma = token

                # split compound lemmas such as 'de-el' or 'comprar-te'
                lemma = lemma.split('-')
                fixed_lemmas.extend(lemma)

            # s = ' '.join(fixed_lemmas)
            # if '_' in s:
            #     print('Warning shitty token: {}'.format(s))
            tokens = fixed_lemmas

        tokens = [t for t in tokens if t not in self._stopwords]
        if self._neg:
            tokens = handle_negations(tokens)
        tokens = [t for t in tokens if not punct.match(t)]
        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]  # also lowercases

        if self._emoji:
            tokens = [self._emoji.get_polarity(t, t) for t in tokens]

        if self._rdup:
            # remove consecutive duplicate user mentions
            new_tokens = []
            prev = None
            for t in tokens:
                if prev != t or t not in {'@user', 'url'}:
                    new_tokens.append(t)
                prev = t
            tokens = new_tokens
            # remove ANY consecutive duplicates:
            # tokens = [t for t, _ in groupby(tokens)]

        return tokens

    def __getstate__(self):
        """Return internal state for pickling, omitting unneeded objects.
        """
        # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        state = self.__dict__.copy()
        if state['_nlp'] is not None:
            state['_nlp'] = 'treetagger'
        return state

    def __setstate__(self, state):
        """
        """
        self.__dict__.update(state)
        if self._nlp == 'treetagger':
            lang = self._lang
            tagger_config = {
                'TAGLANG': lang,
                'TAGDIR': tree_tagger_path,
                'TAGPARFILE': tree_tagger_params_path[lang],
            }
            self._nlp = treetaggerwrapper.TreeTagger(**tagger_config)
