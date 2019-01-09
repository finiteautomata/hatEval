"""Generate synthetic tweets by double translation


Usage:
  generate_synthetic_tweets.py [options] --file <file> --input-lang <input_lang> --mid-lang <mid_lang>

 Options:
    --provider <provider>          Translator to use [default: mymemory]
    --out-file <out_path>          Where to save the translations
    --email <email>                Email of account (only MyMemory)
"""

import json
import os
import time
from docopt import docopt
import multiprocessing
import functools
import re
import pandas as pd
import csv
from json import JSONDecodeError
import translate
import googletrans
from nltk.tokenize import TweetTokenizer


emoticon_re = re.compile(r"(?u)"
    r"[\U00002600-\U000027BF]|"     # emoticons
    r"[\U0001f300-\U0001f64F]|"     # https://stackoverflow.com/a/26740753
    r"[\U0001f680-\U0001f6FF]"      # https://apps.timwhitlock.info/emoji/tables/unicode
)


class UsageLimit(Exception):
    pass


class GoogleTranslator:
    def __init__(self, source_lang, to_lang, **kwargs):
        self.source_lang = source_lang
        self.to_lang = to_lang
        self.translator = googletrans.Translator()

    def translate(self, text):
        try:
            return self.translator.translate(
                text, dest=self.to_lang, src=self.source_lang).text
        except JSONDecodeError as e:
            raise UsageLimit(e)


class MyMemoryTranslator:
    def __init__(self, source_lang, to_lang, **kwargs):
        self.source_lang = source_lang
        self.to_lang = to_lang
        self.translator = translate.Translator(
            from_lang=source_lang, to_lang=to_lang, **kwargs
        )

    def translate(self, text):
        translation = self.translator.translate(text)

        if "MYMEMORY WARNING" in translation:
            raise UsageLimit(translation)
        return translation


tokenize = TweetTokenizer(reduce_len=True, strip_handles=True).tokenize


def clean_tweet(text):
    """
    Removes emojis and handles
    """
    tokens = tokenize(text)
    tokens = [t for t in tokens if not emoticon_re.match(t)]
    clean_tweet = " ".join(tokens)

    return clean_tweet


def translate_tweets(tweets, out_df, translator, back_translator):
    translated_count = 0
    new_tweets = []
    for (tweet_id, tweet) in tweets.iterrows():
        try:
            if tweet_id in out_df.index:
                print("{} tweet already translated - skip".format(tweet_id))
                continue
            clean_text = clean_tweet(tweet["text"])
            intermediate = translator.translate(clean_text)
            new_text = back_translator.translate(intermediate)

            new_row = tweet.copy()
            new_row["text"] = new_text

            new_tweets.append(new_row)
            translated_count += 1

            print("Translated tweet number {}".format(translated_count))
            print("="*40)
            print("Original: {}\n".format(tweet["text"]))
            print("Intermediate: {}\n".format(intermediate))
            print("Double-Translated: {}\n".format(new_text))
            print("\n"*2)
        except UsageLimit as e:
            """ This error usually marks rate limit - stop """
            print(("=" * 80))
            print(e)
            break

    print("{} new tweets".format(len(new_tweets)))
    if len(new_tweets) > 0:
        out_df = out_df.append(new_tweets)

    return out_df


def save_data(out_df, out_file):
    out_df.to_csv(out_file, index_label="id",
                  sep='\t', quoting=csv.QUOTE_NONE)

    print("{} synthetic tweets saved to {}".format(len(out_df), out_file))


def get_outfile_path(opts, file_path, middle_lang):
    if opts['--out-file']:
        return opts['--out-file']
    else:
        base, ext = os.path.splitext(file_path)
        return base + ".synth.{}{}".format(middle_lang, ext)


def create_out_df(out_file):
    try:
        out_df = pd.read_table(out_file, index_col="id")
    except FileNotFoundError as e:
        print(e)
        print("Creating new translation file")
        out_df = pd.DataFrame(
            columns=["id", "text", "HS", "TR", "AG"]
        )

        out_df.set_index("id", inplace=True)

    return out_df

def create_translators(input_lang, middle_lang, provider, **kwargs):
    if provider == 'mymemory':
        klass = MyMemoryTranslator
    elif provider == 'google':
        klass = GoogleTranslator

    forward = klass(
        source_lang=input_lang, to_lang=middle_lang, **kwargs
    )

    back = klass(
        source_lang=middle_lang, to_lang=input_lang, **kwargs
    )

    return forward, back


if __name__ == '__main__':
    opts = docopt(__doc__)
    print(opts)
    file_path = opts['<file>']
    input_lang = opts['<input_lang>']
    middle_lang = opts['<mid_lang>']
    provider = opts['--provider']
    out_file = get_outfile_path(opts, file_path, middle_lang)

    df = pd.read_table(file_path, index_col="id")
    out_df = create_out_df(out_file)
    """
    We translate a small number at a time to avoid banning :-(
    """
    sample_number = 200

    non_translated = df[~df.index.isin(out_df.index)].sample(n=sample_number)

    print("Translating file {} ({} entries)".format(file_path, len(df)))
    print("Saving to file {} ({} entries already found)".format(
           out_file, len(out_df)))
    print("Double translation from '{}' to '{}'".format(
          input_lang, middle_lang))
    print("Using provider: {}".format(provider))

    translator, back_translator = create_translators(
        input_lang, middle_lang, provider, email=opts["--email"]
    )
    out_df = translate_tweets(
        non_translated, out_df, translator, back_translator)

    save_data(out_df, out_file)
