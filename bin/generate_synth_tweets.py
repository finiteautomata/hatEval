"""Generate synthetic tweets by double translation


Usage:
  generate_synthetic_tweets.py [options] --file <file> --out <out_file> --input-lang <input_lang> --mid-lang <mid_lang>

 Options:
    --num-threads <numthreads>          Num of threads to use [default: 50]
    --output-file <output_path>           Output path.
        If not provided, generate a file in TASS/synthetic/<file>
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
from translate import Translator
from nltk.tokenize import TweetTokenizer


emoticon_re = re.compile(r"(?u)"
    r"[\U00002600-\U000027BF]|"     # emoticons
    r"[\U0001f300-\U0001f64F]|"     # https://stackoverflow.com/a/26740753
    r"[\U0001f680-\U0001f6FF]"      # https://apps.timwhitlock.info/emoji/tables/unicode
)


tokenize = TweetTokenizer(reduce_len=True, strip_handles=True).tokenize

def clean_tweet(text):
    """
    Removes emojis and handles
    """
    tokens = tokenize(text)
    tokens = [t for t in tokens if not emoticon_re.match(t)]
    clean_tweet = " ".join(tokens)

    return clean_tweet

class UsageLimit(Exception):
    pass


def double_translate(text, source_lang, intermediate_lang):
    """
    Double translates tweets.

    Raises SystemErrorS
    Arguments:
    ---------

    text: string
        text to be double translated

    source_lang: string
        Source language

    intermediate_lang: string
        Intermediate language

    """
    email = "jmperez.85@gmail.com"
    translator = Translator(
        from_lang=source_lang, to_lang=intermediate_lang,
        email=email
    )
    back_translator = Translator(
        from_lang=intermediate_lang, to_lang=source_lang,
        email=email
    )

    intermediate_tweet = translator.translate(text)
    new_tweet = back_translator.translate(intermediate_tweet)

    if ("MYMEMORY WARNING" in intermediate_tweet) or ("MYMEMORY WARNING" in new_tweet):
        raise UsageLimit(intermediate_tweet)
    return new_tweet, intermediate_tweet


def translate_tweets(tweets, out_df, input_lang, middle_lang):
    translated_count = 0
    new_tweets = []
    for (tweet_id, tweet) in tweets.iterrows():
        try:
            if tweet_id in out_df.index:
                print("{} tweet already translated - skip".format(tweet_id))
                continue

            clean_text = clean_tweet(tweet["text"])
            new_text, intermediate = double_translate(
                clean_text, input_lang, middle_lang)

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


if __name__ == '__main__':
    opts = docopt(__doc__)
    file_path = opts['<file>']
    input_lang = opts['<input_lang>']
    middle_lang = opts['<mid_lang>']
    out_file = opts['<out_file>']

    df = pd.read_table(file_path, index_col="id")

    try:
        out_df = pd.read_table(out_file, index_col="id")
    except FileNotFoundError as e:
        print(e)
        print("Creating new translation file")
        out_df = pd.DataFrame(
            columns=["id", "text", "HS", "TR", "AG"]
        )

        out_df.set_index("id", inplace=True)

    """
    We translate a small number at a time to avoid banning :-(
    """
    sample_number = 500
    df = df.sample(n=sample_number)

    print("Translating file {} ({} entries)".format(file_path, len(df)))
    print("Saving to file {} ({} entries already found)".format(
           out_file, len(out_df)))
    print("Double translation from '{}' to '{}'".format(
          input_lang, middle_lang))

    out_df = translate_tweets(df, out_df, input_lang, middle_lang)

    save_data(out_df, out_file)
