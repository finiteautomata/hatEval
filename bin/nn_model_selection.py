"""
Model selection para MLP

Usage:
  nn_model_selection.py [options] --no-iter <no-iter> --output-path <output_file>

 Options:
    --threads <numthreads>          Num of threads to use [default: 2]
    --splits <numsplits>            Num of splits in CV [default: 3]
"""
import os
import glob
import csv
import numpy as np
import pandas as pd
import pickle
from operator import mul
from docopt import docopt
from functools import reduce
from elmoformanylangs import Embedder
from sklearn.model_selection import ParameterSampler
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.optimizers import Adam
from hate.nn import CharModel, ElmoModel, BowModel, MergeModel
# This avoids verbose log warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_model(params, embedder):
    params = params.copy()

    dropout = params.pop('dropout')
    recursive_class = params.pop('recursive_class')
    dense_last_layer = params.pop('dense_last_layer')
    char_model = CharModel(
        vocab_size=params.pop('char__vocab_size'),
        max_charlen=params.pop('char__max_charlen'),
        embedding_dim=params.pop('char__embedding_dim'),
        tokenize_args={
            "stem": params.pop('char__stem'),
            "alpha_only": params.pop('char__alpha_only'),
        },
        filters=params.pop('char__filters'),
        kernel_size=params.pop('char__kernel_size'),
        pooling_size=params.pop('char__pooling_size'),
        dense_units=dense_last_layer,
        recursive_class=recursive_class, dropout=dropout
    )
    elmo_model = ElmoModel(
        max_len=50, embedder=embedder,
        lstm_units=params.pop('elmo__lstm_units'),
        tokenize_args={'deaccent': params.pop('elmo__deaccent')},
        dense_units=dense_last_layer,
        recursive_class=recursive_class, dropout=dropout
    )
    bow_model = BowModel(
        num_words=params.pop('bow__num_words'),
        dense_units=[1024, dense_last_layer], dropout=dropout,
    )
    merge_model = MergeModel([char_model, elmo_model, bow_model])
    optimizer_args = {
        "lr": params.pop('lr'),
        "decay": params.pop('decay')
    }
    merge_model.compile(loss='binary_crossentropy',
              optimizer=Adam(**optimizer_args),
              metrics=['accuracy'])

    print(params.keys())
    assert(len(params) == 0)


if __name__ == "__main__":
    opts = docopt(__doc__)

    """
    Load datasets
    """

    df_dev = pd.read_table("data/es/dev_es.tsv", index_col="id", quoting=csv.QUOTE_NONE)
    df_train = pd.read_table("data/es/train_es.tsv", index_col="id", quoting=csv.QUOTE_NONE)

    X_train, y_train = df_train["text"], df_train["HS"]
    X_dev, y_dev = df_dev["text"], df_dev["HS"]

    param_space = {
        "dropout": [[x, y] for x in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
                    for y in [0.75, 0.65, 0.55, 0.45]],
        "recursive_class": [CuDNNLSTM, CuDNNGRU],
        "elmo__lstm_units": [32, 64, 128, 256],
        "elmo__deaccent": [False, True],
        "lr": [0.001, 0.0005, 0.00035, 0.00025, 0.00015],
        "decay": [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02],
        "char__vocab_size": [150, 175, 200, 225, 250],
        "char__max_charlen": [200],
        "char__embedding_dim": [64, 96, 128],
        "char__filters": [64, 96, 128, 160, 192, 256],
        "char__kernel_size": [4, 5, 6, 7],
        "char__pooling_size": [2, 3, 4],
        "char__alpha_only": [True, False],
        "char__stem": [True, False],
        "batch_size": [32, 64, 96],
        "bow__num_words": [3000, 4000, 4500],
        "dense_last_layer": [32, 64, 128, 256],
    }


    pos = reduce(mul, [len(values) for _, values in param_space.items()], 1)

    results = []

    output_path = opts['<output_file>']
    no_iter = int(opts["<no-iter>"])

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    print("Cantidad de posibilidades: {}".format(pos))
    print("Cantidad de iteraciones = {}".format(no_iter))
    param_list = list(ParameterSampler(param_space, no_iter))

    embedder = 1# Embedder("models/elmo/es/")
    try:
        with open(output_path, "rb") as f:
            iters = pickle.load(f)
            print("Cargamos {} iteraciones previas")
    except FileNotFoundError:
        print("Archivo nuevo")
        iters = []

    for i, params in enumerate(param_list):
        orig_params = params.copy()
        batch_size = params.pop('batch_size')
        model = create_model(params, embedder=embedder)
        early_stopper = EarlyStopping(monitor='val_loss', patience=15)
        history = merge_model.fit(X_train, y_train,  callbacks=[early_stopper],
                  validation_data=(X_dev, y_dev), epochs=300, batch_size=batch_size)
        iter_info = {
            "number": i,
            "params": params,
            "history": history.history,
            "val_acc": max(history.history['val_acc']),
            "val_loss": min(history.history['val_loss']),
            "no_epochs": len(history.history['val_acc']),
        }

        iters.append(iter_info)

    with open(output_path, "wb+") as f:
        pickle.dump(iters, f)

    print("Salvamos {} iteraciones ({} nuevas) en {}".format(
          len(iters), no_iter, output_path))
