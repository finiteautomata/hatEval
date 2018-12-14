"""Train a HatEval task A model.

Usage:
  train.py [options] -o <file>
  train.py -h | --help

Options:
  -l <lang>             Corpus (es, en) [default: es].
  -m <model>            Model to use [default: basemf]:
                          basemf: Most frequent sentiment
                          clf: Machine Learning Classifier
  -c <clf>              Classifing model to use [default: svm]:
                          maxent: Maximum Entropy (i.e. Logistic Regression)
                          svm: Support Vector Machine
                          mnb: Multinomial Bayes
  -e <vec>              Use word embeddings from file <vec>.
  -n --nobow            Do not use bag-of-words.
  -o <file>             Output model file.
  -h --help             Show this screen.
"""
from docopt import docopt
import pickle
import numpy as np

from hate.settings import corpora
from hate.baselines import MostFrequent
from hate.classifier import HateClassifier


models = {
    'basemf': MostFrequent,
    'clf': HateClassifier,
    # 'mlp': MLP,
}


def get_training_data(opts):
    # load corpora
    train_corpus = 'train_{}'.format(opts['-l'])
    reader = corpora[train_corpus]
    X, y = list(reader.X()), list(reader.y())
    sample_weight = [1.0 for i in range(len(X))]

    return X, y, np.array(sample_weight)


def get_model_params(opts):
    model_type = opts['-m']
    if model_type in {'clf', 'none'}:
        params = {
            'clf': opts['-c'],
            'bow': not opts['--nobow'],
            'embeddings': opts['-e'],
        }
    elif model_type == 'mlp':
        params = {
            'path_to_embeddings': opts['-e'],
            'first_layer': 128,
            'second_layer': 128,
            'scale': False,
            'dropout': [float(d) for d in opts['--dropout'].split(',')],
        }
    elif model_type.startswith('cas'):
        params = {
            'none_clf': opts['-c'],
            'mode': 'split' if model_type == 'cass' else 'overlap',
        }
    else:
        params = {}  # baseline
    external_params = [
        'bow_params',  # word ngrams
        'boc_params',  # character ngrams
        'emb_params',  # sentence embeddings
        'clf_params',  # classifier
    ]
    for p in external_params:
        if p in globals():
            params[p] = globals()[p]
            print('Found external parameters: {}'.format(params[p]))

    print("Training model {} with params {}".format(model_type, params))
    return params


def create_model(opts):
    # instantiate and train model
    model_type = opts['-m']
    params = get_model_params(opts)

    model = models[model_type](**params)

    return model


def save_model(model, filename):
    if hasattr(model, 'save'):
        model.save(filename)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    print("Model saved at {}".format(filename))


def get_fit_params(opts):
    model_type = opts['-m']

    if model_type == 'mlp':
        args = {
            'epochs': int(opts['--epochs']),
            'batch_size': int(opts['--batch-size']),
        }
    else:
        args = {}
    # print("Fitting with args = {}".format(args))
    return args


if __name__ == '__main__':
    opts = docopt(__doc__)

    X, y, sample_weight = get_training_data(opts)

    model = create_model(opts)
    fit_params = get_fit_params(opts)

    model.fit(X, y, sample_weight=sample_weight, **fit_params)

    # save model
    filename = opts['-o']

    save_model(model, filename)
