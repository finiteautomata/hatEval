"""Evaulate a Sentiment Analysis model.

Usage:
  eval.py -i <file> [options]
  eval.py -h | --help

Options:
  -i <file>           Trained model file.
  -l <lang>           Corpus (es, en) [default: es].
  -t --train          Use training set instead of development.
  -r                  Compute ROC AUC on predict_proba or decision_function.
  -s --short          Short output.
  -o <file>           Output run file for TASS submission.
  -h --help           Show this screen.
"""
from docopt import docopt
import pickle
import os

from sklearn.preprocessing import label_binarize

from hate.settings import corpora
from sentiment.evaluator import Evaluator
# from sentiment.mlp import load_model as load_mlp


def load_model(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".json":
        """Load MLP model"""
        return load_mlp(filename)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def get_section(opts):
    if opts['--train']:
        section = 'train_{}'.format(opts['-l'])
    else:
        section = 'dev_{}'.format(opts['-l'])

    return section


if __name__ == '__main__':
    opts = docopt(__doc__)

    if 'model' not in globals():
        # load model
        filename = opts['-i']
        model = load_model(filename)
        # print("Model loaded.")
    else:
        print("Using model {}".format(model))

    section = get_section(opts)
    # print("Evaluating on {}".format(section))
    # load corpora
    reader = corpora[section]
    X, y_true = list(reader.X()), list(reader.y())

    # classify
    y_pred = model.predict(X)
    if opts['-r']:
        # FIXME: broken
        if model._clf.startswith('svm'):
            Y_pred = model.decision_function(X)
        else:
            Y_pred = model.predict_proba(X)

    out_filename = opts['-o']
    if out_filename:
        # FIXME: broken
        f = open(out_filename, 'w')
        for t, x, pred in zip(reader.tweets(), X, y_pred):
            f.write('{}\t{}\n'.format(t['tweetid'], pred))
        f.close()

    # evaluate and print
    labels = ['0', '1']
    evaluator = Evaluator(labels)
    evaluator.evaluate(y_true, y_pred)
    if opts['-r']:
        # FIXME: broken
        Y_true = label_binarize(y_true, model._pipeline.classes_)
        evaluator.roc_auc(Y_true, Y_pred)
        evaluator.rank_error(y_true, Y_pred)
    if opts['--short']:
        evaluator.print_short_results()
    else:
        evaluator.print_results()
        evaluator.print_confusion_matrix()
