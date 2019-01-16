import keras
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score
)
from IPython.display import clear_output
import matplotlib.pyplot as plt


class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.macro_f1 = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        X_val, y_val = self.validation_data[:2]

        y_pred = self.model.predict_classes(X_val)
        y_true = np.argmax(y_val, axis=1)

        _, _, macro_f1 = calculate_metrics(y_true, y_pred)

        self.macro_f1.append(macro_f1)

        clear_output(wait=True)

        plt.subplot(2, 2, 1)

        plt.yscale('log')
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()

        plt.subplot(2, 2, 2)

        plt.plot(self.x, self.acc, label="accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()

        plt.subplot(2, 2, 3)

        plt.plot(self.x, self.macro_f1, label="validation macro f1")
        plt.legend()

        plt.show()


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

def print_evaluation(model, tweets, y):
    loss, accuracy = model.evaluate(tweets, y)
    y_pred = model.predict(tweets) >= 0.5

    precision, recall, f1 = calculate_metrics(
        y, y_pred
    )
    print("Loss        : {:.4f}".format(loss))
    print("Accuracy    : {:.4f}".format(accuracy))
    print("Precision   : {:.4f}".format(precision))
    print("Recall      : {:.4f}".format(recall))
    print("F1          : {:.4f}".format(f1))


    
def load_embedding(path):
    """
    Load embedding from .vec file
    """
    word_to_vec = {}

    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype="float32")
            except:
                print("Problema con la sig línea:")
                print(values[:10])
                word = values[1]
                vec = np.asarray(values[2:], dtype="float32")
            word_to_vec[word] = vec
    return word_to_vec
