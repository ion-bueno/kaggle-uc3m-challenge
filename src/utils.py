import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import keras.backend as K


########################
# Global variables
########################

seed = 41



########################
# Recurrent functions
########################

def separate_data(train_comp):
    X_train = train_comp.iloc[:,2:]
    y_train = train_comp.iloc[:,1]
    X_train.to_csv(f'raw_data/X_train.csv', index=False)
    y_train.to_csv(f'raw_data/y_train.csv', index=False)


def encoding_labels(y):
    return np.where(y==True, 1, 0)

def decoding_labels(y):
    return np.where(y==1, True, False)


def load_data(df_format=False):
    X_df = pd.read_csv('raw_data/X.csv')
    y_df = pd.read_csv('raw_data/y.csv')
    X_comp_df = pd.read_csv('raw_data/test_set.csv').iloc[:, 1:]
    if df_format:
        return X_df, y_df, X_comp_df

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()
    X_comp = X_comp_df.to_numpy()
    return X, y, X_comp 


def load_model(n_sub):
    return pkl.load(open(f'models/model{n_sub}.pkl', 'rb'))


def split(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def predict_threshold(probs, threshold):
    return np.where(probs > threshold, True, False)


def show_perc_labels(y_pred):
    n_true = (y_pred == True).sum()
    n_false = (y_pred == False).sum()
    print(f'There exists a {round(n_true*100/(n_true+n_false), 2)}% of true samples, having {n_true} true samples and {n_false} false samples.')


def prepare_submission(y_pred, n_sub, export=True, export_model=False, model=None, tf=False):
    ids = np.arange(0, 1750)
    submission = pd.DataFrame(index=ids, data={'ID': ids, 'target': y_pred})

    if(export):
        submission.to_csv(f'submissions/submission{n_sub}.csv', index=False)

    if(export_model):
        if tf:
            model.save(f'models/model{n_sub}')
        else:
            with open(f'models/model{n_sub}.pkl', 'wb') as file:
                pkl.dump(model, file)


def read_submissions():
    files = list( Path( 'submissions/' ).glob( 'submission*' ) )
    submissions = {}

    for p in files:
        aux = pd.read_csv(p, index_col=[0])
        s = re.findall(r'\d+', str(p))[0]

        submissions[s] = aux
    return submissions


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



########################
# Classes
########################

import tensorflow as tf


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom Early Stopping callback to monitor multiple metrics by combining them using a harmonic mean calculation.
    Adapted from (TensorFlow EarlyStopping source)[https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/callbacks.py#L1683-L1823].
    Author: Angel Igareta (angel@igareta.com)
    """
    def __init__(
        self,
        metrics_names=["loss"],
        mode="min",
        patience=0,
        restore_weights=False,
        logdir=None,
    ):
        super(CustomEarlyStopping, self).__init__()
        self.metrics_names = metrics_names
        self.mode = mode
        self.patience = patience
        self.restore_weights = restore_weights
        self.logdir = logdir
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_combined_metric = np.Inf if self.mode=="min" else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        metrics = [logs.get(name) for name in self.metrics_names]
        metrics = tf.cast(metrics, dtype=tf.float32)
        metrics_count = tf.cast(tf.size(metrics), dtype=tf.float32)

        # Combined metric is the harmonic mean of the metrics_names.
        combined_metric = tf.math.divide(
            metrics_count, tf.math.reduce_sum(tf.math.reciprocal_no_nan(metrics))
        )

        # Specify logdir if you want to log the combined metric
        if self.logdir:
            with tf.summary.create_file_writer(self.logdir).as_default():
                tf.summary.scalar("combined_metric", data=combined_metric, step=epoch)

        # If harmonic mean is np.greater or np.less depending on min-max mode.
        if (
            self.mode == "min" and np.less(combined_metric, self.best_combined_metric)
        ) or (
            self.mode == "max"
            and np.greater(combined_metric, self.best_combined_metric)
        ):
            self.best_combined_metric = combined_metric
            self.wait = 0
            # Record the best weights if current results is better.
            self.best_weights = self.model.get_weights()
        else:
            self.wait = 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

                # Restoring model weights from the end of the best epoch
                if self.restore_weights:
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


'''self.callback = CustomEarlyStopping(
    metrics_names=["val_precision", "val_recall"],
    mode="max",
    patience=self.params['patience'],
    restore_weights=True,
)'''