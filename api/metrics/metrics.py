import numpy as np
import tensorflow as tf


def PEHE(true, estimate, use_tf=False):
    result = None
    if use_tf:
        result = tf.reduce_mean(
            tf.squared_difference((true[:, 1] - true[:, 0]),
                                  (estimate[:, 1] - estimate[:, 0])))
    else:
        result = np.sqrt(
            np.mean((true.reshape((-1, 1)) - estimate.reshape((-1, 1)))**2))

    return result


def ATE(true, estimate, use_tf=False):
    result = None
    if use_tf:
        result = tf.abs(
            tf.reduce_mean(true[:, 1] - true[:, 0]) -
            tf.reduce_mean(estimate[:, 1] - estimate[:, 0]))
    else:
        result = np.abs(
            np.mean(
                true.reshape((-1, 1)) - np.mean(estimate.reshape((-1, 1)))))
    return result