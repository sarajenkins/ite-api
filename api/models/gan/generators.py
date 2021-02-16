"""
Module containing Generators used in models
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf

from .base import TreatmentNetwork


class CounterfactualGenerator(TreatmentNetwork):
    """
    Counterfactual Generator to be used in GANITE
    """
    def __init__(self, dim: int, dim_outcome: int, h_dim1: int, h_dim2: int):
        self.theta = self._calculate_theta(dim, dim_outcome, h_dim1, h_dim2)
        self._loss = None
        self._solver = None

    def _calculate_theta(self, dim: int, dim_outcome: int, h_dim1: int,
                         h_dim2: int):
        # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector (Z)
        W1, b1 = self.layer_coefficients(dim + dim_outcome, h_dim1)
        W2, b2 = self.layer_coefficients(h_dim1, h_dim2)

        # Output: Estimated Potential Outcomes
        W31, b31 = self.layer_coefficients(h_dim2, h_dim2)
        W32, b32 = self.layer_coefficients(h_dim2, 1)
        W41, b41 = self.layer_coefficients(h_dim2, h_dim2)
        W42, b42 = self.layer_coefficients(h_dim2, 1)

        Theta = namedtuple(
            'Theta', 'W1, W2, W31, W32, W41, W42, b1, b2, b31, b32, b41, b42')
        theta = Theta(W1, W2, W31, W32, W41, W42, b1, b2, b31, b32, b41, b42)
        return theta

    def objective(self, X: np.ndarray, T: np.ndarray,
                  Y: np.ndarray) -> np.ndarray:
        """
        The counterfactual generator uses the feature vector X, the treatment vector T,
        the factual outcome Y to generate potential outcome vector y_tilde

        :param X: the feature vector
        :type X: np.ndarray
        :param T: the treatment vector
        :type T: np.ndarray
        :param Y: the factual outcome Y
        :type Y: np.ndarray

        :return Y_tilde: potential outcome vector
        :rtype Y_tilde: np.ndarray
        """

        inputs = tf.concat(axis=1, values=[X, T, Y])
        h1 = tf.nn.relu(tf.matmul(inputs, self.theta.W1) + self.theta.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.theta.W2) + self.theta.b2)

        h31 = tf.nn.relu(tf.matmul(h2, self.theta.W31) + self.theta.b31)
        prob1 = (tf.matmul(h31, self.theta.W32) + self.theta.b32)

        h41 = tf.nn.relu(tf.matmul(h2, self.theta.W41) + self.theta.b41)
        prob2 = (tf.matmul(h41, self.theta.W42) + self.theta.b42)

        Y_tilde = tf.nn.sigmoid(tf.concat(axis=1, values=[prob1, prob2]))

        return Y_tilde

    def set_loss(self, d_loss, Y, T, Tilde, alpha):
        """
        Counterfactual Block Generator loss function defined in 
        https://openreview.net/pdf?id=ByKWUeWA- Section 4.1
        """
        loss_GAN = -d_loss
        loss_R = tf.reduce_mean(
            tf.losses.mean_squared_error(
                Y, (T * tf.reshape(Tilde[:, 1], [-1, 1]) +
                    (1. - T) * tf.reshape(Tilde[:, 0], [-1, 1]))))

        self._loss = loss_R + alpha * loss_GAN

    def set_solver(self):
        if tf.is_tensor(self.loss):
            self._solver = tf.train.AdamOptimizer().minimize(self.loss,
                                                             var_list=list(
                                                                 self.theta))
        else:
            raise NameError("Loss must be calculated before solver is called.")


class ITEGenerator(TreatmentNetwork):
    def __init__(self, dim, h_dim1, h_dim2):
        self.theta = self._calculate_theta(dim, h_dim1, h_dim2)
        self._solver = None
        self._loss = None

    def _calculate_theta(self, dim, h_dim1, h_dim2):
        W1, b1 = self.layer_coefficients(dim, h_dim1)
        W2, b2 = self.layer_coefficients(h_dim1, h_dim2)

        W31, b31 = self.layer_coefficients(h_dim2, h_dim2)
        W32, b32 = self.layer_coefficients(h_dim2, 1)
        W41, b41 = self.layer_coefficients(h_dim2, h_dim2)
        W42, b42 = self.layer_coefficients(h_dim2, 1)

        Theta = namedtuple(
            'Theta', 'W1, W2, W31, W32, W41, W42, b1, b2, b31,  b32, b41, b42')
        theta = Theta(W1, W2, W31, W32, W41, W42, b1, b2, b31, b32, b41, b42)
        return theta

    def objective(self, X: np.ndarray) -> np.ndarray:
        """
        The ITE generator uses only the feature vector, x, to generate a potential outcome vector
        y_caret

        :param: x: the feature vector
        :type x: np.ndarray

        :return y_caret: the potential outcome vector
        :rtype y_caret: np.ndarray
        """
        h1 = tf.nn.relu(tf.matmul(X, self.theta.W1) + self.theta.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.theta.W2) + self.theta.b2)

        h31 = tf.nn.relu(tf.matmul(h2, self.theta.W31) + self.theta.b31)
        prob1 = (tf.matmul(h31, self.theta.W32) + self.theta.b32)

        h41 = tf.nn.relu(tf.matmul(h2, self.theta.W41) + self.theta.b41)
        prob2 = (tf.matmul(h41, self.theta.W42) + self.theta.b42)

        y_caret = tf.nn.sigmoid(tf.concat(axis=1, values=[prob1, prob2]))

        return y_caret

    def set_loss(self, T: np.ndarray, Y: np.ndarray, Tilde: np.ndarray,
                 Hat: np.ndarray):
        """
        Counterfactual Block Generator loss function defined in 
        https://openreview.net/pdf?id=ByKWUeWA- Section 4.1
        """
        loss1 = tf.reduce_mean(
            tf.losses.mean_squared_error(
                (T) * Y + (1 - T) * tf.reshape(Tilde[:, 1], [-1, 1]),
                tf.reshape(Hat[:, 1], [-1, 1])))
        loss2 = tf.reduce_mean(
            tf.losses.mean_squared_error(
                (1 - T) * Y + (T) * tf.reshape(Tilde[:, 0], [-1, 1]),
                tf.reshape(Hat[:, 0], [-1, 1])))

        self._loss = loss1 + loss2

    def set_solver(self):
        if tf.is_tensor(self.loss):
            self._solver = tf.train.AdamOptimizer().minimize(self.loss,
                                                             var_list=list(
                                                                 self.theta))
        else:
            raise NameError("Loss must be calculated before solver is called.")
