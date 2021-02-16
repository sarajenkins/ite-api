"""
Module containing base Network to be inherited
by generator and discriminator Networks
"""

from abc import ABC, abstractmethod

import tensorflow as tf


class TreatmentNetwork(ABC):
    @property
    def solver(self):
        if self._solver:
            return self._solver
        else:
            raise ValueError(
                "Solver must be instantiated before it is referenced.")

    @abstractmethod
    def set_solver(self):
        raise NotImplementedError

    @property
    def loss(self):
        if tf.is_tensor(self._loss):
            return self._loss
        else:
            raise ValueError(
                "Loss must be calculated before it is referenced.")

    @abstractmethod
    def set_loss(self):
        raise NotImplementedError

    @staticmethod
    def randomness_vector(num_rows, num_cols):
        stddev = 1. / tf.sqrt(num_rows / 2.)
        return tf.random_normal(shape=[num_rows, num_cols], stddev=stddev)

    #  This probably could have a better name ?
    def layer_coefficients(self, dim1, dim2):
        weight = tf.Variable(self.randomness_vector(dim1, dim2))
        bias = tf.Variable(tf.zeros(shape=[dim2]))
        return weight, bias
