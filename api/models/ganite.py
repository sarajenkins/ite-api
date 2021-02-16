import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import expit
from tqdm import tqdm

from .gan.discriminators import CounterfactualDiscriminator
from .gan.generators import CounterfactualGenerator, ITEGenerator


class GANITE:
    def __init__(self, alpha, mini_batch_size, num_iterations, num_kk, h_dim):
        self.alpha = alpha
        self.mini_batch_size = mini_batch_size
        self.num_iterations = num_iterations
        self.num_kk = num_kk
        self.h_dim = h_dim
        self.sess = None

    @staticmethod
    def sample_X(X, size):
        start_idx = np.random.randint(0, X.shape[0], size)
        return start_idx

    def get_feed_dict(self, X, T, Y):
        idx_mb = self.sample_X(X, self.mini_batch_size)
        X_mb = X[idx_mb, :]
        T_mb = np.reshape(T[idx_mb], [self.mini_batch_size, 1])
        Y_mb = np.reshape(Y[idx_mb], [self.mini_batch_size, 1])
        feed_dict = {self.X: X_mb, self.T: T_mb, self.Y: Y_mb}
        return feed_dict

    def _build_model(self, dim, dim_outcome, h_dim1, h_dim2):
        self.X = tf.placeholder(tf.float32, shape=[None, dim])
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y_T = tf.placeholder(tf.float32, shape=[None, dim_outcome])

        self.g = CounterfactualGenerator(dim, 2, h_dim1, h_dim2)
        self.d = CounterfactualDiscriminator(dim, 2, h_dim1, h_dim2)
        self.i = ITEGenerator(dim, h_dim1, h_dim2)

        # counterfactual outcomes
        self.y_tilde = self.g.objective(self.X, self.T, self.Y)
        self.d.objective(self.X, self.T, self.Y, self.y_tilde)
        self.y_carot = self.i.objective(self.X)

        self.d.calculate_loss(self.T)
        self.g.calculate_loss(self.d.loss, self.Y, self.T, self.y_tilde,
                              self.alpha)
        self.i.calculate_loss(self.T, self.Y, self.y_tilde, self.y_carot)

        self.g.create_solver()
        self.d.create_solver()
        self.i.create_solver()

    def counterfactual_block(self, X, T, Y):
        for it in tqdm(range(self.num_iterations)):
            for kk in range(self.num_kk):
                feed_dict = self.get_feed_dict(X, T, Y)

                _, D_loss_curr = self.sess.run([self.d.solver, self.d.loss],
                                               feed_dict=feed_dict)

            feed_dict = self.get_feed_dict(X, T, Y)

            _, g_loss_curr, Tilde_curr = self.sess.run(
                [self.g.solver, self.g.loss, self.y_tilde],
                feed_dict=feed_dict)

            if it % 100 == 0:
                logging.debug('Iter: {}'.format(it))
                logging.debug('D_loss: {:.4}'.format((D_loss_curr)))
                logging.debug('G_loss: {:.4}'.format((g_loss_curr)))
                logging.debug('')
        return Tilde_curr

    def fit(self, Train_X, Train_T, Train_Y, dim_outcome):
        tf.reset_default_graph()

        dim = len(Train_X[0])
        h_dim1 = self.h_dim
        h_dim2 = self.h_dim

        self._build_model(dim, dim_outcome, h_dim1, h_dim2)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.counterfactual_block(Train_X, Train_T, Train_Y)

        for it in tqdm(range(self.num_iterations)):

            feed_dict = self.get_feed_dict(Train_X, Train_T, Train_Y)

            _, I_loss_curr = self.sess.run([self.i.solver, self.i.loss],
                                           feed_dict=feed_dict)

            # Testing
            if it % 100 == 0:
                logging.debug('Iter: {}'.format(it))
                logging.debug('I_loss: {:.4}'.format((I_loss_curr)))
                logging.debug('')

    def predict(self, X):
        result = self.sess.run([self.y_carot], feed_dict={self.X: X})[0]
        return result

    def test(self, X, Y, metric):
        loss = metric(self.Y_T, self.y_carot)

        loss_output, _ = self.sess.run([loss, self.y_carot],
                                       feed_dict={
                                           self.X: X,
                                           self.Y_T: Y
                                       })

        return loss_output
