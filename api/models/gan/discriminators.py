import tensorflow as tf
from collections import namedtuple

from .base import TreatmentNetwork


class CounterfactualDiscriminator(TreatmentNetwork):
    def __init__(self, dim, dim_outcome, h_dim1, h_dim2):
        self.theta = self._calculate_theta(dim, dim_outcome, h_dim1, h_dim2)
        self._loss = None
        self._solver = None
        self.d_logit = None

    def _calculate_theta(self, dim, dim_outcome, h_dim1, h_dim2):
        # Inputs: X + Factual Outcomes + Estimated Counterfactual Outcomes
        W1, b1 = self.layer_coefficients(dim + dim_outcome, h_dim1)

        W2, b2 = self.layer_coefficients(h_dim1, h_dim2)
        W3, b3 = self.layer_coefficients(h_dim2, 1)

        Theta = namedtuple('Theta', 'W1, W2, W3, b1, b2, b3')
        theta = Theta(W1, W2, W3, b1, b2, b3)
        return theta

    def objective(self, X, T, Y, Y_tilde):
        """
        The discriminator maps pair from the feature vector X, and the potential outcome vector
        Y_tilde (x, y^tilde) -> [0,1]^k where i_k = P(y^tilde_i is factual outcome)

        :param X: the feature vector
        :type X:
        :param T: the treatment vector
        :type T:
        :param Y: the factual outcome Y
        :type Y:
        :param Y_tilde: potential outcome vector
        :type Y_tilde:

        :return d_logit:
        :type d_logit:
        """
        # Factual & Counterfactual outcomes concatenate
        inp0 = (1. - T) * Y + T * tf.reshape(Y_tilde[:, 0], [-1, 1])
        inp1 = T * Y + (1. - T) * tf.reshape(Y_tilde[:, 1], [-1, 1])

        inputs = tf.concat(axis=1, values=[X, inp0, inp1])

        h1 = tf.nn.relu(tf.matmul(inputs, self.theta.W1) + self.theta.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.theta.W2) + self.theta.b2)
        self.d_logit = tf.matmul(h2, self.theta.W3) + self.theta.b3

        return self.d_logit

    def calculate_loss(self, t):
        self._loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=t,
                                                    logits=self.d_logit))

    def create_solver(self):
        if tf.is_tensor(self.loss):
            self._solver = tf.train.AdamOptimizer().minimize(self.loss,
                                                             var_list=list(
                                                                 self.theta))
        else:
            raise NameError("Loss must be calculated before solver is called.")