import warnings

import GPy
import numpy as np
import pandas as pd
import scipy
from GPy import Model, Param, kern, likelihoods, util
from GPy.core import GP
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import \
    PosteriorExact as Posterior
from GPy.util import diag
from GPy.util.linalg import dpotrs, pdinv, tdot
from sklearn.neighbors import KNeighborsRegressor


class CMGP:
    # TODO update this
    """
    An implementation of various Gaussian models for Causal inference building on GPy.
    """
    def __init__(self, dim: int = 1, kern='RBF', mkern='ICM'):
        """
        Class constructor. 
        Initialize a GP object for causal inference. 
    
        :dim: the dimension of the input. Default is 1
        :kern: 'RBF' or 'Matern'. Default is the Radial Basis Kernel
        :mkern: For multitask models, can select from IMC and LMC models, default is IMC  
        """
        self.kern_list = ['RBF', 'Matern']
        self.mkern_list = ['ICM', 'LCM']

        self._dim = dim
        self._kern = kern
        self.mkern = mkern
        self.mode = "CMGP"
        self.Bayesian = True
        self.Confidence = True
        self.lik = GPy.likelihoods.Gaussian()

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim):
        if dim < 1 or type(dim) != int:
            raise ValueError('Invalid value for the input dimension! '
                             'Input dimension has to be a positive integer.')
        else:
            self._dim = dim

    @property
    def kern(self):
        return self._kern

    @kern.setter
    def kern(self, kern):
        if kern not in self.kern_list:
            raise ValueError(
                'Invalid value passed for kern argument. '
                'Valid args: {kerns}'.format(kerns=', '.join(self.kern_list)))
        else:
            return self._kern

    def _build_model(self, X, Y, W):
        df = pd.DataFrame(X)
        df['Y'] = Y
        df['W'] = W

        feature_cols = [col for col in df.columns if col not in ['Y', 'W']]

        # Extract data for the first learning task (control population)
        # `````````````````````````````````````````````````````````````````
        X0 = df.loc[df['W'] == 0, feature_cols]
        y0 = df.loc[df['W'] == 0, ['Y']]
        # Extract data for the second learning task (treated population)
        # `````````````````````````````````````````````````````````````````
        X1 = df.loc[df['W'] == 1, feature_cols]
        y1 = df.loc[df['W'] == 1, ['Y']]
        # Create an instance of a GPy Coregionalization model
        # `````````````````````````````````````````````````````````````````
        if self.kern == 'RBF':
            K0 = GPy.kern.RBF(self.dim, ARD=True)
            K1 = GPy.kern.RBF(self.dim, ARD=True)
        elif self.kern == 'Matern':
            K0 = GPy.kern.Matern32(self.dim, ARD=True)
            K1 = GPy.kern.Matern32(self.dim)

        kernel = GPy.util.multioutput.LCM(input_dim=self.dim,
                                          num_outputs=2,
                                          kernels_list=[K0, K1])
        self.model = GPy.models.GPCoregionalizedRegression(X_list=[X0, X1],
                                                           Y_list=[y0, y1],
                                                           kernel=kernel)

        self.initialize_hyperparameters(X, Y, W)

    def fit(self, X, Y, W):
        """
        Optimizes the model hyperparameters using the factual samples for the treated and control arms.
        X has to be an N x dim matrix. 
        
        :X: The input covariates (the features)
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """
        self._build_model(X, Y, W)
        try:
            self.model.optimize('bfgs', max_iters=1)

        except np.linalg.LinAlgError as e:
            print("Covariance matrix not invertible.")
            raise e

    def predict(self, X):
        """
        Infers the treatment effect for a certain set of input covariates. 
        Returns the predicted ITE.
        
        :X: The input covariates at which the outcomes need to be predicted
        """
        if self.dim == 1:
            X_ = X[:, None]
            X_0 = np.hstack(
                [X_, np.reshape(np.array([0] * len(X)), (len(X), 1))])
            X_1 = np.hstack(
                [X_, np.reshape(np.array([1] * len(X)), (len(X), 1))])
            noise_dict_0 = {'output_index': X_0[:, 1:].astype(int)}
            noise_dict_1 = {'output_index': X_1[:, 1:].astype(int)}
            Y_est_0 = self.model.predict(X_0, Y_metadata=noise_dict_0)[0]
            Y_est_1 = self.model.predict(X_1, Y_metadata=noise_dict_1)[0]

        else:

            X_0 = np.array(
                np.hstack(
                    [X, np.zeros_like(X[:, 1].reshape((len(X[:, 1]), 1)))]))
            X_1 = np.array(
                np.hstack(
                    [X, np.ones_like(X[:, 1].reshape((len(X[:, 1]), 1)))]))
            X_0_shape = X_0.shape
            X_1_shape = X_1.shape
            noise_dict_0 = {
                'output_index':
                X_0[:, X_0_shape[1] - 1].reshape((X_0_shape[0], 1)).astype(int)
            }
            noise_dict_1 = {
                'output_index':
                X_1[:, X_1_shape[1] - 1].reshape((X_1_shape[0], 1)).astype(int)
            }
            Y_est_0 = np.array(
                list(self.model.predict(X_0, Y_metadata=noise_dict_0)[0]))
            Y_est_1 = np.array(
                list(self.model.predict(X_1, Y_metadata=noise_dict_1)[0]))

            var_0 = self.model.predict(X_0, Y_metadata=noise_dict_0)
            var_1 = self.model.predict(X_1, Y_metadata=noise_dict_1)

        TE_est = Y_est_1 - Y_est_0

        return TE_est, Y_est_0, Y_est_1

    def initialize_hyperparameters(self, X, Y, W):
        """
        Initializes the multi-tasking model's hyper-parameters before passing to the optimizer
        
        :param X: The input covariates
        :type X:
        :param Y: The corresponding outcomes
        :type Y:
        :param T: The treatment assignments
        :type T:
        """
        # -----------------------------------------------------------------------------------
        # Output Parameters:
        # -----------------
        # :Ls0, Ls1: length scale vectors for treated and control, dimensions match self.dim
        # :s0, s1: noise variances for the two kernels
        # :a0, a1: diagonal elements of correlation matrix 0
        # :b0, b1: off-diagonal elements of correlation matrix 1
        # -----------------------------------------------------------------------------------
        df = pd.DataFrame(X)
        df['Y'] = Y
        df['W'] = W

        feature_cols = [col for col in df.columns if col not in ['Y', 'W']]

        for i in range(2):
            _df = df[df['W'] == i]
            knn = KNeighborsRegressor(n_neighbors=10)
            knn.fit(_df[feature_cols], _df['Y'])
            col_name = 'Yk{i}'.format(i=i)
            df[col_name] = knn.predict(df[feature_cols])

        #`````````````````````````````````````````````````````
        d0 = df[df['W'] == 0]
        d1 = df[df['W'] == 1]
        a0 = np.sqrt(np.mean((d0['Y'] - np.mean(d0['Y']))**2))
        a1 = np.sqrt(np.mean((d1['Y'] - np.mean(d1['Y']))**2))
        b0 = np.mean((df['Yk0'] - np.mean(df['Yk0'])) *
                     (df['Yk1'] - np.mean(df['Yk1']))) / (a0 * a1)
        b1 = b0
        s0 = np.sqrt(np.mean((d0['Y'] - d0['Yk0'])**2)) / a0
        s1 = np.sqrt(np.mean((d1['Y'] - d1['Yk1'])**2)) / a1
        #`````````````````````````````````````````````````````
        self.model.sum.ICM0.rbf.lengthscale = 10 * np.ones(self.dim)
        self.model.sum.ICM1.rbf.lengthscale = 10 * np.ones(self.dim)

        self.model.sum.ICM0.rbf.variance = 1
        self.model.sum.ICM1.rbf.variance = 1
        self.model.sum.ICM0.B.W[0] = b0
        self.model.sum.ICM0.B.W[1] = b0

        self.model.sum.ICM1.B.W[0] = b1
        self.model.sum.ICM1.B.W[1] = b1

        self.model.sum.ICM0.B.kappa[0] = a0**2
        self.model.sum.ICM0.B.kappa[1] = 1e-4
        self.model.sum.ICM1.B.kappa[0] = 1e-4
        self.model.sum.ICM1.B.kappa[1] = a1**2

        self.model.mixed_noise.Gaussian_noise_0.variance = s0**2
        self.model.mixed_noise.Gaussian_noise_1.variance = s1**2
