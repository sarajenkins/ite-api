"""
Module 
"""

import os

import numpy as np
import pandas as pd
from scipy.special import expit

OUTCOME_COLS = {'outcome(t=0)’': 'Y_0', 'outcome(t=1)’': 'Y_1'}


class Twins:
    def __init__(self, file_path='twins/Twin_Data.csv'):
        dirname = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(dirname, file_path))
        df.columns = df.columns.str.replace('\'', '')
        df = df.rename(columns=OUTCOME_COLS)
        self.X = self.set_X(df)
        self.Y = self.set_Y(df)

    def set_X(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [col for col in df.columns if col not in ['Y_0', 'Y_1']]
        X = df[feature_cols].values
        return X

    def set_Y(self, df: pd.DataFrame) -> np.ndarray:
        Y = df[['Y_0', 'Y_1']].values
        return Y

    @staticmethod
    def one_year_mortality(data: np.ndarray) -> np.ndarray:
        """
        Adjust twin outcome to reflect one year mortality
        """
        days_in_year = 365.

        for i in range(len(data[0])):
            idx = np.where(data[:, i] > days_in_year)
            data[idx, i] = days_in_year
        data = 1 - (data / days_in_year)

        return data

    @staticmethod
    def treatment_assignment(X: np.ndarray) -> np.ndarray:
        """
        Patient Treatment Assignment
        """
        num_patients, num_features = X.shape
        coef = 0 * np.random.uniform(-0.01, 0.01, size=[num_features, 1])
        Temp = expit(
            np.matmul(X, coef) +
            np.random.normal(0, 0.01, size=[num_patients, 1]))

        Temp = Temp / (2 * np.mean(Temp))

        Temp[Temp > 1] = 1

        T = np.random.binomial(1, Temp, [num_patients, 1])
        T = T.reshape([
            num_patients,
        ])
        return T

    def observable_outcomes(self, opt_y: np.ndarray,
                            T: np.ndarray) -> np.ndarray:
        """
        Observable outcomes
        """
        num_patients, _ = self.X.shape
        Y = np.zeros([num_patients, 1])

        # # Output
        Y = np.transpose(T) * opt_y[:, 1] + np.transpose(1 - T) * opt_y[:, 0]
        Y = np.transpose(Y)
        Y = np.reshape(Y, [
            num_patients,
        ])
        Y = np.reshape(Y, [
            num_patients,
        ])
        return Y
