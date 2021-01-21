import numpy as np


class LinearRegression():
    def fit(self, x, y):
        self.weight = np.linalg.lstsq(x, y, rcond=None)
    def predict(self, x):
        return x.dot(self.weight)


class LogisticRegression():
    def fit()