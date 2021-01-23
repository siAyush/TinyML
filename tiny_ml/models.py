import numpy as np
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cdist


class LinearRegression():
    def fit(self, x, y):
        self.weight = np.linalg.lstsq(x, y, rcond=None)
    def predict(self, x):
        return x.dot(self.weight)


class LogisticRegression():
    def fit(self, x, y, n_iter=1000, lr=0.01):
        self.weight = np.random.rand(x.shape[1])
        for i in range(n_iter):
            self.weight -= lr * (self.predict(x)-y).dot(x)
    def predict(self, x):
        return sigmoid(x.dot(self.weight))


class KNN():
    def predict(self, k, xt, x, y):
        idx = np.argsort(cdist(xt, x))[:, :k]
        y_pred = [np.bincount(y[i]).argmax() for i in idx]
        return y_pred