from node import Node
import numpy as np

class Regressor:
    def __init__(self, eta = 0.3, n_estimators = 100):
        self.eta = eta
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, x, y):
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        self.trees = []

        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = Node(x, y, grad, hess, 0)
            estimator.split_node()

            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)

    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred

    def grad(self, y0, y1):
        return y1 - y0
    
    def hess(self, y):
        return np.full((y.shape), 1)