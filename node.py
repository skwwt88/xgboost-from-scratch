import numpy as np

def get_default_settings():
    settings = {}
    settings['gamma'] = 0
    settings['lambda'] = 1
    settings['max_depth'] = 5
    settings['min_item_count'] = 1

    return settings

class Node:
    def __init__(self, x, y, grad, hess, depth, lambda_):
        self.x = x
        self.y = y
        self.grad = grad
        self.hess = hess
        self.depth = depth
        self.is_leaf = True
        self.pivot = None
        self.split_col_index = None
        self.left_node = None
        self.right_node = None
        self.lambda_ = lambda_

    def predict(self, x):
        return np.array([self.predict_single_x(single_x) for single_x in x])

    def predict_single_x(self, single_x):
        if self.is_leaf:
            return - np.sum(self.grad) / (np.sum(self.hess) + self.lambda_)
        else:
            return self.left_node.predict_single_x(single_x) if single_x[self.split_col_index] <= self.pivot else self.right_node.predict_single_x(single_x)

