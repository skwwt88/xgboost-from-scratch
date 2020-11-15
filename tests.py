import unittest
import numpy as np
from node import Node
from regressor import Regressor

x = np.array([[2, 4, 6, 8], [1, 2, 3, 4], [4, 8, 12, 16]])
y = np.array([2, 1, 4])
# pred_y = [2, 2, 2]
grad = np.array([0, 1, -1])
hess = np.array([1, 1, 1])

x_valid = np.array([[3, 6, 9, 12], [2, 2, 3, 4], [3, 6, 10, 12]])

class TreeNodeTest(unittest.TestCase):
    def test_split_node(self):
        node = Node(x, y, grad, hess, 0)
        node.split_node()

class RegressorTest(unittest.TestCase):
    def test_split_node(self):
        regressor = Regressor()
        regressor.fit(x, y)

        print(regressor.predict(x_valid))

if __name__ == '__main__':
    unittest.main()