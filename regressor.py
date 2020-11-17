from node import Node
import numpy as np

class NaiveNodeLevelSplitRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, gamma = 0, lambda_ = 1, max_depth = 5, min_item_count = 3):
        self.eta = eta
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.min_item_count = min_item_count

        self.trees = []

    def fit(self, x, y):
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        self.trees = []

        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = Node(x, y, grad, hess, 0, self.lambda_)
            self.split_node(estimator)

            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)

    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred

    def split_node(self, node):
        if self.max_depth <= node.depth:
            return

        pivot, split_col_index = self.find_pivot(node)

        if not pivot:
            return
        
        x_by_col = node.x[:, [split_col_index]]
        left_tree_mask = (x_by_col <= pivot).reshape(-1)
        right_tree_mask = (x_by_col > pivot).reshape(-1)

        node.left_node = Node(
            node.x[left_tree_mask],
            node.y[left_tree_mask],
            node.grad[left_tree_mask],
            node.hess[left_tree_mask],
            node.depth + 1,
            self.lambda_
        )

        node.right_node = Node(
            node.x[right_tree_mask],
            node.y[right_tree_mask],
            node.grad[right_tree_mask],
            node.hess[right_tree_mask],
            node.depth + 1,
            self.lambda_
        )

        node.pivot = pivot
        node.split_col_index = split_col_index
        node.is_leaf = False

        self.split_node(node.left_node)
        self.split_node(node.right_node)

    def find_pivot(self, node):
        max_gain = 0
        total_grad = np.sum(node.grad)
        total_hess = np.sum(node.hess)

        current_objective_value = -0.5 * (total_grad ** 2 / (total_hess + self.lambda_)) + self.gamma
        pivot = None
        split_column_index = None

        for col_index in range(node.x.shape[1]):
            x_by_col = node.x[:, col_index]
            sorted_x_index = np.argsort(x_by_col)
            
            left_grad = 0
            left_hess = 0

            for i, row_index in enumerate(sorted_x_index):
                left_grad += node.grad[row_index]
                left_hess += node.hess[row_index]
                
                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + self.lambda_) + ((total_grad - left_grad) ** 2) / (total_hess - left_hess + self.lambda_)) + 2 * self.gamma
                gain = current_objective_value - next_objective_value

                if gain <= max_gain or i  + 1 < self.min_item_count or i + self.min_item_count > len(sorted_x_index):
                    continue
                
                split_column_index = col_index
                pivot = x_by_col[row_index]
                max_gain = gain

        return pivot, split_column_index

    def grad(self, y0, y1):
        return y1 - y0
    
    def hess(self, y):
        return np.full((y.shape), 1)

class NodeWithLevelInfo(Node):
    def __init__(self, x, y, grad, hess, depth, lambda_):
        Node.__init__(self, x, y, grad, hess, depth, lambda_)



class SameLevelSplitRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, gamma = 0, lambda_ = 1, max_depth = 5, min_item_count = 3):
        self.eta = eta
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.min_item_count = min_item_count

        self.trees = []

    def fit(self, x, y):
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        self.trees = []

        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = NodeWithLevelInfo(x, y, grad, hess, 0, self.lambda_)
            self.split_node([estimator])

            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)

    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred

    def split_node(self, nodes):
        if self.max_depth <= node.depth:
            return

        pivot, split_col_index = self.find_pivot(node)

        if not pivot:
            return
        
        x_by_col = node.x[:, [split_col_index]]
        left_tree_mask = (x_by_col <= pivot).reshape(-1)
        right_tree_mask = (x_by_col > pivot).reshape(-1)

        node.left_node = Node(
            node.x[left_tree_mask],
            node.y[left_tree_mask],
            node.grad[left_tree_mask],
            node.hess[left_tree_mask],
            node.depth + 1,
            self.lambda_
        )

        node.right_node = Node(
            node.x[right_tree_mask],
            node.y[right_tree_mask],
            node.grad[right_tree_mask],
            node.hess[right_tree_mask],
            node.depth + 1,
            self.lambda_
        )

        node.pivot = pivot
        node.split_col_index = split_col_index
        node.is_leaf = False

        self.split_node(node.left_node)
        self.split_node(node.right_node)

    def find_pivot(self, node):
        max_gain = 0
        total_grad = np.sum(node.grad)
        total_hess = np.sum(node.hess)

        current_objective_value = -0.5 * (total_grad ** 2 / (total_hess + self.lambda_)) + self.gamma
        pivot = None
        split_column_index = None

        for col_index in range(node.x.shape[1]):
            x_by_col = node.x[:, col_index]
            sorted_x_index = np.argsort(x_by_col)
            
            left_grad = 0
            left_hess = 0

            for i, row_index in enumerate(sorted_x_index):
                left_grad += node.grad[row_index]
                left_hess += node.hess[row_index]
                
                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + self.lambda_) + ((total_grad - left_grad) ** 2) / (total_hess - left_hess + self.lambda_)) + 2 * self.gamma
                gain = current_objective_value - next_objective_value

                if gain <= max_gain or i  + 1 < self.min_item_count or i + self.min_item_count > len(sorted_x_index):
                    continue
                
                split_column_index = col_index
                pivot = x_by_col[row_index]
                max_gain = gain

        return pivot, split_column_index

    def grad(self, y0, y1):
        return y1 - y0
    
    def hess(self, y):
        return np.full((y.shape), 1)