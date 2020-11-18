from node import Node
import numpy as np

class NaiveNodeLevelSplitRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, gamma = 0, lambda_ = 1, max_depth = 3, min_item_count = 3):
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

        for _ in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = Node(x, y, grad, hess, 0, self.lambda_, self.gamma)
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
            self.lambda_,
            self.gamma
        )

        node.right_node = Node(
            node.x[right_tree_mask],
            node.y[right_tree_mask],
            node.grad[right_tree_mask],
            node.hess[right_tree_mask],
            node.depth + 1,
            self.lambda_,
            self.gamma
        )

        node.pivot = pivot
        node.split_col_index = split_col_index
        node.is_leaf = False

        self.split_node(node.left_node)
        self.split_node(node.right_node)

    def find_pivot(self, node):
        max_gain = 0
        current_objective_value = node.current_objective_value
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
                
                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + self.lambda_) + ((node.total_grad - left_grad) ** 2) / (node.total_hess - left_hess + self.lambda_)) + 2 * self.gamma
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

class SameLevelSplitRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, gamma = 0, lambda_ = 1, max_depth = 3, min_item_count = 3):
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

        pre_sort_record = [np.argsort(x[:, col_index]) for col_index in range(x.shape[1])]

        for _ in range(self.n_estimators):
            level = 0
            data_node_index = [0 for _ in range(x.shape[0])]

            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = Node(x, y, grad, hess, 0, self.lambda_, self.gamma)
            data_node_index = [0 for _ in range(x.shape[0])]
            self.split_nodes_by_level([estimator], data_node_index, x, y, grad, hess, level, pre_sort_record)

            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)

    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred

    def split_nodes_by_level(self, nodes, data_node_index, x, y, grad, hess, level, pre_sort_record):
        if self.max_depth <= level:
            return

        for col_index, sort_index in enumerate(pre_sort_record):
            # record = (left_grad, left_hess)
            tmp_records = [(0.0, 0.0, 0) for _ in range(len(nodes))]

            for index in sort_index:
                currentNodeIndex = data_node_index[index]
                if currentNodeIndex < 0:
                    continue

                record = tmp_records[data_node_index[index]]
                left_grad = record[0] + grad[index]
                left_hess = record[1] + hess[index]
                left_count = record[2] + 1
                tmp_records[data_node_index[index]] = (left_grad, left_hess, left_count)
                currentNode = nodes[currentNodeIndex]

                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + self.lambda_) + ((currentNode.total_grad - left_grad) ** 2) / (currentNode.total_hess - left_hess + self.lambda_)) + 2 * self.gamma
                gain = currentNode.current_objective_value - next_objective_value

                if gain <= currentNode.max_gain or left_count < self.min_item_count or left_count + self.min_item_count > currentNode.x.shape[0]:
                    continue
                
                currentNode.split_col_index = col_index
                currentNode.pivot = x[index, col_index]
                currentNode.max_gain = gain

        for node in nodes:
            self.split_node(node)

        next_nodes = []
        next_nodes_start = []
        leaves_count = 0
        for node in nodes:
            if node.is_leaf:
                next_nodes_start.append(None)
            else: 
                next_nodes_start.append(leaves_count)
                next_nodes.append(node.left_node)
                next_nodes.append(node.right_node)
                leaves_count += 2
        
        next_data_node_index = [-1 for _ in range(x.shape[0])]
        for index in range(x.shape[0]):
            currentNodeIndex = data_node_index[index]
            if currentNodeIndex < 0 or nodes[currentNodeIndex].is_leaf:
                continue
                
            if x[index, nodes[currentNodeIndex].split_col_index] > nodes[currentNodeIndex].pivot:
                next_data_node_index[index] = next_nodes_start[currentNodeIndex] + 1
            else:
                next_data_node_index[index] = next_nodes_start[currentNodeIndex]
        
        self.split_nodes_by_level(next_nodes, next_data_node_index, x, y, grad, hess, level + 1, pre_sort_record)
            
    def split_node(self, node):
        if not node.pivot:
            return
        
        x_by_col = node.x[:, [node.split_col_index]]
        left_tree_mask = (x_by_col <= node.pivot).reshape(-1)
        right_tree_mask = (x_by_col > node.pivot).reshape(-1)

        node.left_node = Node(
            node.x[left_tree_mask],
            node.y[left_tree_mask],
            node.grad[left_tree_mask],
            node.hess[left_tree_mask],
            node.depth + 1,
            self.lambda_,
            self.gamma
        )

        node.right_node = Node(
            node.x[right_tree_mask],
            node.y[right_tree_mask],
            node.grad[right_tree_mask],
            node.hess[right_tree_mask],
            node.depth + 1,
            self.lambda_,
            self.gamma
        )

        node.is_leaf = False

    def grad(self, y0, y1):
        return y1 - y0
    
    def hess(self, y):
        return np.full((y.shape), 1)

class HistogramNodeLevelSplitRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, gamma = 0, lambda_ = 1, max_depth = 5, bin_count = 32, min_hess = 3):
        self.eta = eta
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.bin_count = bin_count
        self.min_hess = min_hess

        self.trees = []

    def fit(self, x, y):
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        self.trees = []

        for _ in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y)
            estimator = Node(x, y, grad, hess, 0, self.lambda_, self.gamma)
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
            self.lambda_,
            self.gamma
        )

        node.right_node = Node(
            node.x[right_tree_mask],
            node.y[right_tree_mask],
            node.grad[right_tree_mask],
            node.hess[right_tree_mask],
            node.depth + 1,
            self.lambda_,
            self.gamma
        )

        node.pivot = pivot
        node.split_col_index = split_col_index
        node.is_leaf = False

        self.split_node(node.left_node)
        self.split_node(node.right_node)

    def find_pivot(self, node):
        max_gain = 0
        current_objective_value = node.current_objective_value
        pivot = None
        split_column_index = None

        for col_index in range(node.x.shape[1]):
            x_by_col = node.x[:, col_index]
            min_value = x_by_col.min()
            max_value = x_by_col.max()

            if max_value - min_value == 0:
                continue

            interval = ((x_by_col.max() - min_value) / self.bin_count )

            histogram_grad = [0 for _ in range(self.bin_count)]
            histogram_hess = [0 for _ in range(self.bin_count)]

            for row_index, x_value in enumerate(x_by_col):
                histogram_grad[min(self.bin_count - 1, int((x_value - min_value) // interval))] += node.grad[row_index]
                histogram_hess[min(self.bin_count - 1, int((x_value - min_value) // interval))] += node.hess[row_index]
            
            left_grad = 0
            left_hess = 0

            for bin_index in range(self.bin_count - 1):
                left_grad += histogram_grad[bin_index]
                left_hess += histogram_hess[bin_index]
                
                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + self.lambda_) + ((node.total_grad - left_grad) ** 2) / (node.total_hess - left_hess + self.lambda_)) + 2 * self.gamma
                gain = current_objective_value - next_objective_value

                if gain <= max_gain or left_hess < self.min_hess or node.total_hess - left_hess < self.min_hess:
                    continue
                
                split_column_index = col_index
                pivot = (bin_index + 1) * interval + min_value
                max_gain = gain

        return pivot, split_column_index

    def grad(self, y0, y1):
        return y1 - y0
    
    def hess(self, y):
        return np.full((y.shape), 1)