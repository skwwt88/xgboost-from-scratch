import numpy as np

def get_default_settings():
    settings = {}
    settings['gamma'] = 0
    settings['lambda'] = 1
    settings['max_depth'] = 5
    settings['min_item_count'] = 1

    return settings

class Node:
    def __init__(self, x, y, grad, hess, depth, settings = None):
        self.x = x
        self.y = y
        self.grad = grad
        self.hess = hess
        self.depth = depth
        self.is_leaf = True
        self.leaves_count = 1
        self.pivot = None
        self.split_col_index = None
        self.left_node = None
        self.right_node = None

        self.settings = settings if settings else get_default_settings()

    def split_node(self):
        if self.settings['max_depth'] <= self.depth:
            return

        pivot, split_col_index = self.find_pivot()

        if not pivot:
            return
        
        x_by_col = self.x[:, [split_col_index]]
        left_tree_mask = (x_by_col <= pivot).reshape(-1)
        right_tree_mask = (x_by_col > pivot).reshape(-1)

        self.left_node = Node(
            self.x[left_tree_mask],
            self.y[left_tree_mask],
            self.grad[left_tree_mask],
            self.hess[left_tree_mask],
            self.depth + 1,
            self.settings
        )

        self.right_node = Node(
            self.x[right_tree_mask],
            self.y[right_tree_mask],
            self.grad[right_tree_mask],
            self.hess[right_tree_mask],
            self.depth + 1,
            self.settings
        )

        self.pivot = pivot
        self.split_col_index = split_col_index
        self.is_leaf = False

        self.left_node.split_node()
        self.right_node.split_node()

    def find_pivot(self):
        max_gain = 0
        total_grad = np.sum(self.grad)
        total_hess = np.sum(self.hess)
        lambada_ = self.settings['lambda']
        gamma = self.settings['gamma']
        min_item_count = self.settings['min_item_count']

        current_objective_value = -0.5 * (total_grad ** 2 / (total_hess + lambada_)) + gamma
        pivot = None
        split_column_index = None

        for col_index in range(self.x.shape[1]):
            x_by_col = self.x[:, col_index]
            sorted_x_index = np.argsort(x_by_col)
            
            left_grad = 0
            left_hess = 0

            for i, row_index in enumerate(sorted_x_index):
                left_grad += self.grad[row_index]
                left_hess += self.hess[row_index]
                
                next_objective_value = -0.5 * (left_grad ** 2 / (left_hess + lambada_) + ((total_grad - left_grad) ** 2) / (total_hess - left_hess + lambada_)) + 2 * gamma
                gain = current_objective_value - next_objective_value

                if gain <= max_gain or i  + 1 < min_item_count or i + min_item_count > len(sorted_x_index):
                    continue
                
                split_column_index = col_index
                pivot = x_by_col[row_index]
                max_gain = gain

        return pivot, split_column_index

    def predict(self, x):
        return np.array([self.predict_single_x(single_x) for single_x in x])

    def predict_single_x(self, single_x):
        if self.is_leaf:
            return - np.sum(self.grad) / (np.sum(self.hess) + self.settings['lambda'])
        else:
            return self.left_node.predict_single_x(single_x) if single_x[self.split_col_index] <= self.pivot else self.right_node.predict_single_x(single_x)

