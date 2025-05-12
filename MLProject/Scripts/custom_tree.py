import numpy as np
from collections import Counter

class Node:
    """
    A node in the decision tree.
    
    Attributes:
        feature_index (int): Index of the feature used for splitting at this node.
        threshold (float): Threshold value for the feature split.
        left_child (Node): Left child node (samples where feature <= threshold).
        right_child (Node): Right child node (samples where feature > threshold).
        value (any): Predicted class label if this node is a leaf. None otherwise.
    """
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value  # Class label for a leaf node

    def is_leaf_node(self):
        return self.value is not None


class MyDecisionTreeClassifier:
    """
    A decision tree classifier implemented from scratch.

    Parameters:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        criterion (str): Function to measure the quality of a split ('gini' or 'entropy').
    """
    def __init__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.classes_ = None
        self.n_features_ = None

    def _calculate_impurity(self, y):
        """Calculates impurity of a set of labels."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")

    def _gini_impurity(self, y):
        """Calculates Gini impurity."""
        if y.size == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.size
        return 1.0 - np.sum(probabilities**2)

    def _entropy(self, y):
        """Calculates Shannon entropy."""
        if y.size == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.size
        # Add epsilon to prevent log(0)
        probabilities = probabilities[probabilities > 0] # Ensure p > 0 for log
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, parent_y, left_y, right_y):
        """Calculates information gain of a split."""
        parent_impurity = self._calculate_impurity(parent_y)
        
        if parent_y.size == 0: 
            return 0

        weight_left = left_y.size / parent_y.size
        weight_right = right_y.size / parent_y.size
        
        children_impurity = (weight_left * self._calculate_impurity(left_y) +
                             weight_right * self._calculate_impurity(right_y))
        
        return parent_impurity - children_impurity

    def _most_common_label(self, y):
        """Finds the most common label in a set of labels."""
        if y.size == 0: 
            if self.classes_ is not None and len(self.classes_) > 0:
                return self.classes_[0] 
            raise ValueError("Cannot determine most common label for empty set y.")

        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _find_best_split(self, X, y):
        """Finds the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_split_info = None 

        current_impurity = self._calculate_impurity(y)
        if current_impurity == 0: 
            return None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            if len(thresholds) <= 1:
                continue

            for threshold in thresholds:
                left_indices = np.where(X_column <= threshold)[0]
                right_indices = np.where(X_column > threshold)[0]

                if (left_indices.size < self.min_samples_leaf or
                    right_indices.size < self.min_samples_leaf):
                    continue
                
                y_left, y_right = y[left_indices], y[right_indices]
                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split_info = (feat_idx, threshold, left_indices, right_indices)
        
        if best_gain <= 0: 
            return None
            
        return best_split_info


    def _grow_tree(self, X, y, depth=0):
        """Recursively grows the decision tree."""
        n_samples = X.shape[0]
        
        # Calculate class counts for the current node's data, aligned with self.classes_
        # This will be the value if it becomes a leaf node.
        current_node_class_counts = np.zeros(len(self.classes_), dtype=int)
        unique_labels_in_node, counts_in_node = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels_in_node, counts_in_node):
            class_idx = np.where(self.classes_ == label)[0][0]
            current_node_class_counts[class_idx] = count

        # Stopping criteria
        is_pure_node = len(unique_labels_in_node) == 1
        if (depth >= self.max_depth or
            is_pure_node or
            n_samples < self.min_samples_split):
            return Node(value=current_node_class_counts) # Store class counts for the leaf

        split_info = self._find_best_split(X, y)

        # If no valid split found
        if split_info is None:
            return Node(value=current_node_class_counts) # Store class counts for the leaf

        feature_idx, threshold, left_indices, right_indices = split_info
        
        left_child = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(feature_index=feature_idx, threshold=threshold,
                    left_child=left_child, right_child=right_child)

    def fit(self, X, y):
        """Builds the decision tree classifier from the training set (X, y)."""
        if X.shape[0] == 0:
            raise ValueError("Input X must have at least one sample.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        """Traverses the tree to predict the class for a single sample."""
        if node.is_leaf_node():
            # node.value is an array of class counts
            # Return the class with the highest count
            return self.classes_[np.argmax(node.value)]
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)

    def predict(self, X):
        """Predicts class labels for samples in X."""
        if self.root is None:
            raise RuntimeError("The decision tree has not been fitted yet.")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        predictions = [self._traverse_tree(sample, self.root) for sample in X]
        return np.array(predictions)

    def _traverse_tree_proba(self, x, node):
        """Traverses the tree to get class probabilities for a single sample."""
        if node.is_leaf_node():
            # node.value is an array of class counts
            counts = node.value
            total_samples_at_leaf = np.sum(counts)
            if total_samples_at_leaf == 0:
                # Should not happen if min_samples_leaf >= 1
                # Return uniform probability as a fallback
                return np.full(len(self.classes_), 1.0 / len(self.classes_))
            probabilities = counts / total_samples_at_leaf
            return probabilities
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree_proba(x, node.left_child)
        else:
            return self._traverse_tree_proba(x, node.right_child)

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Parameters:
            X (np.ndarray): Input samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) with class probabilities.
                        The order of classes corresponds to self.classes_.
        """
        if self.root is None:
            raise RuntimeError("The decision tree has not been fitted yet.")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        
        probabilities = [self._traverse_tree_proba(sample, self.root) for sample in X]
        return np.array(probabilities)

