import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
# Removed: from sklearn.datasets import make_classification
# Removed: from sklearn.model_selection import train_test_split
# Removed: from sklearn.metrics import accuracy_score
# Removed: from sklearn.ensemble import RandomForestClassifier

# Helper function: Custom train_test_split
def custom_train_test_split(X, y, test_size=0.3, random_state=None):
    """
    Splits arrays or matrices into random train and test subsets.

    Parameters:
    - X (array-like): Data to split.
    - y (array-like): Labels to split.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int, optional): Seed used by the random number generator.

    Returns:
    - X_train, X_test, y_train, y_test (tuple): Split data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    
    n_test_samples = int(n_samples * test_size)
    test_indices = shuffled_indices[:n_test_samples]
    train_indices = shuffled_indices[n_test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Helper function: Custom accuracy_score
def custom_accuracy_score(y_true, y_pred):
    """
    Computes accuracy classification score.

    Parameters:
    - y_true (array-like): Ground truth (correct) labels.
    - y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
    - score (float): Accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

# Helper function: Custom make_classification (simplified)
def custom_make_classification(n_samples=100, n_features=20, random_state=None):
    """
    Generates a very simple binary classification dataset.
    This is a simplified replacement for sklearn.datasets.make_classification.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.rand(n_samples, n_features)
    
    # Create a simple decision rule for generating labels
    # Example: if sum of first half of features > sum of second half, class 1, else class 0
    # This is an arbitrary rule for demonstration purposes.
    mid_point = n_features // 2
    if n_features > 1 :
        y = (np.sum(X[:, :mid_point], axis=1) > np.sum(X[:, mid_point:], axis=1)).astype(int)
    else: # Handle case with only 1 feature
        y = (X[:, 0] > np.median(X[:, 0])).astype(int)

    # Ensure there are two classes, especially for small n_samples or skewed rules
    if len(np.unique(y)) < 2:
        # If all samples fall into one class, force a split for demonstration
        split_idx = n_samples // 2
        y[:split_idx] = 0
        y[split_idx:] = 1
        # Shuffle y to mix the forced labels
        current_rng_state = np.random.get_state() # Save current RNG state
        if random_state is not None: # Use the provided seed for this shuffle if available
            np.random.seed(random_state)
        np.random.shuffle(y)
        np.random.set_state(current_rng_state) # Restore RNG state

    return X, y

class MyRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        """
        A custom implementation of the Random Forest classifier.

        Parameters:
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until
          all leaves are pure or until all leaves contain less than min_samples_split samples.
        - min_samples_split (int): The minimum number of samples required to split an internal node.
        - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        - max_features (str, int, float): The number of features to consider when looking for the best split:
            - If int, then consider max_features features at each split.
            - If float, then max_features is a percentage and int(max_features * n_features) features are considered.
            - If "sqrt", then max_features=sqrt(n_features).
            - If "log2", then max_features=log2(n_features).
            - If None, then max_features=n_features (not typical for Random Forest).
        - random_state (int, optional): Controls both the randomness of the bootstrapping of the samples used
          when building trees and the sampling of the features to consider when looking for the best split at each node.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.trees = []
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The training input samples.
        - y (array-like of shape (n_samples,)): The target values.
        """
        self.classes_ = np.unique(y)
        n_samples, self.n_features_in_ = X.shape
        self.trees = []

        if self.random_state is not None:
            master_rng = np.random.RandomState(self.random_state)
        else:
            master_rng = np.random.RandomState()

        for _ in range(self.n_estimators):
            bootstrap_seed = master_rng.randint(0, 2**32 - 1)
            tree_internal_seed = master_rng.randint(0, 2**32 - 1)

            current_bootstrap_rng = np.random.RandomState(bootstrap_seed)
            bootstrap_indices = current_bootstrap_rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_internal_seed
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self

    def _most_common_label(self, y_slice):
        counter = Counter(y_slice)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class for X.
        """
        if not self.trees:
            raise ValueError("Estimator not fitted, call fit before predict.")

        tree_predictions_list = [tree.predict(X) for tree in self.trees]
        tree_predictions_arr = np.array(tree_predictions_list).T 
        
        y_pred = np.array([self._most_common_label(sample_predictions) 
                           for sample_predictions in tree_predictions_arr])
        
        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        if not self.trees:
            raise ValueError("Estimator not fitted, call fit before predict_proba.")

        tree_probas_list = [tree.predict_proba(X) for tree in self.trees]
        tree_probas_arr = np.stack(tree_probas_list, axis=0)
        mean_probas = np.mean(tree_probas_arr, axis=0)
        
        return mean_probas