import numpy as np
from collections import Counter

# Import necessary libraries
from sklearn.metrics import accuracy_score  # To evaluate model performance
import joblib  # For saving the trained model

# Node class for the decision tree structure
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initialize a node in the decision tree.
        If it's a leaf node, 'value' stores the predicted class.
        If it's a decision node, 'feature' and 'threshold' define the split,
        and 'left' and 'right' point to the child nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Prediction value if it's a leaf node

    def is_leaf_node(self):
        """Check if the node is a leaf node."""
        return self.value is not None

# Manual Decision Tree Classifier class
class ManualDecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        Initialize the classifier.
        min_samples_split: Minimum number of samples required to split a node.
        max_depth: Maximum depth of the tree.
        n_features: Number of features to consider for splitting (can be used for feature subsetting).
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None # Root node of the tree

    def fit(self, X, y):
        """Build the decision tree from the training data."""
        # Determine the number of features to consider at each split
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        # Start the recursive tree building process
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria for recursion
        # 1. Max depth reached
        # 2. Not enough samples to split
        # 3. All samples belong to the same class (pure node)
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            # Create a leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select a subset of features to consider for the split
        # This can help prevent overfitting and is used in algorithms like Random Forests
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split (feature and threshold) that maximizes information gain
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # If no split improves impurity, create a leaf node
        if best_feat is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        # Split the data based on the best feature and threshold
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Recursively build the left and right subtrees
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return the decision node
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Find the best feature and threshold for splitting the data."""
        best_gain = -1
        split_idx, split_thresh = None, None

        # Iterate over the selected features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # Consider unique values in the feature column as potential thresholds
            thresholds = np.unique(X_column)

            # Iterate over potential thresholds
            for threshold in thresholds:
                # Calculate information gain for this split
                gain = self._information_gain(y, X_column, threshold)

                # Update best split if current gain is higher
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """Calculate information gain using Gini impurity."""
        # Calculate parent Gini impurity
        parent_gini = self._gini(y)

        # Split data based on the threshold
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # If split results in empty child, gain is 0
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted average Gini impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        # Information gain is the reduction in impurity
        ig = parent_gini - child_gini
        return ig

    def _split(self, X_column, split_thresh):
        """Split a feature column based on a threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        """Calculate Gini impurity for a set of labels."""
        # Count occurrences of each class label
        hist = np.bincount(y.astype(int)) # Ensure labels are integers for bincount
        # Calculate probabilities of each class
        ps = hist / len(y)
        # Gini impurity formula: 1 - sum(p_i^2)
        return 1 - np.sum([p**2 for p in ps if p > 0])

    def _most_common_label(self, y):
        """Find the most common class label in a set of labels."""
        counter = Counter(y)
        # Handle potential empty input during recursion edge cases
        if not counter:
            return None # Or raise an error, or return a default value
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """Predict class labels for new data."""
        # Traverse the tree for each sample in X
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Recursively traverse the tree to predict the label for a single sample."""
        # If it's a leaf node, return its value
        if node.is_leaf_node():
            return node.value

        # If it's a decision node, compare the sample's feature value with the threshold
        if x[node.feature] <= node.threshold:
            # Go left
            return self._traverse_tree(x, node.left)
        else:
            # Go right
            return self._traverse_tree(x, node.right)

# --- Main Script Execution ---

# Load the training and testing data from separate .npy files
# Replace with your actual file paths
print("Loading data...")
try:
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Please ensure 'X_train.npy', 'y_train.npy', 'X_test.npy', and 'y_test.npy' exist.")
    exit() # Exit if data files are missing

# Ensure labels are integers (required for _gini calculation using bincount)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Initialize the Manual Decision Tree Classifier
# You can adjust hyperparameters like max_depth and min_samples_split
clf = ManualDecisionTreeClassifier(max_depth=10, min_samples_split=5)
print("Training the model...")

# Train the classifier on the training data
clf.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test data
print("Making predictions...")
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model to a file using joblib
# Replace 'manual_decision_tree_model.joblib' with your desired file name
model_filename = 'manual_decision_tree_model.joblib'
joblib.dump(clf, model_filename)
print(f"Model saved successfully to {model_filename}.")