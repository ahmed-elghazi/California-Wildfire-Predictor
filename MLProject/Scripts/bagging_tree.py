import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from scipy.stats import mode
import joblib
import os

class BaggingDecisionTree:
    """
    A Bagging classifier using Decision Trees as base estimators.
    """
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        """
        Initializes the BaggingDecisionTree classifier.

        Args:
            n_estimators (int): The number of base estimators (trees) in the ensemble.
            max_depth (int, optional): The maximum depth of the individual decision trees.
                                       Defaults to None (nodes expanded until pure or min_samples_split).
            random_state (int, optional): Controls the randomness of the bootstrap sampling.
                                          Defaults to None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self._rng = np.random.RandomState(random_state) # Internal RNG for reproducibility

    def fit(self, X, y):
        """
        Builds a forest of trees from the training set (X, y).

        Args:
            X (array-like of shape (n_samples, n_features)): The training input samples.
            y (array-like of shape (n_samples,)): The target values.
        """
        self.estimators_ = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Create bootstrap sample
            indices = self._rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self._rng.randint(np.iinfo(np.int32).max))
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predicts class for X.

        The predicted class of an input sample is computed as the majority vote
        of the predictions from the individual trees in the forest.

        Args:
            X (array-like of shape (n_samples, n_features)): The input samples.

        Returns:
            array-like of shape (n_samples,): The predicted classes.
        """
        if not self.estimators_:
            raise ValueError("Estimator has not been fitted yet.")

        # Collect predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # Aggregate predictions using majority vote
        # scipy.stats.mode returns mode and count, we only need the mode ([0])
        majority_vote, _ = mode(predictions, axis=0, keepdims=False)
        return majority_vote

# --- Main Execution ---
if __name__ == "__main__":
    # Define data directory relative to the script location
    DATA_DIR = 'data'
    MODEL_FILENAME = 'models/bagging_tree_model.joblib'

    # 1. Load Data
    print("Loading data...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        print(f"Data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}, "
              f"X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Ensure '{DATA_DIR}' directory exists in the same directory as the script "
              f"and contains X_train.npy, y_train.npy, X_test.npy, y_test.npy")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)


    # 2. Initialize and Train the Bagging Model
    print("Initializing and training the Bagging Decision Tree model...")
    # You can adjust n_estimators and max_depth
    bagging_model = BaggingDecisionTree(n_estimators=50, max_depth=10, random_state=42)
    bagging_model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Evaluate the Model
    print("Evaluating the model on the test set...")
    y_pred = bagging_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # 4. Save the Model using joblib
    print(f"Saving the trained model to '{MODEL_FILENAME}'...")
    try:
        joblib.dump(bagging_model, MODEL_FILENAME)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")

    print("\nVerifying model loading...")
    try:
        loaded_model = joblib.load(MODEL_FILENAME)
        y_pred_loaded = loaded_model.predict(X_test)
        accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
        print(f"Loaded Model Accuracy on Test Set: {accuracy_loaded:.4f}")
        assert np.array_equal(y_pred, y_pred_loaded), "Predictions from loaded model do not match."
        print("Model loading verified successfully.")
    except Exception as e:
        print(f"Error loading or verifying the model: {e}")