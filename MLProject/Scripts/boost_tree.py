import numpy as np
from sklearn.metrics import accuracy_score
from joblib import dump
import os

class DecisionTreeStump:
    """A weak classifier for AdaBoost (tree with a single split)."""

    def __init__(self):
        self.feature_index = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for the split
        self.polarity = 1  # Polarity of the split (1 or -1)
        self.alpha = None  # Weight of the stump (used in AdaBoost)

    def fit(self, X, y, sample_weight):
        """Train the stump using weighted data."""
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Iterate over all features and thresholds to find the best split
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    # Make predictions based on the current split
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    # Calculate weighted error
                    error = np.sum(sample_weight * (predictions != y))

                    # Update the stump if the error is smaller
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_index
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        """Make predictions using the trained stump."""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_index]
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions


def load_data(train_path, test_path):
    """Load training and testing data from .npy files."""
    X_train = np.load(train_path + '/X_train.npy')
    y_train = np.load(train_path + '/y_train.npy')
    X_test = np.load(test_path + '/X_test.npy')
    y_test = np.load(test_path + '/y_test.npy')
    return X_train, y_train, X_test, y_test


def train_adaboost(X_train, y_train, n_estimators=50):
    """Train the AdaBoost model."""
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples  # Initialize sample weights uniformly
    models = []  # List to store weak classifiers
    alphas = []  # List to store alpha values (classifier weights)

    for estimator in range(n_estimators):
        # Train a weak classifier (decision tree stump)
        stump = DecisionTreeStump()
        stump.fit(X_train, y_train, sample_weight=weights)
        predictions = stump.predict(X_train)

        # Compute weighted error
        error = np.sum(weights * (predictions != y_train)) / np.sum(weights)

        # Avoid division by zero or invalid log
        if error == 0:
            error = 1e-10
        elif error >= 0.5:
            break  # Stop if the weak classifier is no better than random guessing

        # Compute alpha (classifier weight)
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)
        models.append(stump)

        # Update sample weights
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)  # Normalize weights

    return models, alphas


def adaboost_predict(X, models, alphas):
    """Make predictions using the trained AdaBoost model."""
    final_predictions = np.zeros(X.shape[0])
    for alpha, model in zip(alphas, models):
        final_predictions += alpha * model.predict(X)
    return np.sign(final_predictions)


def save_model(models, alphas, model_path):
    """Save the trained AdaBoost model to a file."""
    dump({'models': models, 'alphas': alphas}, model_path)
    print(f"Model saved to {model_path}")


def main():
    # Paths to training and testing data
    train_path = 'data'  # Replace with actual path
    test_path = 'data'  # Replace with actual path

    # Load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # Train AdaBoost model
    n_estimators = 50
    models, alphas = train_adaboost(X_train, y_train, n_estimators)

    # Evaluate on test data
    y_pred = adaboost_predict(X_test, models, alphas)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Save the trained model
    model_path = os.path.join('models', 'Adaboost_model.joblib')
    save_model(models, alphas, model_path)


if __name__ == "__main__":
    main()
