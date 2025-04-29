"""
Module: knn_module.py
Provides KNNClassifier and BaggingKNN classes for use by a main script.
"""
import numpy as np

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

class KNNClassifier:
    """
    K-Nearest Neighbors classifier.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y, dtype=int)

    def predict(self, X):
        """Predict labels for each row in X."""
        X = np.array(X)
        preds = [self._predict_point(x) for x in X]
        return np.array(preds, dtype=int)

    def _predict_point(self, x):
        # Compute Euclidean distances
        dists = np.linalg.norm(self.X_train - x, axis=1)
        # Indices of k smallest distances
        idx = np.argsort(dists)[:self.k]
        # Majority vote
        vote = Counter(self.y_train[idx]).most_common(1)[0][0]
        return vote

class BaggingKNN:
    """
    Bagged ensemble of KNN classifiers.
    """
    def __init__(self, k=5, n_estimators=10, sample_fraction=1.0, random_state=None):
        self.k = k
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        """
        Fit an ensemble of KNNs on bootstrap samples of (X, y).
        """
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y, dtype=int)
        n_samples = X.shape[0]
        sample_size = int(self.sample_fraction * n_samples)
        self.models = []
        for _ in range(self.n_estimators):
            inds = np.random.choice(n_samples, size=sample_size, replace=True)
            knn = KNNClassifier(k=self.k)
            knn.fit(X[inds], y[inds])
            self.models.append(knn)

    def predict(self, X):
        """
        Predict labels by majority vote across ensemble.
        """
        X = np.array(X)
        # Collect predictions from each estimator
        all_preds = np.array([m.predict(X) for m in self.models])  # shape = (n_estimators, n_samples)
        # Transpose to iterate samples
        final = []
        for preds in all_preds.T:
            vote = Counter(preds).most_common(1)[0][0]
            final.append(vote)
        return np.array(final, dtype=int)
    

def load_and_clean(path):
    """
    1) Read CSV into a DataFrame.
    2) Drop rows with any NaNs.
    3) Keep only numeric columns (drops station names, regions, dates, etc.).
    4) Split into X (features) and y (label).
    """
    df = pd.read_csv(path)

    # Drop rows with missing values
    missing_mask = df.isnull().any(axis=1)
    total_missing = missing_mask.sum()
    total_rows = len(df)
    print(f"Dropped {total_missing} rows with missing values ({total_rows} -> {total_rows - total_missing}).")
    df = df.dropna(axis=0)

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # The last numeric column is assumed to be the label
    X = df_numeric.iloc[:, :-1].to_numpy()
    y = df_numeric.iloc[:, -1].astype(int).to_numpy()

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Stratified train/test split preserving class ratios.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

if __name__ == "__main__":
    # Example: load real data and evaluate
    DATA_PATH = '/Users/ahmedelghazi/Desktop/ML/Proj/dataset/all_conditions.csv'
    X, y = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data has been split")
    
    # Train a basic KNN
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Custom KNN (k=3) accuracy: {acc:.4f}")

    # Train a bagged KNN ensemble
    bag_knn = BaggingKNN(k=3, n_estimators=5, sample_fraction=1.0, random_state=42)
    bag_knn.fit(X_train, y_train)
    y_pred_bag = bag_knn.predict(X_test)
    acc_bag = np.mean(y_pred_bag == y_test)
    print(f"Bagged KNN (5 estimators) accuracy: {acc_bag:.4f}")