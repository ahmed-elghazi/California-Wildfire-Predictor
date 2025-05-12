import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self._classes = None

    def fit(self, X, y):
        """
        Fit the K-Nearest Neighbors model from the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self._classes = np.unique(self.y_train)

    def _euclidean_distance(self, x1, x2):
        """Compute the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2)**2))

    # def _predict_single(self, x_test):
    #     """
    #     Predict the class label for a single test sample.
    #     """
    #     # Calculate distances from x_test to all training samples
    #     distances = [self._euclidean_distance(x_test, x_train) for x_train in self.X_train]
        
    #     # Get indices of k-nearest neighbors
    #     k_indices = np.argsort(distances)[:self.n_neighbors]
        
    #     # Get labels of k-nearest neighbors
    #     k_nearest_labels = [self.y_train[i] for i in k_indices]
        
    #     # Perform majority vote
    #     most_common = Counter(k_nearest_labels).most_common(1)
    #     return most_common[0][0]

    # def predict(self, X):
    #     """
    #     Predict the class labels for the provided data.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_queries, n_features)
    #         Test samples.

    #     Returns
    #     -------
    #     y_pred : ndarray of shape (n_queries,)
    #         Class labels for each data sample.
    #     """
    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError("Model has not been trained yet. Call fit() first.")
        
    #     X_test = np.array(X)
    #     if X_test.ndim == 1: # Single sample
    #         X_test = X_test.reshape(1, -1)
            
    #     y_pred = [self._predict_single(x) for x in X_test]
    #     return np.array(y_pred)

    def _predict_single_vectorized(self, x_test):
        """
        Optimized version of _predict_single using NumPy vectorization.
        """
        distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        X_test = np.array(X)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # Vectorized distance computation
        distances = cdist(X_test, self.X_train, metric='euclidean')  # shape: (n_test, n_train)
        k_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Get labels and perform majority vote
        y_pred = []
        for neighbor_indices in k_indices:
            neighbor_labels = self.y_train[neighbor_indices]
            majority_class = Counter(neighbor_labels).most_common(1)[0][0]
            y_pred.append(majority_class)

        return np.array(y_pred)



    def _predict_proba_single(self, x_test):
        """
        Predict class probabilities for a single test sample.
        """
        # Calculate distances from x_test to all training samples
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self.X_train]
        
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get labels of k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Count occurrences of each class among neighbors
        counts = Counter(k_nearest_labels)
        
        # Calculate probabilities for each class
        probabilities = np.zeros(len(self._classes))
        for i, cls in enumerate(self._classes):
            probabilities[i] = counts[cls] / self.n_neighbors
            
        return probabilities

    def predict_proba(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        X_test = np.array(X)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        distances = cdist(X_test, self.X_train, metric='euclidean')
        k_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        y_proba = np.zeros((len(X_test), len(self._classes)))

        for i, neighbor_indices in enumerate(k_indices):
            neighbor_labels = self.y_train[neighbor_indices]
            counts = Counter(neighbor_labels)
            for j, cls in enumerate(self._classes):
                y_proba[i, j] = counts[cls] / self.n_neighbors

        return y_proba

