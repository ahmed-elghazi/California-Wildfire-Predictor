import numpy as np
from collections import Counter
from custom_knn import MyKNeighborsClassifier

class BaggedKNN:
    def __init__(self, n_estimators=10, k=3, max_samples_ratio=1.0):
        self.n_estimators = n_estimators
        self.k = k
        self.max_samples_ratio = max_samples_ratio
        self.estimators = []
        self.indices_list = [] # To store indices for each bag

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.max_samples_ratio)
        if sample_size == 0 and n_samples > 0: # Ensure at least one sample if possible
            sample_size = 1
        
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        return X[indices], y[indices], indices

    def fit(self, X, y):
        self.estimators = []
        self.indices_list = []
        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError("Cannot fit on empty data.")

        for _ in range(self.n_estimators):
            X_sample, y_sample, indices_sample = self._bootstrap_sample(X, y)
            
            if X_sample.shape[0] == 0: # Skip if bootstrap sample is empty
                continue

            knn = MyKNeighborsClassifier(self.k)
            knn.fit(X_sample, y_sample)
            self.estimators.append(knn)
            self.indices_list.append(indices_sample) # Store for potential OOB score later

    def predict(self, X_test):
        if not self.estimators:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Collect predictions from each estimator
        predictions_from_estimators = []
        for estimator in self.estimators:
            predictions_from_estimators.append(estimator.predict(X_test))
        
        # Transpose so that each row corresponds to a test sample, 
        # and each column to an estimator's prediction for that sample
        predictions_from_estimators = np.array(predictions_from_estimators).T 
        
        # Majority vote for each sample
        final_predictions = []
        for sample_predictions in predictions_from_estimators:
            most_common = Counter(sample_predictions).most_common(1)
            final_predictions.append(most_common[0][0])
            
        return np.array(final_predictions)
