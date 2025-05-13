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
    def predict_proba(self, X_test):
        if not self.estimators:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Collect probability predictions from each estimator
        proba_from_estimators = []
        for estimator in self.estimators:
            # Assuming MyKNeighborsClassifier has a predict_proba method
            # that returns probabilities for each class
            proba_from_estimators.append(estimator.predict_proba(X_test))
        
        # Average the probabilities across all estimators
        # proba_from_estimators will be a list of arrays, 
        # each array of shape (n_samples, n_classes)
        # We need to sum them up and divide by the number of estimators
        
        if not proba_from_estimators: # Should not happen if self.estimators is not empty and fit worked
            # Handle case where no estimators made valid predictions (e.g., all bootstrap samples were empty)
            # This depends on how MyKNeighborsClassifier.predict_proba handles empty fit or X_test
            # For now, let's assume it returns an array of zeros or similar,
            # or we can raise an error.
            # A simple approach: if X_test has samples, return uniform probability if no classes known,
            # or handle based on MyKNeighborsClassifier's behavior.
            # For simplicity, let's assume MyKNeighborsClassifier.predict_proba returns valid shapes.
            # If all estimators failed to produce probas (e.g. if all bootstrap samples were empty and estimators list is empty)
            # this part of the code won't be reached due to the initial check.
            # If some estimators are there but their predict_proba returns empty or incompatible shapes,
            # np.mean will fail.
            # We assume MyKNeighborsClassifier.predict_proba returns (n_samples, n_classes)
            # and all estimators agree on n_classes.
            pass

        # Stack probabilities into a 3D array (n_estimators, n_samples, n_classes)
        stacked_probas = np.stack(proba_from_estimators)
        
        # Average across estimators (axis 0)
        mean_probas = np.mean(stacked_probas, axis=0)
            
        return mean_probas
