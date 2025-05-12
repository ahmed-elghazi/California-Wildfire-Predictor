import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

class MyAdaBoost:
    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state # For reproducibility of weak learners
        self.models = []
        self.alphas = []
        self.classes_ = None

    def fit(self, X, y):
        self.models = []
        self.alphas = []
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This AdaBoost implementation only supports binary classification.")

        # Transform y to -1, 1
        y_transformed = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))

        for i in range(self.n_estimators):
            # Set random_state for DecisionTreeClassifier if self.random_state is provided
            # Increment random_state for each estimator to ensure they are different but reproducible
            current_random_state = self.random_state + i if self.random_state is not None else None
            
            model = DecisionTreeClassifier(max_depth=1, random_state=current_random_state)
            
            # Fit weak learner using sample weights
            # DecisionTreeClassifier expects original class labels (0, 1)
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)
            
            # Transform predictions to -1, 1
            y_pred_transformed = np.where(y_pred == self.classes_[0], -1, 1)

            # Calculate weighted error
            # Incorrectly classified samples are those where y_transformed != y_pred_transformed
            misclassified = (y_pred_transformed != y_transformed)
            err = np.sum(sample_weights[misclassified]) / np.sum(sample_weights)

            # Add epsilon to prevent division by zero or log(0)
            epsilon = 1e-10
            if err == 0.0: # Perfect learner
                err = epsilon
            elif err >= 0.5: # Worse than random or random
                # Standard AdaBoost might stop here or assign alpha=0 or a very small alpha
                # For simplicity, we can skip this estimator or assign a small alpha
                # If err is 1.0, alpha becomes -inf.
                # We can break or re-initialize weights and continue (not standard)
                # For this implementation, if error is too high, we effectively give it low/zero weight
                # by capping error or handling alpha.
                # Let's cap error to prevent issues with alpha calculation if err is too high.
                # However, standard AdaBoost formula handles err > 0.5 by making alpha negative.
                # If err is exactly 0.5, alpha is 0.
                # If err is 1.0, alpha is -infinity. Add epsilon to avoid this.
                 if err == 1.0:
                    err = 1.0 - epsilon


            alpha = 0.5 * np.log((1.0 - err) / (err + epsilon)) # Added epsilon to err as well

            # Update sample weights
            sample_weights *= np.exp(-alpha * y_transformed * y_pred_transformed)
            sample_weights /= np.sum(sample_weights)  # Normalize

            self.models.append(model)
            self.alphas.append(alpha)
            
            # Early stopping if a learner is perfect and alpha is huge, or if error is too high
            if err <= epsilon and alpha > 0: # Effectively perfect learner
                 # If a learner is perfect, it might dominate. AdaBoost handles this.
                 pass # Continue building estimators
            if err >= 0.5 and alpha <=0: # If learner is no better than random
                # Could break, but standard AdaBoost continues.
                # The negative alpha will be used in prediction.
                pass


    def predict(self, X):
        n_samples = X.shape[0]
        # Initialize with zeros, representing the boundary
        agg_predictions = np.zeros(n_samples)

        for alpha, model in zip(self.alphas, self.models):
            y_pred_model = model.predict(X)
            # Transform predictions to -1, 1
            y_pred_transformed = np.where(y_pred_model == self.classes_[0], -1, 1)
            agg_predictions += alpha * y_pred_transformed
        
        # Final prediction is the sign of the aggregated sum
        # Map -1 to self.classes_[0] and 1 to self.classes_[1]
        final_predictions_transformed = np.sign(agg_predictions)
        
        # Handle cases where agg_predictions is exactly 0 (np.sign returns 0)
        # Assign to one class, e.g., self.classes_[0] or randomly.
        # For simplicity, assign to self.classes_[0] if 0.
        final_predictions = np.where(final_predictions_transformed == 1, self.classes_[1], self.classes_[0])
        
        return final_predictions

    def predict_proba(self, X):
        n_samples = X.shape[0]
        agg_predictions = np.zeros(n_samples)

        for alpha, model in zip(self.alphas, self.models):
            y_pred_model = model.predict(X)
            y_pred_transformed = np.where(y_pred_model == self.classes_[0], -1, 1)
            agg_predictions += alpha * y_pred_transformed
        
        # Convert aggregated scores to probabilities
        # P(Y=1|X) = 1 / (1 + exp(-2 * sum(alpha_m * h_m(X))))
        # This is probability for the class mapped to +1 (self.classes_[1])
        prob_class_1 = 1.0 / (1.0 + np.exp(-2.0 * agg_predictions))
        prob_class_0 = 1.0 - prob_class_1

        # Output shape (n_samples, 2)
        # Column 0: probability of self.classes_[0]
        # Column 1: probability of self.classes_[1]
        probabilities = np.vstack((prob_class_0, prob_class_1)).T
        return probabilities