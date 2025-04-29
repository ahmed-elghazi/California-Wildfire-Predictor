import numpy as np
import os
import joblib
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.utils import resample # For bootstrapping

def load_data(data_dir="data"):
    """
    Loads training and testing data from .npy files in the specified directory.

    Args:
        data_dir (str): The directory containing the data files
                        (X_train.npy, y_train.npy, X_test.npy, y_test.npy).

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test).
               Returns None for all if any file is missing.
    """
    try:
        X_train_path = os.path.join(data_dir, "X_train.npy")
        y_train_path = os.path.join(data_dir, "y_train.npy")
        X_test_path = os.path.join(data_dir, "X_test.npy")
        y_test_path = os.path.join(data_dir, "y_test.npy")

        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        print(f"Data loaded successfully from {data_dir}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure all .npy files exist in '{data_dir}'.")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None

def train_bagging_knn(X_train, y_train, n_estimators=10, n_neighbors=5, max_samples_ratio=1.0):
    """
    Trains a Bagging KNN ensemble model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_estimators (int): The number of KNN models (base estimators) to train.
        n_neighbors (int): The number of neighbors (k) for each KNN model.
        max_samples_ratio (float): The ratio of samples to draw for each bootstrap sample.

    Returns:
        list: A list containing the trained KNeighborsClassifier models.
    """
    n_samples = X_train.shape[0]
    n_bootstrap_samples = int(max_samples_ratio * n_samples)
    estimators = [] # List to hold each trained KNN model

    print(f"Training {n_estimators} KNN models with k={n_neighbors}...")
    for i in range(n_estimators):
        # Create a bootstrap sample (sampling with replacement)
        X_bootstrap, y_bootstrap = resample(
            X_train, y_train,
            replace=True,
            n_samples=n_bootstrap_samples,
            random_state=i # Ensure reproducibility for each estimator
        )

        # Initialize and train a KNN classifier on the bootstrap sample
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_bootstrap, y_bootstrap)

        # Add the trained model to our list of estimators
        estimators.append(knn)
        print(f"  Trained estimator {i+1}/{n_estimators}")

    print("Bagging KNN training complete.")
    return estimators

def predict_bagging_knn(estimators, X_test):
    """
    Makes predictions using the trained Bagging KNN ensemble via majority voting.

    Args:
        estimators (list): A list of trained KNeighborsClassifier models.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: The aggregated predictions for the test set.
    """
    # Collect predictions from each individual estimator
    all_predictions = []
    print("Making predictions with individual estimators...")
    for i, knn in enumerate(estimators):
        predictions = knn.predict(X_test)
        all_predictions.append(predictions)
        # print(f"  Predictions made by estimator {i+1}/{len(estimators)}")

    # Transpose the predictions array so rows correspond to samples
    # and columns correspond to predictions from different estimators
    # Shape becomes (n_samples, n_estimators)
    predictions_array = np.array(all_predictions).T

    # Perform majority voting for each sample
    final_predictions = []
    print("Aggregating predictions using majority vote...")
    for sample_predictions in predictions_array:
        # Count occurrences of each predicted class for the current sample
        vote_counts = Counter(sample_predictions)
        # Find the class with the highest count (majority vote)
        majority_vote = vote_counts.most_common(1)[0][0]
        final_predictions.append(majority_vote)

    print("Aggregation complete.")
    return np.array(final_predictions)

def save_model(model, filename="bagging_knn_model.joblib"):
    """
    Saves the trained model (list of estimators) to a file using joblib.

    Args:
        model (list): The list of trained estimators.
        filename (str): The path to save the model file.
    """
    try:
        joblib.dump(model, filename)
        print(f"Model saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    """
    Main function to drive the Bagging KNN process.
    """
    # --- Configuration ---
    DATA_DIR = "data"  # Directory containing .npy files
    N_ESTIMATORS = 20  # Number of KNN models in the ensemble
    N_NEIGHBORS = 7    # K value for each KNN
    MAX_SAMPLES_RATIO = 0.8 # Use 80% of the data for each bootstrap sample
    MODEL_FILENAME = "bagging_knn_model.joblib" # File to save the trained model

    # --- Load Data ---
    X_train, y_train, X_test, y_test = load_data(DATA_DIR)
    if X_train is None:
        print("Exiting due to data loading errors.")
        return # Exit if data loading failed

    # --- Train Model ---
    bagging_knn_estimators = train_bagging_knn(
        X_train, y_train,
        n_estimators=N_ESTIMATORS,
        n_neighbors=N_NEIGHBORS,
        max_samples_ratio=MAX_SAMPLES_RATIO
    )

    # --- Make Predictions ---
    y_pred = predict_bagging_knn(bagging_knn_estimators, X_test)

    # --- Evaluate Model ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBagging KNN Test Accuracy: {accuracy:.4f}")

    # --- Save Model ---
    save_model(bagging_knn_estimators, MODEL_FILENAME)

if __name__ == "__main__":
    main()