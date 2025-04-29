import numpy as np
import os
from collections import Counter
import math
import joblib # Added import

# ... (Keep existing functions: euclidean_distance, get_neighbors, predict_classification, calculate_accuracy, load_data) ...
def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points (vectors).

    Args:
        point1 (np.ndarray): The first point.
        point2 (np.ndarray): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    # Ensure points are numpy arrays for vectorized operations
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    # Calculate the sum of squared differences between coordinates
    sum_sq_diff = np.sum((point1 - point2)**2)
    # Return the square root of the sum
    return math.sqrt(sum_sq_diff)

def get_neighbors(X_train, y_train, test_instance, k):
    """
    Finds the k nearest neighbors for a given test instance.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        test_instance (np.ndarray): The data point for which to find neighbors.
        k (int): The number of nearest neighbors to find.

    Returns:
        list: A list of labels of the k nearest neighbors.
    """
    distances = []
    # Calculate distance from the test instance to all training points
    for i in range(len(X_train)):
        dist = euclidean_distance(test_instance, X_train[i])
        # Store the distance and the corresponding label
        distances.append((dist, y_train[i]))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])

    # Get the labels of the top k neighbors
    neighbors = [distances[i][1] for i in range(k)]
    return neighbors

def predict_classification(X_train, y_train, test_instance, k):
    """
    Predicts the class label for a test instance using KNN.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        test_instance (np.ndarray): The data point to classify.
        k (int): The number of nearest neighbors to consider.

    Returns:
        The predicted class label.
    """
    # Get the labels of the k nearest neighbors
    neighbors = get_neighbors(X_train, y_train, test_instance, k)
    # Count the occurrences of each label among the neighbors
    output_values = [row for row in neighbors]
    # Find the most common class label
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

def calculate_accuracy(y_true, y_pred):
    """
    Calculates the accuracy of predictions.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        float: The accuracy score.
    """
    correct_count = 0
    # Compare true labels with predicted labels
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1
    # Calculate accuracy as the fraction of correct predictions
    return correct_count / float(len(y_true))

def load_data(data_dir):
    """
    Loads training and testing data from .npy files in the specified directory.

    Args:
        data_dir (str): The directory containing the data files.

    Returns:
        tuple: A tuple containing X_train, y_train, X_test, y_test.
               Returns None if any file is missing or loading fails.
    """
    files_to_load = {
        'X_train': 'X_train.npy',
        'y_train': 'y_train.npy',
        'X_test': 'X_test.npy',
        'y_test': 'y_test.npy'
    }
    data = {}

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return None

    # Load each required file
    for key, filename in files_to_load.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            print(f"Error: Data file '{file_path}' not found.")
            return None
        try:
            data[key] = np.load(file_path)
            print(f"Loaded '{file_path}' successfully.")
        except Exception as e:
            print(f"Error loading data file '{file_path}': {e}")
            return None

    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


# Added save_model function
def save_model(X_train, y_train, k, filename="manual_knn_model.joblib"):
    """
    Saves the components needed for the manual KNN model (training data and k)
    to a file using joblib. In this implementation, the 'model' consists
    of the training data and the value of k, as prediction requires them directly.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        k (int): The number of neighbors used.
        filename (str): The path to save the model file.
    """
    # The 'model' for this KNN implementation is the training data and k
    model_components = {
        'X_train': X_train,
        'y_train': y_train,
        'k': k
    }
    try:
        # Ensure the directory exists if filename includes a path
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        joblib.dump(model_components, filename)
        print(f"Model components saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving model components: {e}")


def main():
    """
    Main function to load data, run manual KNN, evaluate accuracy, and save model components.
    """
    # Define the directory containing the data
    data_dir = 'data'
    # Set the number of neighbors
    k = 5
    # Define the filename for the saved model components
    model_filename = "models/manual_knn_model.joblib" # Changed filename

    # Load the data
    loaded_data = load_data(data_dir)
    if loaded_data is None:
        return # Exit if data loading failed
    X_train, y_train, X_test, y_test = loaded_data

    # --- Save the model components after loading ---
    # For this KNN, saving the training data and k allows reusing them later.
    print(f"\nSaving model components (X_train, y_train, k={k})...")
    save_model(X_train, y_train, k, model_filename)
    # ---------------------------------------------------------

    # Make predictions on the test set using the manual KNN implementation
    predictions = []
    print(f"\nMaking predictions using manual KNN (k={k})...")
    for i in range(len(X_test)):
        # Pass the loaded X_train, y_train, and k to the prediction function
        prediction = predict_classification(X_train, y_train, X_test[i], k)
        predictions.append(prediction)
        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(X_test):
             print(f"Predicted {i+1}/{len(X_test)} instances.")

    # Calculate the accuracy of the predictions
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"\nManual KNN model accuracy on the test set: {accuracy:.4f}")

if __name__ == "__main__":
    main()