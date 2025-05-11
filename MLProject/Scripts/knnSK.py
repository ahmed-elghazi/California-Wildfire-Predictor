import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Define relative paths
DATA_DIR = 'data'
MODELS_DIR = 'models' # Changed from 'Models' to match original script's convention
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.npy')
MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.joblib')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load training data
try:
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    print(f"Loaded data from {X_TRAIN_PATH} and {Y_TRAIN_PATH}")
except FileNotFoundError:
    print(f"Error: Training data not found at {X_TRAIN_PATH} or {Y_TRAIN_PATH}")
    print("Please ensure 'X_train.npy' and 'y_train.npy' exist in the 'data' directory.")
    exit() # Exit if data is not found
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# Initialize KNN classifier
n_neighbors = 5 # Example value, adjust as needed
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
print(f"Initialized KNN classifier with k={n_neighbors}")

# Train the classifier
print("Training KNN classifier...")
try:
    knn.fit(X_train, y_train)
    print("KNN classifier trained successfully.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit()

# Save the trained model
try:
    joblib.dump(knn, MODEL_PATH)
    print(f"Trained KNN model saved to {MODEL_PATH}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

