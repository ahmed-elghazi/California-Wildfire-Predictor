import numpy as np
import joblib
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define relative paths for data and models
DATA_DIR = 'data'
MODELS_DIR = 'Models' # Match the example's directory name

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load training data
try:
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
except FileNotFoundError:
    print(f"Error: Training data files not found in {DATA_DIR}. Please ensure X_train.npy and y_train.npy exist.")
    exit() # Exit if data is missing

print(f"Loaded training data: X_train={X_train.shape}, y_train={y_train.shape}")

# Initialize KNN classifier
# Consider tuning n_neighbors based on your data/validation
knn = KNeighborsClassifier(n_neighbors=5)

# Initialize Bagging classifier with KNN
# Consider tuning n_estimators, max_samples, max_features based on your data/validation
bagging_knn = BaggingClassifier(
    estimator=knn,
    n_estimators=10,      # Number of base estimators (KNN models)
    max_samples=1.0,      # Fraction of samples for each base estimator (1.0 uses all)
    max_features=1.0,     # Fraction of features for each base estimator (1.0 uses all)
    random_state=42,
    n_jobs=-1             # Use all available CPU cores
)

# Train the Bagging KNN model
print("Training Bagging KNN model...")
bagging_knn.fit(X_train, y_train)
print("Training complete.")

# Define model save path
model_filename = 'saved_model_bagging_knn.joblib' # Match the example's naming convention
model_path = os.path.join(MODELS_DIR, model_filename)

# Save the trained model
print(f"Saving trained model to {model_path}...")
joblib.dump(bagging_knn, model_path)
print("Model saved successfully.")
