import numpy as np
import joblib
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import os

# Define paths relative to the script location or a known base directory
# Assuming the script is run from a location where 'data' and 'models' are subdirectories
data_dir = 'data'
models_dir = 'models'
model_save_path = os.path.join(models_dir, 'bagging_tree_model.joblib')

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# Load data
X_train_path = os.path.join(data_dir, 'X_train.npy')
y_train_path = os.path.join(data_dir, 'y_train.npy')

# Basic check if files exist before loading
if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
    print(f"Error: Training data not found in {data_dir}")
    exit() # Or handle the error appropriately

X_train = np.load(X_train_path)
y_train = np.load(y_train_path)

# Ensure y_train is 1-dimensional
if y_train.ndim > 1 and y_train.shape[1] == 1:
    y_train = y_train.ravel()

# Define the Bagging Regressor model
bagging_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,  # Number of base estimators (trees)
    random_state=42,
    n_jobs=-1,       # Use all available CPU cores
    oob_score=True   # Optional: Use out-of-bag samples
)

# Train the model
print("Training Bagging Tree Regressor...")
bagging_model.fit(X_train, y_train)
print("Training complete.")
if hasattr(bagging_model, 'oob_score_') and bagging_model.oob_score:
     print(f"OOB Score: {bagging_model.oob_score_:.4f}")


# Save the model
print(f"Saving model to {model_save_path}...")
joblib.dump(bagging_model, model_save_path)
print("Model saved successfully.")
