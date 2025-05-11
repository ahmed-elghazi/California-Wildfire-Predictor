import numpy as np
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os

# Define paths relative to the script location (optional, but good practice)
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data')
models_dir = os.path.join(script_dir, '..', 'models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Define file paths
x_train_path = os.path.join(data_dir, 'X_train.npy')
y_train_path = os.path.join(data_dir, 'y_train.npy')
model_path = os.path.join(models_dir, 'adaboost_tree.joblib')

# Load training data
X_train = np.load(x_train_path)
y_train = np.load(y_train_path)

# Initialize the base estimator and the AdaBoost classifier
base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
boosted_tree = AdaBoostClassifier(estimator=base_tree, n_estimators=50, random_state=42)

# Train the model
boosted_tree.fit(X_train, y_train)

# Save the trained model
joblib.dump(boosted_tree, model_path)

print(f"AdaBoost Tree model trained and saved to {model_path}")
