import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Define relative paths assuming the script is run from the project root or similar
# Adjust these paths if necessary based on your execution context
data_dir = 'data'
models_dir = 'models'

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Define file paths
X_train_path = os.path.join(data_dir, 'X_train.npy')
y_train_path = os.path.join(data_dir, 'y_train.npy')
model_path = os.path.join(models_dir, 'decision_tree_model.joblib')

# Load training data
X_train = np.load(X_train_path)
y_train = np.load(y_train_path)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(dt_classifier, model_path)

print(f"Decision Tree model trained and saved to {model_path}")
