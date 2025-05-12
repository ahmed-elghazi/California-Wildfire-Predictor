import joblib
import os
from custom_adaboost import MyAdaBoost
import numpy as np


# Define paths relative to the script location
script_dir = os.path.dirname(__file__)
# Assuming 'data' and 'models' directories are peers to 'Scripts' directory
base_dir = os.path.join(script_dir, '..') 
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')

        # Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

        # Define file paths
        # Ensure these .npy files exist in the data_dir or adjust paths as needed
x_train_path = os.path.join(data_dir, 'X_train.npy')
y_train_path = os.path.join(data_dir, 'y_train.npy')
model_path = os.path.join(models_dir, 'my_custom_adaboost.joblib')

        # Load training data
try:
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
except FileNotFoundError:
        print(f"Error: Could not load training data from {data_dir}. Ensure X_train.npy and y_train.npy exist.")
        exit()

# Initialize the custom AdaBoost classifier
# Using random_state=42 for reproducibility, consistent with the original example's DecisionTreeClassifier
my_boosted_tree = MyAdaBoost(n_estimators=50, random_state=42)

        # Train the model
print("Training MyAdaBoost model...")
my_boosted_tree.fit(X_train, y_train)
print("Training complete.")

# Save the trained model
joblib.dump(my_boosted_tree, model_path)
print(f"MyAdaBoost model trained and saved to {model_path}")    

model_path = os.path.join(models_dir, 'my_custom_randomforest.joblib')
from custom_randomForest import MyRandomForest
# Initialize the custom Random Forest classifier
my_random_forest = MyRandomForest(n_estimators=100, random_state=42, max_depth=10)
# Train the model
print("Training MyRandomForest model...")
my_random_forest.fit(X_train, y_train)
print("Training complete.")
# Save the trained model
joblib.dump(my_random_forest, model_path)
print(f"MyRandomForest model trained and saved to {model_path}")

model_path = os.path.join(models_dir, 'my_custom_tree.joblib')
from custom_tree import MyDecisionTreeClassifier
# Initialize the custom Decision Tree classifier
my_decision_tree = MyDecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1)
# Train the model
print("Training MyDecisionTreeClassifier model...")
my_decision_tree.fit(X_train, y_train)
print("Training complete.")
# Save the trained model
joblib.dump(my_decision_tree, model_path)
print(f"MyDecisionTreeClassifier model trained and saved to {model_path}")

model_path = os.path.join(models_dir, 'my_custom_knn.joblib')
from custom_knn import MyKNeighborsClassifier
# Initialize the custom KNN classifier
my_knn = MyKNeighborsClassifier(n_neighbors=5)
# Train the model
print("Training MyKNN model...")
my_knn.fit(X_train, y_train)
print("Training complete.")
# Save the trained model
joblib.dump(my_knn, model_path)
print(f"MyKNN model trained and saved to {model_path}")

model_path = os.path.join(models_dir, 'my_custom_bagging_knn.joblib')
from custom_bagged_knn import BaggedKNN
# Initialize the custom Bagging KNN classifier
my_bagging_knn = BaggedKNN(n_estimators=10, k=5, max_samples_ratio=0.8)
# Train the model
print("Training MyBaggingKNN model...")
my_bagging_knn.fit(X_train, y_train)
print("Training complete.")
# Save the trained model
joblib.dump(my_bagging_knn, model_path)
print(f"MyBaggingKNN model trained and saved to {model_path}")