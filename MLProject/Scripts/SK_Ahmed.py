#FOR COMPARISON PURPOSES ONLY
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Define relative paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.npy')

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
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- Decision Tree ---
MODEL_PATH_DT = os.path.join(MODELS_DIR, 'SKtree_model.joblib')
dt_classifier = DecisionTreeClassifier(random_state=42)
print("Initialized Decision Tree classifier.")
print("Training Decision Tree classifier...")
try:
    dt_classifier.fit(X_train, y_train)
    print("Decision Tree classifier trained successfully.")
    joblib.dump(dt_classifier, MODEL_PATH_DT)
    print(f"Trained Decision Tree model saved to {MODEL_PATH_DT}")
except Exception as e:
    print(f"An error occurred with Decision Tree: {e}")

# --- Random Forest ---
MODEL_PATH_RF = os.path.join(MODELS_DIR, 'SKforest_model.joblib')
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
print("Initialized Random Forest classifier.")
print("Training Random Forest classifier...")
try:
    rf_classifier.fit(X_train, y_train)
    print("Random Forest classifier trained successfully.")
    joblib.dump(rf_classifier, MODEL_PATH_RF)
    print(f"Trained Random Forest model saved to {MODEL_PATH_RF}")
except Exception as e:
    print(f"An error occurred with Random Forest: {e}")

# --- AdaBoost Trees ---
MODEL_PATH_ADABOOST = os.path.join(MODELS_DIR, 'SKadaboost.joblib')
# AdaBoost with DecisionTree base estimator
adaboost_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    random_state=42
)
print("Initialized AdaBoost classifier.")
print("Training AdaBoost classifier...")
try:
    adaboost_classifier.fit(X_train, y_train)
    print("AdaBoost classifier trained successfully.")
    joblib.dump(adaboost_classifier, MODEL_PATH_ADABOOST)
    print(f"Trained AdaBoost model saved to {MODEL_PATH_ADABOOST}")
except Exception as e:
    print(f"An error occurred with AdaBoost: {e}")

# --- KNN ---
MODEL_PATH_KNN = os.path.join(MODELS_DIR, 'SKknn.joblib')
n_neighbors_knn = 5
knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
print(f"Initialized KNN classifier with k={n_neighbors_knn}.")
print("Training KNN classifier...")
try:
    knn_classifier.fit(X_train, y_train)
    print("KNN classifier trained successfully.")
    joblib.dump(knn_classifier, MODEL_PATH_KNN)
    print(f"Trained KNN model saved to {MODEL_PATH_KNN}")
except Exception as e:
    print(f"An error occurred with KNN: {e}")

# --- Bagged KNN ---
MODEL_PATH_BAGGED_KNN = os.path.join(MODELS_DIR, 'SKnnBag.joblib')
n_neighbors_bagged_knn = 5
# BaggingClassifier with KNeighborsClassifier as base estimator
bagged_knn_classifier = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=n_neighbors_bagged_knn),
    n_estimators=10,  # Number of base estimators (KNN models)
    random_state=42
)
print(f"Initialized Bagged KNN classifier with base k={n_neighbors_bagged_knn}.")
print("Training Bagged KNN classifier...")
try:
    bagged_knn_classifier.fit(X_train, y_train)
    print("Bagged KNN classifier trained successfully.")
    joblib.dump(bagged_knn_classifier, MODEL_PATH_BAGGED_KNN)
    print(f"Trained Bagged KNN model saved to {MODEL_PATH_BAGGED_KNN}")
except Exception as e:
    print(f"An error occurred with Bagged KNN: {e}")

print("\nAll models processed.")