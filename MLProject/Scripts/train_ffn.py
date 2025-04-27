import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier

# Load preprocessed data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

# Build and train FFN model
ffn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
ffn.fit(X_train, y_train)

# Save the model
joblib.dump(ffn, 'Models/saved_model_ffn.joblib')
