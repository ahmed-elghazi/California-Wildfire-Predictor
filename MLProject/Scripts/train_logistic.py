import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

logreg = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
# The class_weight parameter is set to 'balanced' to handle class imbalance
# in the dataset. This will automatically adjust the weights inversely proportional to class frequencies
logreg.fit(X_train, y_train)

joblib.dump(logreg, 'Models/saved_model_logistic.joblib')
