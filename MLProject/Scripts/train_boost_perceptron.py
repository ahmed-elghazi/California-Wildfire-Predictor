import numpy as np
import joblib
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

base_perceptron = Perceptron(max_iter=200, random_state=42)
boosted_perceptron = AdaBoostClassifier(estimator=base_perceptron, n_estimators=50, random_state=42)
boosted_perceptron.fit(X_train, y_train)

joblib.dump(boosted_perceptron, 'Models/saved_model_boost_perceptron.joblib')
