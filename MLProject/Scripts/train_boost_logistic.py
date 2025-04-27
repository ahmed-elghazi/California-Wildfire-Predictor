import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

base_logreg = LogisticRegression(max_iter=200, random_state=42)
boosted_logreg = AdaBoostClassifier(estimator=base_logreg, n_estimators=50, random_state=42)
boosted_logreg.fit(X_train, y_train)

joblib.dump(boosted_logreg, 'Models/saved_model_boost_logistic.joblib')
