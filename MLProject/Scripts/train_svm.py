import numpy as np
import joblib
from sklearn.svm import SVC

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

svm = SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced', probability=True)
# The class_weight parameter is set to 'balanced' to handle class imbalance
# in the dataset. This will automatically adjust the weights inversely proportional to class frequencies
# The kernel is set to 'rbf' (Radial Basis Function) which is a common choice for SVMs
# The C parameter is set to 1.0, which is a common default value for SVMs
# The random_state is set to 42 for reproducibility
# The SVC class is used for classification tasks
# The fit method is used to train the SVM model on the training data
svm.fit(X_train, y_train)

joblib.dump(svm, 'Models/saved_model_svm.joblib')
