import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import joblib




# Load validation data
X_val = np.load('data/X_val.npy')  
y_val = np.load('data/y_val.npy')

# List of models and their file paths
model_files = {
    # "Feedforward Neural Net (FFN)": "Models/saved_model_ffn.joblib",
    # "Support Vector Machine (SVM)": "Models/saved_model_svm.joblib",
    # "Logistic Regression": "Models/saved_model_logistic.joblib",
    # "Perceptron": "Models/saved_model_perceptron.joblib",
    # "Boosted Logistic Regression": "Models/saved_model_boost_logistic.joblib",
    # "Boosted Perceptron": "Models/saved_model_boost_perceptron.joblib",
    "adaboost": "models/Adaboost_model.joblib",
    "bagged trees": "models/bagging_tree_model.joblib",
    "normal tree": "models/manual_decision_tree_model.joblib",
    "KNN": "models/manual_knn_model.joblib",
    "bagged knn": "models/bagging_knn_model.joblib",

}


for model_name, model_path in model_files.items():
    print(f"Evaluating: {model_name}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_val)
    
    print(classification_report(y_val, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"plots/conf_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

    # --- 3. Save ROC curve (if model supports predict_proba) ---
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = roc_auc_score(y_val, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(f"plots/roc_curve_{model_name.replace(' ', '_')}.png")
        plt.close()