# California Wildfire Predictor: Advanced Data Processing & Model Evaluation

This project implements and evaluates multiple machine learning algorithms for binary classification using weather and environmental data. The project focuses on predicting a target variable using various meteorological features from CIMIS (California Irrigation Management Information System) stations.

## 🔬 Data Processing Methodology

This project employs a comprehensive data preprocessing pipeline that addresses common challenges in real-world machine learning:

### 1. Data Cleansing & Quality Control
- **Missing Value Handling**: Automatic removal of incomplete records using `df.dropna()`
- **Feature Selection**: Strategic removal of non-predictive columns (Station Name, CIMIS Region, Date)
- **Data Type Consistency**: Proper handling of numerical vs categorical features

### 2. Feature Engineering Pipeline
The `DataSplitter.py` implements a robust preprocessing pipeline using scikit-learn's `ColumnTransformer`:
- **Numerical Features**: StandardScaler normalization for 15 meteorological variables
- **Categorical Features**: One-hot encoding with unknown value handling
- **Pipeline Persistence**: Saves preprocessing steps for consistent future predictions

### 3. Stratified Data Splitting
Advanced splitting methodology ensures representative samples:
- **Three-way Split**: 60% training, 20% validation, 20% testing
- **Stratified Sampling**: Maintains original class distribution across all splits
- **Random State Control**: Reproducible splits with `random_state=42`

### 4. Class Imbalance Handling
Addresses skewed target distributions through undersampling:
- **RandomUnderSampler**: Applied only to training data to prevent data leakage
- **Class Balance Verification**: Automated distribution reporting
- **Validation Integrity**: Validation and test sets remain unchanged

### 5. Main.py Execution Pipeline
The main script orchestrates the entire preprocessing workflow:

```python
# 1. Data Loading & Initial Setup
df = pd.read_csv("all_conditions.csv")  # Load 128K+ weather records

# 2. Feature Configuration
drop_cols = ['Stn Name', 'CIMIS Region', 'Date']  # Remove non-predictive features
numeric_cols = [15 meteorological features]        # Define numerical variables

# 3. Preprocessing & Splitting
splitter = DataPreprocessorAndSplitter(...)        # Initialize custom preprocessor
splits = splitter.stratified_split()               # Create stratified splits

# 4. Class Balancing (Training Only)
undersample = RandomUnderSampler(random_state=42)  # Initialize undersampler
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# 5. Data Persistence
# Save processed data as .npy files for efficient loading
# Save preprocessing pipeline for future use
```

This methodology ensures:
- **Data Quality**: Clean, consistent input data
- **Reproducibility**: Fixed random seeds and saved pipelines
- **Preventing Data Leakage**: Separate preprocessing for train/validation/test
- **Class Balance**: Addressing skewed distributions without losing validation integrity
- **Scalability**: Efficient data storage and loading for large datasets

## 📁 Project Structure

```
MLProject/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── all_conditions.csv          # Main dataset (128K+ records)
├── data/                       # Preprocessed data splits
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
├── models/                     # Trained model files
│   ├── Adaboost_model.joblib
│   ├── bagging_knn_model.joblib
│   ├── bagging_tree_model.joblib
│   ├── manual_decision_tree_model.joblib
│   ├── manual_knn_model.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── saved_model_boost_logistic.joblib
│   ├── saved_model_boost_perceptron.joblib
│   ├── saved_model_ffn.joblib
│   ├── saved_model_logistic.joblib
│   ├── saved_model_perceptron.joblib
│   └── saved_model_svm.joblib
├── plots/                      # Visualization outputs
│   ├── conf_matrix_*.png       # Confusion matrices
│   ├── roc_curve_*.png        # ROC curves
│   └── Plots.zip              # Archived plots
└── Scripts/                   # Source code
    ├── Main.py                # Main data preprocessing script
    ├── DataSplitter.py        # Data preprocessing and splitting utility
    ├── evaluate_models.py     # Model evaluation and visualization
    ├── train_*.py            # Individual model training scripts
    ├── bagging_tree.py       # Bagging with decision trees
    ├── bagknn.py             # Bagging with K-NN
    ├── boost_tree.py         # Boosting implementation
    ├── dtree2.py             # Decision tree implementation
    └── KNN.py                # K-Nearest Neighbors implementation
```

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline that includes:

- **Data Preprocessing**: Feature scaling, handling missing values, and class balancing
- **Multiple Algorithms**: Implementation of various ML algorithms from scratch and using scikit-learn
- **Ensemble Methods**: Bagging and boosting techniques
- **Model Evaluation**: Performance metrics, confusion matrices, and ROC curves
- **Visualization**: Comprehensive plots for model comparison

## 📊 Dataset

The dataset (`all_conditions.csv`) contains **128,127 records** of weather and environmental data from CIMIS stations with the following features:

### Features:
- **Station Info**: Station ID, Name, CIMIS Region
- **Weather Data**: 
  - Temperature: Max/Min/Avg Air Temp, Avg Soil Temp
  - Humidity: Max/Min/Avg Relative Humidity, Dew Point
  - Precipitation and Solar Radiation
  - Wind: Speed and Run measurements
  - Evapotranspiration (ETo)
  - Vapor Pressure

### Target Variable:
- Binary classification target (0/1)

## 🤖 Implemented Algorithms

### Traditional ML Models:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Perceptron**
4. **K-Nearest Neighbors (K-NN)**
5. **Decision Trees**
6. **Feedforward Neural Network (FFN)**

### Ensemble Methods:
1. **AdaBoost** - Adaptive boosting
2. **Bagging with Decision Trees**
3. **Bagging with K-NN**
4. **Boosted Logistic Regression**
5. **Boosted Perceptron**

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLProject
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies:
- `scikit-learn==1.6.1` - Machine learning library
- `pandas==2.2.3` - Data manipulation
- `numpy==2.0.2` - Numerical computing
- `matplotlib==3.9.4` - Plotting
- `seaborn==0.13.2` - Statistical visualization
- `imbalanced-learn==0.12.4` - Handling imbalanced datasets
- `keras==3.9.2` - Deep learning for FFN

## 🏃‍♂️ Running the Project

### 1. Data Preprocessing and Splitting
```bash
cd Scripts
python Main.py
```
This script:
- Loads and cleans the dataset
- Applies feature preprocessing (scaling, encoding)
- Performs stratified train/validation/test splits
- Applies random undersampling to handle class imbalance
- Saves processed data to the `data/` directory

### 2. Train Individual Models
Run the individual training scripts:
```bash
python train_logistic.py      # Logistic Regression
python train_svm.py           # Support Vector Machine
python train_perceptron.py    # Perceptron
python train_ffn.py           # Feedforward Neural Network
python KNN.py                 # K-Nearest Neighbors
python dtree2.py              # Decision Trees
python bagging_tree.py        # Bagging with Trees
python bagknn.py              # Bagging with K-NN
python boost_tree.py          # AdaBoost
```

### 3. Evaluate All Models
```bash
python evaluate_models.py
```
This generates:
- Classification reports for all models
- Confusion matrices (saved as PNG files)
- ROC curves for models supporting probability prediction

## 📈 Model Evaluation

The project evaluates models using:

- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: Visual representation of prediction accuracy
- **ROC Curve & AUC**: For models with probability prediction capability

All visualizations are automatically saved to the `plots/` directory.

## 🛠️ Key Features

### Data Preprocessing Pipeline
- **Missing Value Handling**: Automatic removal of rows with missing data
- **Feature Scaling**: StandardScaler for numerical features
- **Class Balancing**: RandomUnderSampler to handle imbalanced classes
- **Stratified Splitting**: Maintains class distribution across splits

### Modular Design
- Separate training scripts for each algorithm
- Reusable `DataPreprocessorAndSplitter` class
- Consistent model saving/loading with joblib

### Comprehensive Evaluation
- Automated evaluation across all trained models
- Visual comparison through confusion matrices and ROC curves
- Performance metrics for model selection

## 📝 Results

Model performance results are displayed in the console when running `evaluate_models.py`. The best performing models are saved in the `models/` directory for future use.

## 🔮 Future Improvements

### 🎛️ Hyperparameter Optimization
- **Grid Search/Random Search**: Implement automated hyperparameter tuning for all models
- **Bayesian Optimization**: Use advanced optimization techniques (Optuna, Hyperopt) for efficient search
- **Cross-Validation**: Replace single validation split with k-fold CV for more robust evaluation
- **Early Stopping**: Implement early stopping for iterative algorithms to prevent overfitting

### � Advanced Algorithms & Architectures
- **Deep Learning**: 
  - Implement more sophisticated neural networks (CNN, RNN, Transformer)
  - Add regularization techniques (Dropout, Batch Normalization)
  - Experiment with different activation functions and optimizers
- **Modern Ensemble Methods**:
  - XGBoost and LightGBM implementations
  - Stacking and blending techniques
  - Voting classifiers with different algorithm combinations
- **Feature Engineering**:
  - Automated feature selection (RFE, LASSO, Mutual Information)
  - Polynomial features and interaction terms
  - Time-based features from date columns

### 📊 Enhanced Data Processing
- **Advanced Sampling Techniques**:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Combined over/under-sampling strategies
- **Feature Scaling Alternatives**:
  - RobustScaler for outlier handling
  - QuantileTransformer for non-normal distributions
  - PowerTransformer for normalization
- **Missing Value Handling**:
  - Multiple imputation strategies
  - Model-based imputation (KNN, Iterative)
  - Domain-specific imputation for weather data

### 🔍 Model Interpretability & Analysis
- **Explainable AI**:
  - SHAP (SHapley Additive exPlanations) values
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance analysis across all models
- **Model Analysis**:
  - Learning curves for bias-variance analysis
  - Validation curves for hyperparameter sensitivity
  - Error analysis and failure case identification

### 🚀 Performance & Scalability
- **Computational Optimization**:
  - Parallel processing for model training
  - GPU acceleration for deep learning models
  - Memory-efficient data loading for large datasets
- **Model Serving**:
  - REST API for model predictions
  - Model versioning and experiment tracking (MLflow)
  - Containerization with Docker for deployment

### 📈 Advanced Evaluation Metrics
- **Beyond Accuracy**:
  - Precision-Recall curves for imbalanced datasets
  - Matthews Correlation Coefficient (MCC)
  - Cost-sensitive evaluation metrics
- **Statistical Testing**:
  - McNemar's test for model comparison
  - Bootstrap confidence intervals
  - Statistical significance testing

### 🔄 MLOps & Production Readiness
- **Automated Pipeline**:
  - CI/CD integration for model training
  - Automated data quality checks
  - Model performance monitoring
- **Data Drift Detection**:
  - Statistical tests for feature drift
  - Model performance degradation alerts
  - Automatic retraining triggers

### 🌐 Data Enhancement
- **External Data Sources**:
  - Integration with additional weather APIs
  - Satellite imagery for geographical features
  - Historical climate patterns
- **Temporal Features**:
  - Seasonal decomposition
  - Lag features for time series patterns
  - Rolling window statistics

### 📱 User Interface & Visualization
- **Interactive Dashboard**:
  - Streamlit/Plotly Dash for model comparison
  - Real-time prediction interface
  - Interactive feature importance plots
- **Advanced Visualizations**:
  - t-SNE/UMAP for dimensionality reduction visualization
  - Partial dependence plots
  - Model decision boundary visualization

## �🤝 Contributing

Feel free to contribute to this project by implementing any of the future improvements listed above, or by:
1. Adding new algorithms and ensemble methods
2. Improving preprocessing techniques and data quality
3. Enhancing visualization capabilities and interpretability
4. Optimizing performance and scalability
5. Implementing MLOps best practices

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions, please open an issue in the repository.
