# END_TO_END_PROJECT

# 🌾 Yield Prediction and Analysis using Machine Learning

## 📘 Overview

This project performs comprehensive data analysis, modeling, and visualization on an agricultural yield dataset. It covers all stages of the machine learning workflow — from preprocessing and regression/classification modeling to clustering and PCA-based dimensionality reduction.

It serves as an end-to-end example of supervised and unsupervised learning using Scikit-learn.

## Yield Prediction App:


## 🚀 Features
### 🔹 1. Data Preprocessing

Handles missing values and categorical features using factorization.

Applies StandardScaler for normalization.

Supports both numerical and categorical columns automatically.
<img width="1189" height="396" alt="image" src="https://github.com/user-attachments/assets/81f4ffe0-9481-493f-a960-e761bd4bdc4a" />

outliers:
<img width="989" height="397" alt="image" src="https://github.com/user-attachments/assets/566b1dff-73f1-4a4c-bdb5-4077d5260d2e" />

### 🔹 2. Regression Models (Predicting Yield Value)

Compares multiple regression algorithms and evaluates performance with R² and MSE:

Simple Linear Regression (top correlated feature)

Multiple Linear Regression

Ridge Regression

Lasso Regression

Polynomial Regression (degree = 2)

Random Forest Regressor

Gradient Boosting Regressor

### 🔹 3. Classification Models (High vs Low Yield)

Converts target into binary classes using the median split and evaluates:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

Naive Bayes

AdaBoost

Gradient Boosting

(Optional) XGBoost (if installed)

Metrics:

Accuracy

Precision

Recall

F1-Score

### 🔹 4. Clustering (Unsupervised Learning)

Groups similar data points using unsupervised techniques:

KMeans

Agglomerative (Hierarchical) Clustering

DBSCAN

Displays cluster membership counts for each method.

## 📊 Example Outputs

#### ✅ Regression Results
Model	R²	MSE
RandomForestRegressor	0.85	10245.12
GradientBoostingRegressor	0.83	11890.23
Ridge	0.79	14233.51

#### ✅ Classification Results
Model	Accuracy	Precision	Recall	F1-score
Random Forest	0.91	0.92	0.89	0.91
KNN	0.86	0.86	0.86	0.86
SVM	0.83	0.86	0.79	0.82
🧩 Project Structure
├── yield.csv               # Input dataset
├── yield_analysis.py       # Main script (this file)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

### ⚙️ Installation

Clone this repository:

git clone https://github.com/yourusername/yield-prediction-ml.git
cd yield-prediction-ml


Install dependencies:

pip install -r requirements.txt


requirements.txt example:

pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost


Place your dataset file as:

yield.csv


Run the main script:

python yield_analysis.py

### 🧠 Technologies Used

Python 3.x

Scikit-learn – Machine Learning models

Matplotlib / Seaborn – Visualization

Pandas / NumPy – Data manipulation

XGBoost (optional) – Advanced boosting model

### 📈 Visualizations

PCA 2D scatter plot (colored by class)
<img width="789" height="590" alt="image" src="https://github.com/user-attachments/assets/16ced9f2-794e-43d0-8a66-32c9567a41a0" />

PCA cumulative variance plot
<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/989a5d45-ec48-4113-b1b4-d485fca564c5" />

Cluster count summaries

### 🏁 Results Summary

This project provides a side-by-side performance comparison across regression and classification models.
It also visualizes dimensionality reduction via PCA and explores unsupervised structure using clustering.

### 📚 Future Enhancements

Hyperparameter tuning with GridSearchCV

Model persistence (Pickle joblib saving)

Streamlit dashboard for interactive results visualization

SHAP/LIME for explainable AI insights
