# ML-churn-modeling
A machine learning project for predicting customer churn using a bank dataset. The project includes data preprocessing, exploratory data analysis (EDA), and training multiple models (Decision Tree, Random Forest, Logistic Regression, SVM, and MLPClassifier) to predict customer churn. Implemented in Python using Pandas, Scikit-learn, Matplotlib, and SMOTE for handling imbalanced data.
# Customer Churn Prediction

## Overview
This project focuses on predicting customer churn for a bank using a dataset containing customer details such as credit score, geography, gender, age, and account activity. Multiple machine learning models, including Decision Tree, Random Forest, Logistic Regression, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLPClassifier), are implemented to predict whether a customer will leave the bank. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and handling imbalanced data using SMOTE.

## Features
- **Data Preprocessing**: Dropping irrelevant columns, encoding categorical variables, and feature scaling using StandardScaler.
- **Exploratory Data Analysis (EDA)**: Visualizing customer demographics and account balances by geography using pie charts and value counts.
- **Machine Learning Models**: Training and evaluating Decision Tree, Random Forest, Logistic Regression, SVM, and MLPClassifier.
- **Handling Imbalanced Data**: Applying SMOTE to address class imbalance in the target variable.
- **Model Evaluation**: Using accuracy, precision, recall, F1-score, and confusion matrix for performance assessment.
- **Cross-Validation**: Performing 10-fold cross-validation to ensure model robustness.

## Dataset
The dataset (`Churn.csv`) contains 10,000 customer records with the following features:
- CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited (target variable).

## Requirements
- Python 3.11
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `imblearn`, `joblib`, `scikeras`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
  
2.Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3.Run the Jupyter Notebook:
```bash
  jupyter notebook churn_modeling.ipynb
```
## Usage
- Load and preprocess the dataset (handle missing values, encode categorical variables, scale features).
- Perform EDA to understand customer demographics and churn patterns.
- Train and evaluate multiple machine learning models.
- Use cross-validation to assess model robustness.
- Save the final model using joblib for future predictions.

## Model Performance
- Decision Tree: Training Accuracy: 100%, Testing Accuracy: 79.84%
- Random Forest: Training Accuracy: 100%, Testing Accuracy: 86.92%
- Logistic Regression: Training Accuracy: 80.96%, Testing Accuracy: 80.68%
- SVM: Training Accuracy: 86.25%, Testing Accuracy: 86.08%
- MLPClassifier: Training Accuracy: 88.60%, Testing Accuracy: 86.04%


## Future Improvements
- Perform hyperparameter tuning to improve model performance.
- Experiment with advanced techniques like XGBoost or deep learning models.
- Enhance feature engineering to capture more predictive patterns.

