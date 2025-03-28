# Multiple Disease Prediction System WebApp
This repository contains a Multiple Disease Prediction System WebApp developed using Streamlit and hosted on Streamlit Cloud. The web app integrates three different disease prediction systems, each utilizing machine learning models to provide accurate predictions. The diseases covered are:


1. Diabetes Prediction System

2. Heart Disease Prediction System

3. Breast Cancer Prediction System


Table of Contents:

* Overview
* Usage
* Dataset Description
* Technologies Used
* Model Development Process
* Models Used
* Model Evaluation
* Conclusion

# Overview
This web application allows users to select from four different disease prediction systems and get predictions based on the input features. Each prediction system was developed through extensive data analysis and model selection processes, ensuring high accuracy and reliability.


# Usage
To start the Streamlit web app, run the following command in your terminal: streamlit run streamlit_app.py
This will launch the web app in your default web browser. You can then select the desired disease prediction system from the sidebar and input the required features to get a prediction.

# Dataset Description
1. Diabetes Prediction System

Description: This dataset contains 768 instances of patient data, with 8 features including glucose levels, blood pressure, and insulin levels, used to predict diabetes.

2. Heart Disease Prediction System

Description: This dataset includes 1025 instances with 14 features such as age, sex, chest pain type, and resting blood pressure, used to predict the presence of heart disease.

3. Breast Cancer Prediction System

Description: This dataset contains 569 instances with 30 features such as radius, texture, perimeter, and area, used to predict breast cancer.


# Technologies Used
Programming Language: Python

Web Framework: Streamlit

Machine Learning Libraries: Scikit-learn, XGBoost

Data Analysis and Visualization: Pandas, NumPy, Matplotlib, Seaborn


# Model Development Process
Each disease prediction system was developed through the following steps:

1. Importing the Dependencies

2. Exploratory Data Analysis (EDA)

3. Data Preprocessing
   * Handling missing values
   * Handling outliers
   * Label encoding/One-hot encoding
   * Standardizing the data

4. Model Selection
   * Selected the most common 5 classification models
   * Trained each model and checked cross-validation scores
   * Chose the top 3 models based on cross-validation scores

5. Model Building and Evaluation
   * Selected best features using Recursive Feature Elimination (RFE)
   * Performed hyperparameter tuning using Grid Search CV
   * Built the final model with the best hyperparameters and features
   * Evaluated the model using classification reports


# Models Used
The top 3 models for each disease prediction system are as follows:

1. Diabetes Prediction System
- Support Vector Classifier: Effective in high-dimensional spaces.
- Logistic Regression: Simple and effective binary classification model.
- Random Forest Classifier: Ensemble method that reduces overfitting.

2. Heart Disease Prediction System
- XGBoost: Boosting algorithm known for high performance.
- Random Forest Classifier: Robust and handles missing values well.
- Logistic Regression: Interpretable and performs well with binary classification.

3. Breast Cancer Prediction System
- Logistic Regression: Highly interpretable and performs well with binary classification.
- XGBoost: Excellent performance with complex datasets.
- K-Nearest Neighbour: Effective with smaller datasets and straightforward implementation.


# Model Evaluation

1. Diabetes Prediction System
Model	Accuracy
- Support Vector Classifier	78.480%
- Logistic Regression	70.129%
- Random Forest Classifier	75.324%

2. Heart Disease Prediction System
Model	Accuracy
- XGBoost	98.324%
- Random Forest Classifier	93.411%
- Logistic Regression	88.311%%

4. Breast Cancer Prediction System
Model	Accuracy
- Logistic Regression	97.368%
- XGBoost	97.368%
- K-Nearest Neighbour	96.491%

# Dataset Links:-

1. Diabetes Dataset
https://www.kaggle.com/datasets/mathchi/diabetes-data-set

2. Heart Disease Dataset
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

3. Breast Cancer Dataset
https://www.kaggle.com/code/jainilcoder/breast-cancer-prediction/input

# Conclusion
This Multiple Disease Prediction System WebApp provides an easy-to-use interface for predicting the likelihood of various diseases based on input features. The models used are well-validated and tuned for high accuracy. The system aims to assist in early diagnosis and better decision-making in healthcare.