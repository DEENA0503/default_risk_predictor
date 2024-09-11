# Default Risk Predictor

## App link: https://credit-risk-modeling-hgn9.onrender.com
* The app is deployed and hosted on Render.com, but the free tier only runs the app when accessed, and it goes to sleep during inactivity
  
## Overview
The Default Risk Predictor is a Flask-based application that predicts whether a borrower is likely to default on a loan based on various financial and demographic factors. It uses a **LightGBM classifier** to solve a binary classification problem, predicting loan default with a high level of accuracy and precision.

## Key Features:
* Predicts loan default status with an **F1 score** of **0.8279** and **Accuracy** of **0.9327**.
* Provides a detailed **risk percentage** based on prediction probabilities.
* Preprocessed data with outlier removal, imputation, and feature scaling.
* Handles data imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
* Categorical features are one-hot encoded, and numerical features are standardized for optimal performance.
* Deployed using **Render.com** for live access.

## Usage:
* ### Input the required data fields:
  * Age of the individual
  * Annual income
  * Home ownership status (Rent, Mortgage, Own, Other)
  * Employment length (in years)
  * Loan intent (e.g., Personal, Home Improvement)
  * Loan grade (A to G)
  * Loan amount
  * Percent of income required to repay the loan
  * Credit bureau's historical default status (Yes/No)
  * Credit history length (in years)

* ### Click on the "**Predict Default status**" button to get:
  * The predicted loan status (default or non-default).
  * Risk percentage for providing a loan to the borrower.
  
* ### Click on the "Reset" button to:
  * reset the app
* ### Click on the "Go to Home" button  to:
  * use the predictor again with new values

## Model Information

* **Algorithm**: LightGBM (LGBMClassifier)

* **Dataset Shape**: (32581, 12)

* **Label**: loan_status where 1 indicates default and 0 indicates non-default.

* **Metrics**:
  * **F1 Score**: 0.8279
  * **Accuracy**: 0.9327
  * **Matthews correlation coefficient (MCC)**: 0.7953

* **Preprocessing**:
* Duplicated data is dropped.
* Outliers are removed.
* Missing values are imputed using **IterativeImputer**(which uses **KNeighborsRegressor** as an estimator).
* Numerical columns are scaled using **StandardScaler**.
* Categorical columns are one-hot encoded using **OneHotEncoder**.
* **SMOTE** is applied to oversample the minority class.
* Hyperparameter tuning was performed using **RandomizedSearchCV** to optimize model performance.

## Important Notes:
* The loan_int_rate column was dropped because interest rates are determined post-loan approval and are based on the risk assessment of the borrower.
* The model outputs the default probability, which is communicated as a risk percentage.

## Technologies Used
* **Preprocessing**: Pandas, Numpy, Scikit-learn (IterativeImputer(KNeighborsRegressor), StandardScaler, OneHotEncoder, SMOTE, RandomizedSearchCV)
* **Machine Learning**: LightGBM (LGBMClassifier), Scikit-learn, XGBoost (XGBClassifier)
* **Model Saving**: Pickle (.pkl format)
* **Backend**: Flask
* **Frontend**: HTML, Jinja, CSS
* **Deployment**: Render.com
