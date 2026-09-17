# Loan Approval Prediction

## Overview

Loan approval decisions are critical for both financial institutions and applicants. Approving a high-risk applicant results in financial loss for the bank, while rejecting a creditworthy applicant means lost business and potential reputational harm.

This project builds a binary classification machine learning pipeline to predict loan approval outcomes (`Approved` vs. `Rejected`) based on applicant financial profiles, asset portfolios, and credit histories.

## Dataset

* **Source:** Kaggle — Loan Approval Prediction Dataset
* **Size:** 4,269 records, 13 features
* **Target Variable:** `loan_status` (`Approved` / `Rejected`)
* **Class Distribution:** 62% Approved, 38% Rejected

### Features Summary

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `no_of_dependents` | Numerical | Number of dependents of the applicant |
| `education` | Categorical | Education level (`Graduate` / `Not Graduate`) |
| `self_employed` | Categorical | Employment status (`Yes` / `No`) |
| `income_annum` | Numerical | Annual income of the applicant |
| `loan_amount` | Numerical | Requested loan amount |
| `loan_term` | Numerical | Duration of the loan (in years) |
| `cibil_score` | Numerical | Credit score (300–900) |
| `residential_assets_value` | Numerical | Total value of residential assets |
| `commercial_assets_value` | Numerical | Total value of commercial assets |
| `luxury_assets_value` | Numerical | Total value of luxury assets |
| `bank_asset_value` | Numerical | Total value of bank assets |

## Project Objectives

1. **Feature Engineering & Analysis:** Identify key financial drivers and asset attributes influencing loan decisions.
2. **Data Preprocessing:** Handle whitespaces, drop non-predictive identifiers (`loan_id`), encode categorical features, and scale numerical variables.
3. **Model Evaluation:** Train and compare multiple classification models (Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost) using Accuracy, F1 Score, and Log Loss.
4. **Risk Assessment:** Evaluate model reliability and potential business limitations.

## Tech Stack

* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy, SciPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost

## Project Workflow

```text
├── Data Ingestion & Exploration
│   ├── Whitespace stripping & Identifier removal (`loan_id`)
│   └── Duplicate & missing value inspection
├── Exploratory Data Analysis (EDA)
│   ├── Categorical distributions vs. Loan Status
│   └── Asset & Income correlations
├── Data Preprocessing
│   ├── Categorical Encoding (OneHotEncoder / LabelEncoder)
│   ├── Outlier treatment (Winsorization)
│   └── Feature Scaling (StandardScaler)
├── Model Training & Comparison
│   ├── Logistic Regression
│   ├── K-Nearest Neighbors (KNN)
│   ├── Random Forest Classifier
│   └── XGBoost Classifier
└── Evaluation & Final Selection
