# Car-Classification--Don-t-Get-Kicked-by-Lalit-Hire
Authour: Lalit Kailas Hire (BSC Data Science Department of Technology, Savitribai Phule Pune University)
I did this project for the Practice!
NOTE: IPYNB FILE DOES NOT CONTAIN OUTPUT'S DUE TO SIZE ISSUES. 
📌 Introduction
This project is based on the "Don't Get Kicked!" dataset from Kaggle, where the goal is to predict whether a used car purchase is a bad buy using structured auction data. Dealers need to avoid bad purchases that may result in losses, and this model helps them make better, data-driven decisions.

🧠 Problem Statement
Objective:
Given a dataset of used car auction transactions, predict whether a car is a "bad buy" (IsBadBuy = 1) or not (IsBadBuy = 0) based on features like vehicle age, make, odometer reading, market prices, and more.

This is a binary classification problem with imbalanced classes, as the number of good buys is significantly higher than bad buys.

📊 Dataset Overview
Source: Kaggle - Don't Get Kicked!

Type: Tabular data

Size: ~70,000+ records

Target Variable: IsBadBuy

Features Include:

Vehicle and auction information (e.g., VehicleAge, Auction)

Pricing data (MMRAcquisitionAuctionCleanPrice, etc.)

Vehicle condition (Transmission, WheelType, etc.)

Geographic and seller-related features

🔍 Exploratory Data Analysis (EDA) Insights
Detected missing values in several categorical and numerical columns.

Observed correlation between auction prices and the IsBadBuy label.

Some features like VehicleAge, Odometer, and Transmission had strong influence on predictions.

The dataset had imbalance in the target variable (bad buys are fewer).

🧰 Preprocessing & Modeling
Missing value imputation for numerical and categorical columns.

One-hot encoding for categorical variables.

Feature scaling (if needed).

Used Random Forest Classifier for final predictions.

Evaluation based on ROC AUC Score and Probability Outputs.

✅ Results
Model outputs probabilities for each vehicle being a bad buy.

Saved predictions in a submission file: rf_Submissions.csv

A sample of top predictions is printed for quick validation.

Used_cars_classification.ipynb   # Main Jupyter notebook with full code
rf_Submissions.csv               # Final predictions ready for submission


 Future Improvements
Try advanced models like XGBoost or LightGBM

Handle class imbalance using SMOTE or class weights

Add feature selection and hyperparameter tuning

Improve interpretability using SHAP values
