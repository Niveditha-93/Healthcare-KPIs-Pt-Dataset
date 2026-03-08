
🏥 Hospital Readmission Prediction using Machine Learning

📌 Project Overview

Hospital readmissions significantly increase healthcare costs and indicate potential gaps in patient care.This project uses Machine Learning techniques to predict whether a patient is likely to be readmitted based on hospital records.The goal is to help healthcare providers identify high-risk patients early and improve treatment planning.

📊 Dataset
The dataset contains hospital patient information including:
*Age
*Gender
*Medical Condition
*Procedure
*Length of Stay
*Treatment Cost
*Patient Satisfaction
*Treatment Outcome
*Readmission Status (Target Variable)
*Total records: 984 patients

🔬 Project Workflow

1️Exploratory Data Analysis (EDA)
2️⃣Data Cleaning & Encoding
3️⃣Feature Scaling
4️Model Training
5️⃣Hyperparameter Tuning
6️⃣Model Evaluation
7️⃣Feature Importance Analysis
8️⃣Cross Validation

📈 Visualizations
Readmission Distribution

Feature Importance

ROC Curve Comparison

🤖 Machine Learning Model

Model	Accuracy	AUC
Logistic Regression	0.868	0.867
Random Forest	1.00	1.00

Random Forest significantly outperformed Logistic Regression due to its ability to capture nonlinear relationships and feature interactions.

🔑 Key Insights

-Length of Stay is one of the strongest predictors of hospital readmission.

-Certain medical conditions increase the probability of readmission.

-Higher treatment costs often indicate complex medical cases.

-Random Forest effectively captures complex relationships in healthcare data.

-Predictive analytics can help hospitals reduce readmission rates and improve patient care.

⚙️Technologies Used

*Python
*Pandas
*NumPy
*Matplotlib
*Scikit-Learn

📊 Model Evaluation Metrics

-Accuracy
-Precision
-Recall
-F1 Score
-ROC-AUC
-Cross Validation

Cross-validation score: 0.993

📌Conclusion

This project demonstrates how machine learning models can successfully predict hospital readmissions using patient demographics and treatment data. Such predictive systems can help healthcare providers identify high-risk patients and improve hospital management strategies.

