import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
"Load the Dataset"
ds = pd.read_csv("hospital data analysis.csv")
" EDA- Exploratory Data Analysis"
Pat_sat_head = (ds.head())
print(Pat_sat_head)
Pat_sat_info = (ds.info())
print(Pat_sat_info)
Pat_sat_describe = (ds.describe())
print(Pat_sat_describe)
Pat_sat_mis_val = (ds.isnull().sum())
print(Pat_sat_mis_val)
ds_encoded_Prod = LabelEncoder().fit_transform(ds['Procedure'])
print(ds_encoded_Prod)
ds_encoded_Gendr = LabelEncoder().fit_transform(ds['Gender'])
print(ds_encoded_Gendr)
ds_encoded_Cond = LabelEncoder().fit_transform(ds['Condition'])
print(ds_encoded_Cond)
ds_encoded_Readm = LabelEncoder().fit_transform(ds['Readmission'])
print(ds_encoded_Readm)
ds_encoded_Out = LabelEncoder().fit_transform(ds['Outcome'])
print(ds_encoded_Out)

"Checking for Outliers"
ds_Length_of_Stay= ds['Length_of_Stay'].quantile(0.25)
print(ds_Length_of_Stay)
ds_Satisfaction = ds['Satisfaction'].quantile(0.50)
print(ds_Satisfaction)
'IQR(Inter-Quantile Range) = Middle Range'
IQR = ds_Satisfaction-ds_Length_of_Stay
lower_bound = ds_Length_of_Stay-1.5*IQR
print(lower_bound)
upper_bound =readmission_counts = ds['Readmission'].value_counts()

plt.figure()
plt.bar(readmission_counts.index, readmission_counts.values)
plt.xlabel("Readmission")
plt.ylabel("Count")
plt.title("Readmission Distribution")
plt.show()
print(upper_bound)

'Outlier-visualize'
plt.boxplot(ds['Length_of_Stay'],vert=False)
plt.title("Length_of_Stay")
plt.tight_layout()
plt.show()

plt.boxplot(ds['Satisfaction'],vert=False)
plt.title("Satisfaction")
plt.tight_layout()
plt.show()

"Visualization"
"Average Cost by Readmission"
avg_cost = ds.groupby("Readmission")["Cost"].mean()
plt.figure()
plt.bar(avg_cost.index, avg_cost.values)
plt.xlabel("Readmission")
plt.ylabel("Average Cost")
plt.title("Average Treatment Cost by Readmission")
plt.show()

"Length of Stay vs Readmission"
avg_los = ds.groupby("Readmission")["Length_of_Stay"].mean()
plt.figure()
plt.bar(avg_los.index, avg_los.values)
plt.xlabel("Readmission")
plt.ylabel("Average Length of Stay")
plt.title("Length of Stay vs Readmission")
plt.show()

"Condition-wise Readmission Rate"
dept_readmission = pd.crosstab(ds['Condition'], ds['Readmission'])
dept_readmission.plot(kind='bar')
plt.title("Condition-wise Readmission")
plt.xlabel("Condition")
plt.ylabel("Count")
plt.show()

"Age Distribution"
plt.figure()
plt.hist(ds['Age'], bins=10)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Patients")
plt.show()

"Preprocessing"
ds['Readmission'] = LabelEncoder().fit_transform(ds['Readmission'])
"Encoding categorical features"
categorical_cols = ds.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
 ds[col] = LabelEncoder().fit_transform(ds[col])

"Define features and target"
X = ds.drop("Readmission", axis=1)
y = ds["Readmission"]

"Train-test split"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

"Feature Scaling"
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"Build Logistic Regression Model"
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

"Metrics"
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

"Improvising Model"
"Feature Scaling"
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"Improved Logistic Regression"
log_model = LogisticRegression(max_iter=5000)

"Hyperparameter tuning"
param_grid = { 'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']}
grid_log = GridSearchCV(log_model, param_grid, cv=5)
grid_log.fit(X_train_scaled, y_train)
best_log = grid_log.best_estimator_
y_pred_log = best_log.predict(X_test_scaled)
y_prob_log = best_log.predict_proba(X_test_scaled)[:, 1]
print("Best Logistic Parameters:", grid_log.best_params_)
print("Logistic Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

"Random Forest Classifier"
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]}
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print("Best Random Forest Parameters:", grid_rf.best_params_)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

"Feature Importance"
"Logistic Regression Feature Importance"
log_importance = pd.Series(
    best_log.coef_[0],
    index=X.columns
).sort_values(ascending=False)

print("\nLogistic Regression Feature Importance:")
print(log_importance)

"Random Forest Feature Importance"
rf_importance = pd.Series(
    best_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

"Plot Random Forest Importance"
plt.figure()
rf_importance.plot(kind='bar')
plt.title("Feature Importance - Random Forest")
plt.show()

"ROC Curve Comparison"
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.figure()
plt.plot(fpr_log, tpr_log)
plt.plot(fpr_rf, tpr_rf)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Logistic vs Random Forest")
plt.show()
print("Logistic AUC:", auc(fpr_log, tpr_log))
print("Random Forest AUC:", auc(fpr_rf, tpr_rf))

"Cross-validation"
X = ds.drop('Readmission', axis=1)
y = ds['Readmission']

X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier( n_estimators=100,max_depth=5, min_samples_leaf=5, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross Validation Score:", scores.mean())