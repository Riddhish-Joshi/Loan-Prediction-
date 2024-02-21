# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import sklearn.metrics as metrics
from statsmodels.formula.api import ols
import statsmodels.api as sm
from tabulate import tabulate
import warnings

# Suppressing warnings
warnings.filterwarnings("ignore")

# Reading the dataset
loan_original = pd.read_csv("C:/Users/riddh/OneDrive/Desktop/Project/modified_loan_approval_dataset_new4.csv")

# Displaying the first few rows of the dataset
print(loan_original.head())

# Displaying information about the dataset
print(loan_original.info())

# Descriptive statistics of the dataset
print(loan_original.describe(include="all"))

# Checking for duplicated loan IDs
loan_original[loan_original['loan_id'].duplicated(keep=False) == True].sort_values(['loan_id'])

# Removing spaces from column names
loan_original.columns = loan_original.columns.str.replace(' ', '')

# Dropping the 'loan_id' column as it seems to be an identifier
loan = loan_original.drop(['loan_id'], axis=1)

# Exploratory Data Analysis (EDA)

# Pairplot to visualize relationships between numerical variables
sns.pairplot(loan)

# Checking for outliers in 'loan_amount' using a boxplot
sns.boxplot(loan['loan_amount'])
plt.title("Loan Amount")
plt.xlabel("Loan Amount")
plt.show()

# Visualizing the relationship between 'loan_amount' and 'loan_status' using a histogram
sns.histplot(loan, x='loan_amount', hue='loan_status')
plt.title("Does loan status relate to the loan amount?")
plt.xlabel("Loan Amount")
plt.ylabel("Count")
plt.show()

# Scatter plot to visualize the relationship between 'income_annum', 'loan_amount', and 'loan_status'
sns.scatterplot(x=loan['income_annum'], y=loan['loan_amount'], hue=loan['loan_status'])
plt.title("Loan Status, Loan Amount, Annual Income")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.show()

# Finding the entry with maximum 'income_annum' and 'loan_status' as 'Rejected'
print(loan.loc[(loan['income_annum'] == loan['income_annum'].max()) & (loan['loan_status'] == 'Rejected')])

# Scatter plot to visualize the relationship between 'cibil_score', 'loan_amount', and 'loan_status'
sns.scatterplot(x=loan['cibil_score'], y=loan['loan_amount'], hue=loan['loan_status'])
plt.title("Loan Status, Loan Amount, Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Loan Amount")
plt.show()

# Finding entries with 'cibil_score' greater than 740 and 'loan_status' as 'Rejected'
print(loan.loc[(loan['cibil_score'] > 740) & (loan['loan_status'] == 'Rejected')].sort_values(['cibil_score'], ascending=False))

# Visualizing different assets values distribution by loan status using histograms
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
sns.histplot(loan, x='residential_assets_value', hue='loan_status', ax=axes[0, 0])
axes[0, 0].set_xlabel("Residential Assets Value")
axes[0, 0].set_ylabel("Count")
# Similarly for other asset types
plt.tight_layout()
plt.show()

# Scatter plot to visualize relationships between assets values and annual income by loan status
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
sns.scatterplot(loan, x='residential_assets_value', y='income_annum', hue='loan_status', ax=axes[0, 0])
axes[0, 0].set_xlabel("Residential Assets Value")
# Similarly for other asset types
plt.tight_layout()
plt.show()

# Crosstab and visualization for loan status by loan term
cross_loan_term = pd.crosstab(index=loan['loan_term_months'], columns=loan['loan_status'])
# Calculating percentages
cross_loan_term['Total'] = cross_loan_term['Approved'] + cross_loan_term['Rejected']
cross_loan_term['Approved_percentage'] = (cross_loan_term['Approved'] / cross_loan_term['Total']) * 100
cross_loan_term['Rejected_percentage'] = (cross_loan_term['Rejected'] / cross_loan_term['Total']) * 100
print(cross_loan_term)

# Plotting loan status by loan term
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
cross_loan_term.plot(kind='line', marker='o', ax=axes[0])
axes[0].set_xlabel('Loan Term')
axes[0].set_ylabel('Count')
axes[0].set_title('Loan Status by Loan Term')

# Scatter plot to visualize the relationship between 'loan_term_months', 'loan_amount', and 'loan_status'
sns.scatterplot(loan, x='loan_term_months', y='loan_amount', hue='loan_status', ax=axes[1])
axes[1].set_title("Loan Status, Loan Amount, Loan Term")
axes[1].set_xlabel("Loan Term")
axes[1].set_ylabel("Loan Amount")
plt.tight_layout()
plt.show()

# Finding entries with 'loan_term_months' less than or equal to 4 and 'loan_status' as 'Rejected'
print(loan.loc[(loan['loan_term_months'] <= 4) & (loan['loan_status'] == 'Rejected')].sort_values(['loan_amount']).head(10))

# Histogram for 'no_of_dependents'
plt.figure(figsize=(5, 3))
sns.histplot(loan['no_of_dependents'])
plt.xlabel("Number of dependents")
plt.title("Histogram of Number of dependents")
plt.show()

# Crosstab and visualization for loan status by number of dependents
cross_dependents = pd.crosstab(index=loan['no_of_dependents'], columns=loan['loan_status'])
cross_dependents['Approved_percentage'] = (cross_dependents['Approved'] / (cross_dependents['Approved'] + cross_dependents['Rejected'])) * 100
cross_dependents['Rejected_percentage'] = (cross_dependents['Rejected'] / (cross_dependents['Approved'] + cross_dependents['Rejected'])) * 100
print(cross_dependents)

# Plotting counts for different levels of education
plt.figure(figsize=(6, 4))
sns.countplot(loan, x='education', hue='loan_status')
plt.xlabel("Education")
plt.title("Counts for education")
plt.show()

# Grouping by education level and aggregating statistics
education = loan.groupby(["education"], as_index=False).agg(
    count_by_education=("education", "count"),
    median_annual_income=("income_annum", "median"),
    average_loan_amount=("loan_amount", "mean"),
    average_credit_score=("cibil_score", "mean"),
    average_loan_term=("loan_term_months", "mean"),
    avg_residential_value=('residential_assets_value', "mean"),
    avg_commerical_value=('commercial_assets_value', "mean"),
    avg_luxury_value=('luxury_assets_value', "mean"),
    avg_bank_value=('bank_asset_value', "mean")).round(2).reset_index(drop=True)

print(education)

# Grouping by self employment status and aggregating statistics
self_employed = loan.groupby(["self_employed"], as_index=False).agg(
    count=("education", "count"),
    median_annual_income=("income_annum", "median"),
    average_loan_amount=("loan_amount", "mean"),
    average_credit_score=("cibil_score", "mean"),
    average_loan_term=("loan_term_months", "mean"),
    avg_residential_value=('residential_assets_value', "mean"),
    avg_commerical_value=('commercial_assets_value', "mean"),
    avg_luxury_value=('luxury_assets_value', "mean"),
    avg_bank_value=('bank_asset_value', "mean")).round(2).reset_index(drop=True)

print(self_employed)

# Performing Chi-Square test for number of dependents, education, and self employment status
contingency_dependents = pd.crosstab(loan['no_of_dependents'], loan['loan_status'])
chi2_dependents, p_dependents, dof_dependents, expected_dependents = stats.chi2_contingency(contingency_dependents)
print("Chi-Square Value:", chi2_dependents)
print("p-value:", p_dependents)
print("Degrees of Freedom:", dof_dependents)
print("Expected Frequencies Table:")
print(expected_dependents)

contingency_education = pd.crosstab(loan['education'], loan['loan_status'])
chi2_education, p_education, dof_education, expected_education = stats.chi2_contingency(contingency_education)
print("Chi-Square Value:", chi2_education)
print("p-value:", p_education)
print("Degrees of Freedom:", dof_education)
print("Expected Frequencies Table:")
print(expected_education)

contingency_self_employed = pd.crosstab(loan['self_employed'], loan['loan_status'])
chi2_self_employed, p_self_employed, dof_self_employed, expected_self_employed = stats.chi2_contingency(contingency_self_employed)
print("Chi-Square Value:", chi2_self_employed)
print("p-value:", p_self_employed)
print("Degrees of Freedom:", dof_self_employed)
print("Expected Frequencies Table:")
print(expected_self_employed)

# Creating dummy variables for categorical variables
loan_dummies = pd.get_dummies(loan)
loan_dummies.rename(columns={'education_Graduate': 'education', 'self_employed_ Yes': 'self_employed', 'loan_status_Approved': 'loan_status'}, inplace=True)
loan_dummies = loan_dummies.drop(['education_Not Graduate', 'self_employed_No', 'loan_status_Rejected'], axis=1)
print(loan_dummies.columns)

# Correlation matrix heatmap
loan_corr = loan_dummies.corr()
plt.figure(figsize=(9, 9))
sns.heatmap(loan_corr, annot=True, fmt=".2f", cmap="coolwarm")

# Model Building

# Splitting the data into train, validation, and test sets
y = loan_dummies['loan_status']
X = loan_dummies.drop(['loan_status'], axis=1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# Scaling features
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_val_stand = scaler.transform(X_val)
X_test_stand = scaler.transform(X_test)

# Logistic Regression
clf = LogisticRegression().fit(X_train_stand, y_train)
coefficients = clf.coef_[0]
intercept = clf.intercept_
variables = list(X_train.columns)
clf_summary = []
for var, coef in zip(variables, coefficients):
    clf_summary.append([var, coef])
clf_summary.append(["Intercept", intercept[0]])
print(tabulate(clf_summary, headers=["Variables", "Coefficient"], tablefmt="grid"))

# Evaluating Logistic Regression model
y_lr = clf.predict(X_val_stand)
print('Accuracy:', '%.3f' % accuracy_score(y_val, y_lr))
print('Precision:', '%.3f' % precision_score(y_val, y_lr))
print('Recall:', '%.3f' % recall_score(y_val, y_lr))
print('F1 Score:', '%.3f' % f1_score(y_val, y_lr))

# Random Forest
param_dist = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 10, 20],
    'min_samples_split': [50, 100, 150],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_forest = random_search.best_estimator_
print(best_params, best_forest)

# Optimized Random Forest model
rf_opt = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=1, min_samples_split=5, random_state=23)
rf_opt.fit(X_train, y_train)
y_rf = rf_opt.predict(X_val)
print('Accuracy:', '%.3f' % accuracy_score(y_val, y_rf))
print('Precision:', '%.3f' % precision_score(y_val, y_rf))
print('Recall:', '%.3f' % recall_score(y_val, y_rf))
print('F1 Score:', '%.3f' % f1_score(y_val, y_rf))

# Evaluating on test set
y_test_rf = rf_opt.predict(X_test)
print('Accuracy:', '%.3f' % accuracy_score(y_test, y_test_rf))
print('Precision:', '%.3f' % precision_score(y_test, y_test_rf))
print('Recall:', '%.3f' % recall_score(y_test, y_test_rf))
print('F1 Score:', '%.3f' % f1_score(y_test, y_test_rf))

# Confusion matrix visualization
cm = metrics.confusion_matrix(y_val, y_rf, labels=rf_opt.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_opt.classes_)
disp.plot()

# Feature importances
importances = rf_opt.feature_importances_
forest_importances = pd.Series(importances, index=X.columns)
forest_importances_sorted = forest_importances.sort_values(ascending=False)

# Plotting feature importances
fig, ax = plt.subplots()
forest_importances_sorted.plot.bar(ax=ax)
