# Task-5-Mainflow

import pandas as pd

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv('heart.csv')

# Display basic information about the dataset
print(df.info())

# Display first few rows
print(df.head())

# Summary statistics
print(df.describe())
# Check for missing values
print(df.isnull().sum())

# Handle missing values (impute with mean for continuous, mode for categorical)
df['age'].fillna(df['age'].mean(), inplace=True)
df['sex'].fillna(df['sex'].mode()[0], inplace=True)

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Confirm the changes
print(df.head())
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the distribution of 'Age' and 'Cholesterol'
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['age'], kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['chol'], kde=True, color='green')
plt.title('Cholesterol Distribution')

plt.tight_layout()
plt.show()
# Compute the correlation matrix
corr = df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

## QUESTIONS

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv('heart_disease.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# 1. What is the average age of patients with heart disease and without heart disease?
print("\nAverage Age of Patients with and without Heart Disease:")
avg_age = df.groupby('Target')['Age'].mean()
print(avg_age)

# 2. What is the distribution of cholesterol levels in patients with heart disease compared to those without?
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Target'] == 1]['Cholesterol'], kde=True, color='red', label='Heart Disease', bins=15)
sns.histplot(df[df['Target'] == 0]['Cholesterol'], kde=True, color='green', label='No Heart Disease', bins=15)
plt.legend()
plt.title('Cholesterol Levels Distribution by Heart Disease Status')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# 3. What is the correlation between age, cholesterol, and resting blood pressure in predicting heart disease?
print("\nCorrelation Matrix (Age, Cholesterol, RestingBloodPressure, Target):")
correlation_matrix = df[['Age', 'Cholesterol', 'RestingBloodPressure', 'Target']].corr()
print(correlation_matrix)

# Plot heatmap to visualize correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# 4. What percentage of male and female patients have heart disease?
print("\nPercentage of Male and Female Patients with Heart Disease:")
gender_heart_disease = df.groupby(['Sex', 'Target']).size().unstack().fillna(0)
gender_heart_disease_percent = (gender_heart_disease / gender_heart_disease.sum(axis=1)) * 100
print(gender_heart_disease_percent)

# 5. What is the average maximum heart rate achieved during exercise for patients with and without heart disease?
print("\nAverage Max Heart Rate Achieved by Heart Disease Status:")
avg_max_heart_rate = df.groupby('Target')['MaxHeartRateAchieved'].mean()
print(avg_max_heart_rate)

# 6. Which age group has the highest number of heart disease patients?
print("\nHeart Disease Count by Age Group:")
bins = [29, 39, 49, 59, 69, 79, 89]
labels = ['30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_group_heart_disease = df[df['Target'] == 1]['AgeGroup'].value_counts()
print(age_group_heart_disease)

# 7. Build a logistic regression model to predict the likelihood of heart disease based on relevant features.
print("\nBuilding Logistic Regression Model to Predict Heart Disease:")
# Prepare the data
X = df[['Age', 'Sex', 'trestbps', 'Chol', 'thalach', 'exang', 'Oldpeak']]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is the loaded dataset
# Plot the distribution of 'Age' for patients with and without heart disease

plt.figure(figsize=(10, 6))
sns.histplot(df[df['Target'] == 1]['Age'], kde=True, color='red', label='Heart Disease', bins=15)
sns.histplot(df[df['Target'] == 0]['Age'], kde=True, color='green', label='No Heart Disease', bins=15)
plt.legend()
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df[df['Target'] == 1]['Cholesterol'], kde=True, color='red', label='Heart Disease', bins=15)
sns.histplot(df[df['Target'] == 0]['Cholesterol'], kde=True, color='green', label='No Heart Disease', bins=15)
plt.legend()
plt.title('Cholesterol Levels Distribution by Heart Disease Status')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()



