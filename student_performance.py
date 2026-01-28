# Student Performance Prediction using Random Forest
# Author: Md Zameer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/content/data.csv")

#top 5 rows printing
print("First 5 rows:\n", data.head())
print("\nData Info:\n")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())

#  Feature selection & target
X = data[['hours_study', 'attendance', 'assignments_submitted', 'previous_score']]
y = data['pass_fail']  # 1 = Pass, 0 = Fail

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

 #4. Feature scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, min_samples_split=5)
model.fit(X_train_scaled, y_train)

#  Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))


corr = data.corr()['pass_fail'].drop('pass_fail')  # exclude target itself

# Plot correlation as feature importance
plt.figure(figsize=(6,4))
sns.barplot(x=corr.values, y=corr.index, palette="viridis")
plt.title("Feature Importance (Correlation with Target)")
plt.xlabel("Correlation with Pass/Fail")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


