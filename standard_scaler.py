#!/usr/bin/env python3

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Initialize the Support Vector Machine (SVM) classifier
svm_classifier = SVC()

# Train the classifier on the scaled training set
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {test_accuracy * 100:.2f}%")
