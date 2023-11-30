#!/usr/bin/env python3

# Iterate over dataset to estimate models performance #

from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
import numpy as np

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1)

# Perform k-fold cross-validation (k=5)
cv_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Print the average cross-validation score
average_cv_score = np.mean(cv_scores)
print("Average Cross-Validation Score:", average_cv_score)

# Train the classifier on the full training set
svm_classifier.fit(X_train, y_train)

# Evaluate the accuracy on the test set
test_accuracy = svm_classifier.score(X_test, y_test)
print(f"Accuracy on Test Set: {test_accuracy * 100:.2f}%")
