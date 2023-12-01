#!/usr/bin/env python3

# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine (SVM) classifier
svm_classifier = SVC()

# Define a parameter grid to search
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the classifier with the best hyperparameters
best_svm_classifier = SVC(**best_params)
best_svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_svm_classifier.predict(X_test)

# Evaluate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {test_accuracy * 100:.2f}%")
