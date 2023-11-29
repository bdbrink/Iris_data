#!/usr/bin/env python3

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
tree_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = tree_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# Feature Importance
feature_importance = tree_classifier.feature_importances_

# Display feature importance scores
print("\nFeature Importance:")
for i, feature in enumerate(iris.feature_names):
    print(f"{feature}: {feature_importance[i]:.4f}")
