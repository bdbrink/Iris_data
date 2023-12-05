#!/usr/bin/env python3

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize PCA with 2 components (for visualization purposes)
pca = PCA(n_components=2)

# Fit and transform the PCA on the training data
X_train_pca = pca.fit_transform(X_train)

# Transform the test data using the same PCA
X_test_pca = pca.transform(X_test)

# Initialize the Support Vector Machine (SVM) classifier
svm_classifier = SVC()

# Train the classifier on the reduced-dimensional training set
svm_classifier.fit(X_train_pca, y_train)

# Make predictions on the reduced-dimensional test set
y_pred = svm_classifier.predict(X_test_pca)

# Evaluate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {test_accuracy * 100:.2f}%")

# Visualize the reduced-dimensional data
plt.figure(figsize=(8, 6))
for i, c in zip(range(3), ['red', 'green', 'blue']):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], c=c, label=iris.target_names[i])

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
