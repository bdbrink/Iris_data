import unittest


class IrisClassificationTest(unittest.TestCase):
    def test_data_loading(self):
        """Test if Iris data is loaded correctly."""
        from sklearn import datasets
        iris = datasets.load_iris()
        self.assertEqual(iris.data.shape, (150, 4))
        self.assertEqual(iris.target.shape, (150,))

    def test_data_splitting(self):
        """Test if data is split correctly into training and testing sets."""
        from sklearn.model_selection import train_test_split
        from sklearn import datasets
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.assertEqual(X_train.shape[0], 120)
        self.assertEqual(X_test.shape[0], 30)

    def test_knn_training(self):
        """Test if KNN model is trained successfully."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)
        self.assertIsInstance(knn_classifier, KNeighborsClassifier)

    def test_knn_prediction(self):
        """Test if KNN model predicts correctly."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))
