import numpy as np
from sklearn.metrics import confusion_matrix as cf


class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_of_iterations=1500):
        self.learning_rate = learning_rate
        self.no_of_iterations = num_of_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y):
        z = np.dot(X, self.theta)
        first = np.multiply(-y, np.log(self.sigmod(z)))
        second = np.multiply((1 - y), np.log(1 - self.sigmod(z)))
        return np.sum(first - second) / (len(X))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.no_of_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict(self, X_test):
        y_pred = []
        for data in X_test.values:
            z = np.dot(data, self.theta)
            h = self.sigmoid(z)
            if h > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)

    def accuracy(self, X_test, Y_test):
        y_pred = self.predict(X_test)
        print(y_pred)
        return (y_pred == Y_test).mean()

    def confusion_matrix(self, X_test, Y_test):
        y_pred = self.predict(X_test)
        return cf(Y_test, y_pred)
