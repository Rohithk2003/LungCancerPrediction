import numpy as np


class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []
        for x in X_test.values:
            distances = [
                np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train.values
            ]
            k_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
        return np.array(y_pred)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
