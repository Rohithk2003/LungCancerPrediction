import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.tree = {}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.parent_entropy = self.entropy(self.y_train)

    def entropy(self, y):
        unique_classes = y.value_counts().to_dict()
        entropy_ = 0
        for class_ in unique_classes:
            prob = unique_classes[class_] / len(y)
            entropy_ += prob * np.log2(prob)
        return -entropy_

    def total_gain_feature(self, feature):
        distinct_values = self.X_train[feature].value_counts().to_dict()
        entropies_children = {}
        probabilities_children = {}
        for class_ in distinct_values:
            prob = distinct_values[class_] / len(self.X_train[feature])
            probabilities_children[class_] = prob.__round__(2)
            indexes = self.X_train[self.X_train[feature] == class_].index
            entropies_children[class_] = self.entropy(self.y_train[indexes]).__round__(2)
        return self.parent_entropy - (
            np.sum(np.dot(list(probabilities_children.values()), list(entropies_children.values())))).__round__(3)
