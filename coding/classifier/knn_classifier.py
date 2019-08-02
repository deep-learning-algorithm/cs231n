# -*- coding: utf-8 -*-

# @Time    : 19-7-11 下午8:02
# @Author  : zj

from builtins import range
from builtins import object
import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_iris(iris_path, shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(iris_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return x_train, x_test, y_train, y_test


def load_german_data(data_path, shuffle=True, tsize=0.8):
    data_list = pd.read_csv(data_path, header=None, sep='\s+')

    data_array = data_list.values
    height, width = data_array.shape[:2]
    data_x = data_array[:, :(width - 1)]
    data_y = data_array[:, (width - 1)]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=shuffle)

    y_train = np.array(list(map(lambda x: 1 if x == 2 else 0, y_train)))
    y_test = np.array(list(map(lambda x: 1 if x == 2 else 0, y_test)))

    return x_train, x_test, y_train, y_test


class KNN(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for tests data using this classifier.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing tests data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          tests data, where y[i] is the predicted label for the tests point X[i].
        """
        dists = self._compute_distances(X)

        return self._predict_labels(dists, k=k)

    def _compute_distances(self, X):
        """
        Compute the distance between each tests point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        """
        temp_test = np.atleast_2d(np.sum(X ** 2, axis=1)).T
        temp_train = np.atleast_2d(np.sum(self.X_train ** 2, axis=1))
        temp_test_train = -2 * X.dot(self.X_train.T)

        dists = np.sqrt(temp_test + temp_train + temp_test_train)
        return dists

    def _predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between tests points and training points,
        predict a label for each tests point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith tests point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          tests data, where y[i] is the predicted label for the tests point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            idxes = np.argsort(dists[i])

            y_sorted = self.y_train[idxes]
            closest_y = list(y_sorted[:k])

            nums = np.array([closest_y.count(m) for m in closest_y])
            y_pred[i] = closest_y[np.argmax(nums)]

        return y_pred


def compute_accuracy(y, y_pred):
    num = y.shape[0]
    num_correct = np.sum(y_pred == y)
    acc = float(num_correct) / num
    return acc


def cross_validation(x_train, y_train, k_choices, num_folds=5, Classifier=KNN):
    X_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # 计算预测标签和验证集标签的精度
    k_to_accuracies = {}
    for k in k_choices:
        k_accuracies = []
        # 随机选取其中一份为验证集，其余为测试集
        for i in range(num_folds):
            x_folds = X_train_folds.copy()
            y_folds = y_train_folds.copy()

            x_vals = x_folds.pop(i)
            x_trains = np.vstack(x_folds)

            y_vals = y_folds.pop(i)
            y_trains = np.hstack(y_folds)

            classifier = Classifier()
            classifier.train(x_trains, y_trains)

            y_val_pred = classifier.predict(x_vals, k=k)
            k_accuracies.append(compute_accuracy(y_vals, y_val_pred))
        k_to_accuracies[k] = k_accuracies

    return k_to_accuracies


def plot(k_choices, k_to_accuracies):
    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


if __name__ == '__main__':
    # iris_path = '/home/zj/data/iris-species/Iris.csv'
    # x_train, x_test, y_train, y_test = load_iris(iris_path, shuffle=True, tsize=0.8)

    data_path = '/home/zj/data/german/german.data-numeric'
    x_train, x_test, y_train, y_test = load_german_data(data_path, shuffle=True, tsize=0.8)

    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100]
    k_to_accuracies = cross_validation(x_train, y_train, k_choices)

    # print(k_to_accuracies)
    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    plot(k_choices, k_to_accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    k = k_choices[np.argmax(accuracies_mean)]
    print('最好的k值是：%d' % k)

    # 测试集测试
    classifier = KNN()
    classifier.train(x_train, y_train)

    y_test_pred = classifier.predict(x_test, k=k)
    y_test_acc = compute_accuracy(y_test, y_test_pred)
    print('测试集精度为：%f' % y_test_acc)
