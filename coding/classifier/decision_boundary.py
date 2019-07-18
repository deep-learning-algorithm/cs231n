# -*- coding: utf-8 -*-

# @Time    : 19-7-16 上午9:52
# @Author  : zj

import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from nn_classifier import NN


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
    data_y = np.array(data['Species'])

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return x_train, x_test, y_train, y_test


def pca(X, ratio=0.99, **kwargs):
    """
    pca降维
    :param X: 大小为NxM，其中M是个数，N是维度，每个字段已是零均值
    :param ratio: 表示投影均方误差和方差比值，默认为0.99,保持99%的方差
    :param kwargs: 字典参数，如果指定了k值，则直接计算
    :return: 降维后数据
    """
    N, M = X.shape[:2]
    C = X.dot(X.T) / M
    u, s, v = np.linalg.svd(C)

    k = 1
    if 'k' in kwargs:
        k = kwargs['k']
    else:
        while k < N:
            s_k = np.sum(s[:k])
            s_N = np.sum(s)
            if (s_k * 1.0 / s_N) >= ratio:
                break
            k += 1
    p = u.transpose()[:k]
    y = p.dot(X)

    return y, p


if __name__ == '__main__':
    iris_path = '/home/zj/data/iris-species/Iris.csv'
    x_train, x_test, y_train, y_test = load_iris(iris_path)

    # 零中心
    mu = np.mean(x_train, axis=0)
    x_train = x_train - mu
    # 训练分类器
    classifier = NN([20, 20], input_dim=4, num_classes=3, learning_rate=5e-2, reg=1e-3)
    classifier.train(x_train, y_train, num_iters=5000, batch_size=120, verbose=True)
    # PCA降维
    y, p = pca(x_test, k=2)
    # 编辑网络，预测结果
    x_min, x_max = min(p[0]) - 0.05, max(p[0]) + 0.05
    y_min, y_max = p[1].min() - 0.05, p[1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))
    x_grid = np.vstack((xx.reshape(-1), yy.reshape(-1))).T.dot(y)
    x_grid = x_grid - mu
    y_pred = classifier.predict(x_grid).reshape(xx.shape)
    # 绘制等高轮廓
    plt.contourf(xx, yy, y_pred, cmap=mpl.cm.jet)
    # 绘制测试点
    indexs_0 = np.argwhere(y_test == 0).squeeze()
    indexs_1 = np.argwhere(y_test == 1).squeeze()
    indexs_2 = np.argwhere(y_test == 2).squeeze()
    plt.scatter(p[0, indexs_0], p[1, indexs_0], c='r', marker='<')
    plt.scatter(p[0, indexs_1], p[1, indexs_1], c='g', marker='8')
    plt.scatter(p[0, indexs_2], p[1, indexs_2], c='y', marker='*')
    plt.show()
