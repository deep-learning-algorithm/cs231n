# -*- coding: utf-8 -*-

# @Time    : 19-7-18 下午9:22
# @Author  : zj


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from nn_classifier import NN
import warnings

warnings.filterwarnings('ignore')


def load_scores(data_path, shuffle=True, tsize=0.8):
    df = pd.read_csv(data_path, header=None, sep=',')
    values = np.array(df.values)
    x = values[:, :2].astype(np.float64)
    y = values[:, 2].astype(np.uint8)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=shuffle, train_size=tsize)

    return x_train, x_test, y_train, y_test


def draw_decision_boundary(classifier, x, y):
    # 编辑网络，预测结果
    x_min, x_max = min(x[:, 0]) - 0.5, max(x[:, 0]) + 0.5
    y_min, y_max = min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    x_grid = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
    # 预测结果
    y_pred = classifier.predict(x_grid).reshape(xx.shape)
    # 绘制等高轮廓
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.cool_r)
    # 绘制测试点
    indexs_0 = np.argwhere(y == 0).squeeze()
    indexs_1 = np.argwhere(y == 1).squeeze()
    plt.scatter(x[indexs_0, 0], x[indexs_0, 1], c='r', marker='<')
    plt.scatter(x[indexs_1, 0], x[indexs_1, 1], c='g', marker='8')
    plt.show()


if __name__ == '__main__':
    scores_path = '/home/zj/data/scores.csv'
    x_train, x_test, y_train, y_test = load_scores(scores_path)

    # 零中心 + 单位方差
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    x_train = (x_train - mu) / np.sqrt(var + eps)
    x_test = (x_test - mu) / np.sqrt(var + eps)

    # 训练分类器
    classifier = NN([100], input_dim=2, num_classes=2, learning_rate=5e-2, reg=1e-3)
    classifier.train(x_train, y_train, num_iters=20000, batch_size=256, verbose=True)

    draw_decision_boundary(classifier, x_test, y_test)
