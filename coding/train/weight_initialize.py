# -*- coding: utf-8 -*-

# @Time    : 19-8-3 下午8:51
# @Author  : zj

from datas.cifar import get_CIFAR10_data
from classifier.nn_classifier import NN
import matplotlib.pyplot as plt

cifar_path = '/home/zj/data/cifar-10-batches-py'


def plot(results):
    for k, v in results.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_dict = get_CIFAR10_data(cifar_path, val_size=0.1)

    x_val = data_dict['X_val']
    y_val = data_dict['y_val']

    x_val_flatten = x_val.reshape(x_val.shape[0], -1)

    D = 32 * 32 * 3
    H1 = 400
    H2 = 200
    O = 10

    weight_init_list = ['v1', 'v2', 'v3']
    loss_dict = {}
    for item in weight_init_list:
        classifier = NN([H1, H2], input_dim=D, num_classes=O, learning_rate=1e-3, weight_init=item)
        loss_history = classifier.train(x_val_flatten, y_val, num_iters=50, verbose=True)
        loss_dict[item] = loss_history
    plot(loss_dict)
