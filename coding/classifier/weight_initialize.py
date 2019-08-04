# -*- coding: utf-8 -*-

# @Time    : 19-8-4 上午10:07
# @Author  : zj

import numpy as np


def weight_initialize_v1(D, H, weight_scale=1e-2):
    """
    小随机数初始化权重
    :return:
    """
    return weight_scale * np.random.randn(D, H)


def weight_initialize_v2(D, H, weight_scale=1e-2):
    """
    方差校正方式一
    :return:
    """
    return weight_scale * np.random.randn(D, H) / np.sqrt(D)


def weight_initialize_v3(D, H, weight_scale=1e-2):
    """
    方差校正方式二
    :return:
    """
    return weight_scale * np.random.randn(D, H) * np.sqrt(2.0 / D)
