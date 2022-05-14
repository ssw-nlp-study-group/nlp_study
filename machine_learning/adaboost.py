# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/3/2 2:25 下午
import numpy as np
from numpy import *
import util.util
import matplotlib.pyplot as plt

from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import *


def buildGx(data, threshIneq, threshVal, dim):
    result = np.ones((m, 1))
    if threshIneq == 'gt':
        result[data[:, dim] <= threshVal] = -1.0
    elif threshIneq == 'lt':
        result[data[:, dim] > threshVal] = -1.0
    return result


def loadDataSet():  # 加载测试数据
    dataMat = np.array([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [1.1, 1.2],
                        [2., 1.]])

    labelList = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    return dataMat, labelList  # 数据集返回的是矩阵类型，标签返回的是列表类型

# 预测函数g(x)
class Gx():
    def __init__(self):
        self.inequal = None
        # 维度
        self.dim = None
        # 阀值
        self.threshval = None
        # 预测数值
        self.predict = None
        # 误差(和权重的乘积)
        self.error = None
        self.am = None
        self.gx_eval = None

    def __str__(self):
        return str(self.threshval) + " " + str(self.gx_eval)


if __name__ == '__main__':
    x, y = loadDataSet()
    m, n = x.shape
    wm = np.ones((m, 1)) / m
    num_step = 20
    num_iter = 10

    gxs = []

    for i in range(num_iter):
        bestgx = Gx()
        min_error = inf
        for dim in range(n):
            range_min = x[:, dim].min()
            range_max = x[:, dim].max()
            threshvals = np.arange(range_min - 0.5, range_max + 0.5, (range_max - range_min) / num_step)
            for threshval in threshvals:
                for inequal in ['lt', 'gt']:
                    predict = buildGx(x, inequal, threshval, dim=dim)
                    _filter = np.array([predict[:, 0] != y]).astype(np.int)
                    gx_eval = (_filter @ wm)[0, 0]
                    if gx_eval < min_error:
                        min_error = gx_eval
                        bestgx.inequal = inequal
                        bestgx.dim = dim
                        bestgx.threshval = threshval
                        bestgx.predict = predict
                        bestgx.error = min_error
                        bestgx.gx_eval = gx_eval

        em = bestgx.error
        am = 0.5 * np.log((1 - em) / em)

        wm = wm.T * np.exp(-am * np.array([y]) * bestgx.predict.T)
        wm = wm / (np.sum(wm, axis=1)[0])
        wm = wm.T
        bestgx.am = am
        gxs.append(bestgx)

    y_compounds = np.zeros((1, m))

    for gx in gxs:
        y_compounds += gx.predict.T * gx.am

    print("预测结果", sign(y_compounds)[0])
    print("实际结果", y)
