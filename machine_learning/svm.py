# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/3/2 2:25 下午
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def plotFeature(dataSet, labelMat, weights, b):
    dataArr = dataSet
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # print(dataArr[0][0])
    # print(int(labelMat[9]))
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
        else:
            xcord2.append(dataArr[i][0])
            ycord2.append(dataArr[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-1, 7.0, 0.2)
    # print(shape(-b - weights[0][0] * x))
    # print(shape(weights[1][0]))

    y = (-b - weights[0][0] * x) / weights[1][0]
    ax.plot(x, y.A[0])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


class struct:
    def __init__(self, dataSet, labels, C, toler):
        self.dataSet = dataSet
        self.labels = labels
        self.C = C
        self.toler = toler
        self.m = shape(dataSet)[0]  # 100
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('a')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def calcEk(i, struct):
    left = multiply(struct.alphas, mat(struct.labels).transpose())
    left = left.T
    right = dot(struct.dataSet, mat(struct.dataSet[i]).T)
    fx = dot(left, right) + struct.b
    ek = fx - struct.labels[i]
    return ek.A[0][0]


def calcWs(struct):
    m, n = shape(struct.dataSet)
    w = zeros((1, n))
    # struct.alphas[0][0] = 1
    # struct.alphas[1][0] = 2
    left = multiply(struct.alphas, mat(struct.labels).transpose())
    # print(left.T)
    # print(struct.dataSet)

    # print(shape(left[0] * struct.dataSet[0]))
    for i in range(struct.m):
        w += left[i] * struct.dataSet[i]
    return w


def updateEk(i, struct):
    eki = calcEk(i, struct)
    struct.eCache[i] = [1, eki]
    return eki


# 随机选择一个不等于i的下标
def selectJrand(i, struct):
    j = i
    while j == i:
        j = int(random.uniform(0, struct.m))
    return j


def selectJ(i, struct):
    eki = updateEk(i, struct)
    cacheList = nonzero(struct.eCache[:, 0].A)[0]

    maxDeltaE = 0
    jIndex = -1
    if (len(cacheList) > 1):
        print(cacheList)
        for k in cacheList:
            if k != i:
                ekj = calcEk(k, struct)
                delta = abs(eki - ekj)
                if (delta > maxDeltaE):
                    maxDeltaE = delta
                    jIndex = k
        return jIndex
    else:
        return selectJrand(i, struct)
    # for j in range(struct.m):
    #     if (i != j):
    #         ekj = calcEk(j, struct)
    #         if (abs(eki - ekj) > maxDeltaE):
    #             maxDeltaE = abs(eki - ekj)
    #             jIndex = j
    #
    # return jIndex


def clipAlpha(alpha, L, H):
    if alpha <= L:
        return L
    if alpha >= H:
        return H
    return alpha


def innerL(i, struct):
    j = selectJ(i, struct)
    eki = updateEk(i, struct)
    ekj = updateEk(j, struct)

    aiold = struct.alphas[i].copy()
    ajold = struct.alphas[j].copy()

    if ((struct.labels[i] * eki < -struct.toler) and (struct.alphas[i] < struct.C)) or (
            (struct.labels[i] * eki > struct.toler) and (struct.alphas[i] > 0)):
        eta = 2.0 * dot(struct.dataSet[i], struct.dataSet[j]) - dot(struct.dataSet[i], struct.dataSet[i]) - dot(
            struct.dataSet[j], struct.dataSet[j])
        if eta >= 0:
            print("eta error,eta = " + eta)
            return 0

        if struct.labels[i] != struct.labels[j]:
            L = max(0, struct.alphas[j] - struct.alphas[i])
            H = min(struct.C, struct.C + struct.alphas[j] - struct.alphas[i])
        else:
            L = max(0, struct.alphas[j] + struct.alphas[i] - struct.C)
            H = min(struct.C, struct.alphas[j] + struct.alphas[i])

        if L == H:
            return 0
        struct.alphas[j] -= struct.labels[j] * (eki - ekj) / eta
        struct.alphas[j] = clipAlpha(struct.alphas[j], L, H)

        updateEk(j, struct)
        if abs(struct.alphas[j] - ajold) < 0.00001:
            # 变化太小
            return 0

        struct.alphas[i] += struct.labels[i] * struct.labels[j] * (ajold - struct.alphas[j])
        updateEk(i, struct)
        b1 = struct.b - eki - struct.labels[i] * (struct.alphas[i] - aiold) * dot(struct.dataSet[i],
                                                                                  struct.dataSet[i]) - \
             struct.labels[j] * (struct.alphas[j] - ajold) * dot(struct.dataSet[i], struct.dataSet[j])
        b2 = struct.b - ekj - struct.labels[i] * (struct.alphas[i] - aiold) * dot(struct.dataSet[i],
                                                                                  struct.dataSet[i]) - \
             struct.labels[j] * (struct.alphas[j] - ajold) * dot(struct.dataSet[i], struct.dataSet[j])
        if (0 < struct.alphas[i]) and (struct.alphas[i] < struct.C):
            struct.b = b1
        elif (0 < struct.alphas[j]) and (struct.alphas[j] < struct.C):
            struct.b = b2
        else:
            struct.b = (b1 + b2) / 2.0
        return 1
    return 0


def smop():
    changed = 0
    entireSet = True
    maxIter = 100
    iter = 0

    while True:
        changed = 0
        if entireSet:
            for i in range(struct.m):
                changed += innerL(i, struct)
            entireSet = False
            print("entire +1")
            if changed == 0:
                break
        else:
            nonBoundsIds = nonzero((struct.alphas.A > 0) * (struct.alphas.A < struct.C))[0]
            for i in nonBoundsIds:
                changed += innerL(i, struct)
            print("non bounds +1")
        if changed == 0:
            entireSet = True

    # print(changed)
    # print(nonBoundsIds)
    # print(struct.alphas)


trainDataSet, labels = loadDataSet("../data/testSet.txt")
struct = struct(trainDataSet, labels, 2, 0.0001)

# print(dot(struct.dataSet[0], struct.dataSet[1]))
smop()

# print(struct.alphas)

ws = calcWs(struct)

# print(ws)
plotFeature(struct.dataSet, mat(struct.labels).transpose(), ws.transpose(), struct.b)

# print(mat([1.2]).Tmat([3,4]))
