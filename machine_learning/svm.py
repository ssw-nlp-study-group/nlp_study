# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/3/2 2:25 下午
import numpy as np
from numpy import *
import util.util
import matplotlib.pyplot as plt


# 参考知乎svm推导,最简单的svm实现
# 公式推导细节可以自己手推几遍帮助理解,不了解kkt的先学习一下kkt条件
# 计算公式参考: 机器学习算法实践-SVM中的SMO算法 - 邵正将的文章 - 知乎
# https://zhuanlan.zhihu.com/p/29212107
# 由于缓存e_i等问题属于工程问题，所以为了保证代码简洁(偷懒)，暂时不加上了
# smo本来i选择违背kkt条件程度最大的，这里直接暴力选取了，因为每次选取迭代一对alpha也可以达到效果，只是前者效率更高

# svm步骤
# 1. 选取一对a_i,a_j
# 2. 计算a_i_new,a_j_new
# 3. 对a进行clip(必须满足kkt条件 0<=a<=C)
# 4. 根据新的a计算b(计算公式也是kkt条件之一),同样考虑kkt条件a在边界0 or c具有不同的公式，若在边界内部，则b1=b2
# 5. 收敛之后计算w

# 随便网上抄的一段生成数据代码
def generate_dataset():
    class1, class2 = [], []
    label1, label2 = [], []

    def f(x):
        return 1 * x

    for _ in range(100):
        x = np.random.rand() * 10
        y = np.random.rand() * 10
        if y - f(x) > 1:
            class1.append([x, y])
            label1.append(1)
        elif y - f(x) < -1:
            class2.append([x, y])
            label2.append(-1)
    return np.array(class1 + class2), np.array(label1 + label2)


# 特别简单:计算不等于i的j
def selectj(i, n):
    j = i
    while (j == i):
        j = np.random.randint(0, n)
    return j


# 计算k_ij
def calc_k(i, j):
    return x[i] @ x[j]


def clip_alpha(a, L, H):
    if a < L:
        return L
    if a > H:
        return H
    return a


def calc_w():
    # # 根据公式计算w
    w = np.sum(alphas * y[:, np.newaxis] * x, axis=0)
    return w


# 为了效率可以将e_i缓存,但本代码仅仅是个demo，所以暂时不做处理
def calc_e(i):
    w = calc_w()
    fx_i = x[i] @ w + b
    e_i = fx_i - y[i]
    return e_i


if __name__ == '__main__':
    x, y = generate_dataset()

    n, width = x.shape

    alphas = np.zeros((n, 1))
    b = 0
    c = 2

    # 迭代次数，可以写成while循环直至收敛
    iters = 200

    for iter in range(iters):
        for i in range(n):
            j = selectj(i, n)
            k_ij = calc_k(i, j)
            k_ii = calc_k(i, i)
            k_jj = calc_k(j, j)

            # k11+k22-2*k12
            eta = k_ii + k_jj - 2 * k_ij

            if eta <= 0:
                print("eta <= 0, continue")
                continue

            e_i = calc_e(i)
            e_j = calc_e(j)

            alphas_j_new = alphas[j] + y[j] * (e_i - e_j) / eta

            # 分两种情况讨论y_i==y_j or y_i!=y_j 使得最后的 0<=a<=c
            if not y[i] == y[j]:
                L = np.max([0, alphas[j] - alphas[i]])
                H = np.min([c, c + alphas[j] - alphas[i]])

            if y[i] == y[j]:
                L = np.max([0, alphas[i] + alphas[j] - c])
                H = np.min([c, alphas[j] + alphas[i]])

            if L >= H:
                continue

            alphas_j_old = alphas[j]
            alphas_j_new = clip_alpha(alphas_j_new, L, H)
            alphas_i_old = alphas[i]
            alphas_i_new = alphas[i] + y[i] * y[j] * (alphas[j] - alphas_j_new)

            # 更新计算的alpha
            alphas[i] = alphas_i_new
            alphas[j] = alphas_j_new

            b1 = -e_i - y[i] * k_ii * (alphas_i_new - alphas_i_old) - y[j] * k_ij * (alphas_j_new - alphas_j_old) + b
            b2 = -e_j - y[i] * k_ij * (alphas_i_new - alphas_i_old) - y[j] * k_jj * (alphas_j_new - alphas_j_old) + b

            # 依据是 0<a<c时候 根据kkt条件 y_i*(w*+b)=1 就可以直接求出b
            # 当a==0 或者 a==c的时候是不等式 只能是区间
            if 0 < alphas[j] < c:
                b = b2
            elif 0 < alphas[i] < c:
                b = b1
            else:
                b = (b1 + b2) / 2

    w = calc_w()

    # 将结果画图展示
    x_arr = np.linspace(-2, 12, 20)
    y_arr = -w[0] * x_arr / w[1] - b / w[1]
    plt.plot(x_arr, y_arr, c='b')
    x_1 = x[y == 1]
    plt.scatter(x_1[:, 0], x_1[:, 1], c='r')
    x_2 = x[y == -1]
    plt.scatter(x_2[:, 0], x_2[:, 1], c='g')

    for i in range(len(x)):
        # alphas[i] 超过一定数值就可以认为是边界
        if alphas[i] > 1e-3:
            xi_1, xi_2 = x[i, 0], x[i, 1]
            plt.scatter(xi_1, xi_2, s=150, c='none', linewidths=1.5, edgecolors='#1f77b4')
            x_bound = np.linspace(-2, 10, 50)
            y = -w[0] / w[1] * x_bound + (xi_2 + w[0] / w[1] * xi_1)
            plt.plot(x_bound, y, 'y--')

    plt.show()
