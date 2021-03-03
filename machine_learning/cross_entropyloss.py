# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/1/23 9:04 下午
import torch
import torch.nn as nn
import numpy as np


# 自定义实现nn.CrossEntropyLoss()

def my_corss_entropy_loss(input, target):
    """
    假设有 A,B两个事件 概率分别为 p,q
    熵:表示信息量的大小 p*logp , q*logq
    交叉熵:plogp 可以用来表示从事件A的角度来看，如何描述事件B 为了保持对称改为 plogq+qlogp
    torch的nn.CrossEntropyLoss是将 nn.LogSoftmax() 和 nn.NLLLoss() 组合在一个类中
    计算公式使用的 p_i*log(exp_i/exp_sum)
    :param input: (3,5) 输入的向量矩阵
    :param target: (3,) 要计算loss的类别
    :return: loss计算
    """
    _exp = np.exp(input)
    _sum = np.sum(_exp, axis=1)

    _len = len(target)

    res = 0
    for i in range(_len):
        first = _exp[i, target[i]]
        res += np.log(first / _sum[i])
    return -res / _len


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)

input = input.detach().numpy()
target = target.detach().numpy()
output = output.detach().numpy()

print("输入为3个5类:")
print(input)
print("要计算loss的类别:")
print(target)
print("nn.CrossEntropyLoss计算loss的结果:")
print(output)
print("自定义函数计算loss结果:")
print(my_corss_entropy_loss(input, target))
