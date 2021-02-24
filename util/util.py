# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/21 9:49 下午
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置plt显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']

# 固定随机种子，以防止多次训练结果不一致
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)