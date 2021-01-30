# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/1/29 2:13 下午

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

#判断是否有GPU
USE_CUDA = torch.cuda.is_available()

#固定随机种子，以防止多次训练结果不一致
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)

#设定超参数（hyper parameters）
C= 3 #周围单词个数（context window）
K = 100 #下采样（number of negative samoles）
NUM_EPOCHS =  2 #迭代次数
MAX_VOCAB_SIZE = 30000 #训练词向量的单词数
BATCH_SIZE = 128 #批样本数
LEARNING_RATE = 0.2 #学习率
EMBEDDING_SIZE = 100 #词向量长度
LOG_FILE = "word-embedding.log"


print(USE_CUDA)