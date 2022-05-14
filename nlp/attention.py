# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/12 1:31 下午
import torch.nn as nn
import torch
import numpy as np
import math
import torch.optim as optim
import random
from util import data, util
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from nlp.lstm import lstm_model


class attention_layer(nn.Module):
    def __init__(self):
        super(attention_layer, self).__init__()
        n_features = 256
        out_features = 10
        self.k_w = nn.Linear(n_features, out_features, bias=False)
        self.q_w = nn.Linear(n_features, out_features, bias=False)
        self.v_w = nn.Linear(n_features, out_features, bias=False)

    def forward(self, x, y):
        q_s = self.q_w(y)
        k_s = self.k_w(x)
        v_s = self.v_w(x)

        k_s = k_s.permute(0, 2, 1)
        scores = torch.bmm(q_s, k_s)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v_s)
        return out, attn_weights


if __name__ == '__main__':
    # embbding层长度
    data = np.zeros((64, 20, 256))
    data = torch.tensor(data).float()
    attn = attention_layer()
    attn(data, data)
