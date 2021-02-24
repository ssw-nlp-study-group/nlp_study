# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/21 10:30 上午

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.optim as optim
import random
from util import data, util
import torch.utils.data.dataloader as DataLoader


class single_head_attention_layer(nn.Module):

    def __init__(self, n_hidden: int):
        super(single_head_attention_layer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embdding_dim)

        self.w_q = torch.zeros(n_hidden, vocab_size)
        self.w_k = torch.zeros(n_hidden, vocab_size)
        self.w_v = torch.zeros(n_hidden, vocab_size)

    def forward(self, x):
        # [batch_size, len_seq, embedding_dim]
        x = self.embedding(x)
        # [10,3,8]

        for i in range(len(x)):
            input = x[i]
            # input [3,8]


        # input = input.transpose(1, 0)

        # # q [8 4]
        # q = self.w_q @ input
        # k = self.w_k @ input
        #
        # # score [4]
        # score = torch.sum(q * k, axis=0)
        #
        # score = torch.sqrt(score)
        #
        # out = F.softmax(score, dim=0)
        # print(out)
        pass


if __name__ == '__main__':
    batch_size = 10
    seq_num = 4
    vocab_size = 26
    n_hidden = 8
    embdding_dim = 8

    datas = data.word_seq_datas()

    n_class = datas.dict_len  # number of class(=number of vocab)

    loader = DataLoader.DataLoader(dataset=datas, batch_size=len(datas))
    input_batch, target_batch = iter(loader).next()

    model = single_head_attention_layer(n_hidden=n_hidden)
    # input = torch.ones(batch_size, seq_num, vocab_size)

    model(input_batch)
