# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/12 1:31 下午
import torch.nn as nn
import torch
import numpy as np
import math
import torch.optim as optim
import random
import sys, os
sys.path.append(os.getcwd())
from util import data, util
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from nlp.lstm import lstm_model

# 给lstm加上attention注意力机制
class net(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(net, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        self.emb = nn.Embedding(n_class + 1, n_emb)
        self.lstm = lstm_model(n_emb, self.hidden_sz)
        self.fc = nn.Linear(hidden_sz, n_class)

    # attention网络 个人理解 将attention信息通过训练保存在h_t中 对历史lstm_output进行加权
    def attn(self, lstm_output, h_t):
        # lstm_output [3, 10, 16]  h_t[10, 16] # [D B C]
        h_t = h_t.unsqueeze(0)
        # [10, 16, 1]
        h_t = h_t.permute(1, 2, 0)
        lstm_output = lstm_output.permute(1, 0, 2)

        attn_weights = torch.bmm(lstm_output, h_t)
        attn_weights = attn_weights.permute(1, 0, 2).squeeze()

        # [3, 10]
        attention = F.softmax(attn_weights, 1)
        # bmm: [10, 16, 3] [10, 3, 1]

        attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(-1).transpose(1,0))
        return attn_out.squeeze()

    def forward(self, x):
        # x [10,3]
        x = x.long()
        x = self.emb(x)

        # x 10 3 16
        lstm_output, (h_t, c_t) = self.lstm(x)
        # h_t [10,16]

        # 对lstm_output做attention
        attn_out = self.attn(lstm_output, h_t)

        # 10 26
        lstm_output = self.fc(attn_out)
        # [3]
        return lstm_output


if __name__ == '__main__':
    # embbding层长度
    n_emb = 16

    datas = data.word_seq_datas()

    n_class = datas.dict_len  # number of class(=number of vocab)

    loader = DataLoader.DataLoader(dataset=datas, batch_size=len(datas))
    input_batch, target_batch = iter(loader).next()

    # 转换为float32
    input_batch = input_batch.float()

    # 输入的一条数据长度
    input_sz = n_class
    hidden_sz = 16
    model = net(input_sz, hidden_sz)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(20000):
        optimizer.zero_grad()

        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in datas.seq_data]
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [datas.number_dict[n.item()] for n in predict.squeeze()])
