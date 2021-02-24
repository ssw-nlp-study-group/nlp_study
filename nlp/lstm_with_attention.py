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


# 给lstm加上attention
# 参考 https://www.bilibili.com/video/BV1qZ4y1H7Pg
class lstm_model_attention(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super(lstm_model_attention, self).__init__()

        # input_sz 输入的数组长度
        self.input_sz = input_sz
        # hidden_sz cell_state和hidden_state长度
        self.hidden_sz = hidden_sz

        # i_t 遗忘
        self.w_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.u_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.w_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.u_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.w_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.u_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.w_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.u_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x (batch_size批数据长度,序列长度seq_sz,单条数据长度,data_len)
        bs, seq_sz = x.shape[0], x.shape[1]

        # hidden_state和cell_state 用来保存短期和长期记忆
        h_t = torch.zeros(bs, self.hidden_sz)
        c_t = torch.zeros(bs, self.hidden_sz)

        hidden_seq = []
        # 原始的lstm实现 参考pytorch官网的lstm公式
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        for i in range(seq_sz):
            input = x[:, i, :]
            f_t = torch.sigmoid(input @ self.w_f + h_t @ self.u_f + self.b_f)
            i_t = torch.sigmoid(input @ self.w_i + h_t @ self.u_i + self.b_i)
            o_t = torch.sigmoid(input @ self.w_o + h_t @ self.u_o + self.b_o)
            g_t = torch.tanh(input @ self.w_c + h_t @ self.u_c + self.b_c)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        lstm_output = torch.cat(hidden_seq, dim=0)
        lstm_output = lstm_output.view(-1, hidden_seq[0].shape[0], hidden_seq[0].shape[1])

        return lstm_output, (h_t, c_t)


class net(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(net, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        self.emb = nn.Embedding(n_class + 1, n_emb)
        self.lstm = lstm_model_attention(n_emb, self.hidden_sz)
        self.fc = nn.Linear(hidden_sz, n_class)

    # attention网络 个人理解 将attention信息通过训练保存在h_t中 对历史lstm_output进行加权
    def attn(self, lstm_output, h_t):
        # lstm_output [3, 10, 16]  h_t[10, 16]
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

        # 10 16
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
