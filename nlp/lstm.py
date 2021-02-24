# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/12 1:31 下午
import torch.nn as nn
import torch
import numpy as np
import math
import torch.optim as optim
from util import data,util
import torch.utils.data.dataloader as DataLoader

# 暂未实现双向lstm  双向lstm适用于长文本 保存上下文信息
class lstm_model(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super(lstm_model, self).__init__()

        # input_sz 输入的数组长度
        self.input_sz = input_sz
        # hidden_sz cell_state和hidden_state长度
        self.hidden_sz = hidden_sz

        # i_t
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
            i_t = torch.sigmoid(input @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(input @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.tanh(input @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(input @ self.w_o + h_t @ self.u_o + self.b_o)
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
        self.lstm = lstm_model(n_emb, self.hidden_sz)
        self.fc = nn.Linear(hidden_sz, n_class)

    def forward(self, x):
        x = x.long()
        x = self.emb(x)

        # 10 3 26
        output, (_, _) = self.lstm(x)

        # 10 16
        output = output[-1]
        output = self.fc(output)
        # [3]
        return output


if __name__ == '__main__':
    n_emb = 16

    datas = data.word_seq_datas()

    n_class = datas.dict_len  # number of class(=number of vocab)

    loader = DataLoader.DataLoader(dataset=datas, batch_size=len(datas))
    input_batch, target_batch = iter(loader).next()

    # 转换为float32
    input_batch = input_batch.float()

    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # 输入的一条数据长度
    input_sz = n_class
    hidden_sz = 16
    model = net(input_sz, hidden_sz)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(1000):
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
