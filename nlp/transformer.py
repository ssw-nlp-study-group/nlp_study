# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/25 3:25 下午
import torch.nn as nn
import torch
import numpy as np
import math
import torch.optim as optim
from util import data, util
import torch.utils.data.dataloader as DataLoader
import matplotlib.pyplot as plt


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


# 解释之前的mask是什么意思。
# 在decoder里面，我们也很想让当前decode layer的每一个位置，能处理上一层decode layer的每一个位置。
# 但为了不发生信息穿越，decode layer做self-attention时，不应该注意到自己之后的位置（因为自己之后的位置此时并没有输出任何东西）。
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # [[[False, False, False, False, True]]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

# 所以我们用mask技术，把上三角遮住了【不给看】
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# 在embedding中加上位置信息 保留单词在句子中的顺序问题
# 另外一个问题，为什么不用自己学习的position embedding呢？
# 作者解释：
# 其一，试了，效果几乎是一样的，那么还不如使用固定的position向量，减少参数。
# 其二，使用正弦曲线可以让模型推断的序列长度大于训练时给定的长度。
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([encoder_layer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        # print(enc_self_attns[0].shape) [1, 8, 5, 5]  8个head 5个单词 对 5个单词求attn
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # print(V.shape) [1, 8, 5, 64]
        # print(scores.shape) [1, 8, 5, 5]
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # print(attn_mask.shape) [1, 8, 5, 5]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        # print(attn.shape) [1, 8, 5, 5]
        # print(V.shape) [1, 8, 5, 64]
        context = torch.matmul(attn, V)
        return context, attn


class multi_head_attention(nn.Module):
    def __init__(self):
        super(multi_head_attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # print(Q.shape) [1, 5, 512]
        # print(attn_mask.shape) [1, 5, 5]
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # print(q_s.shape) [1, 5, 8, 64]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # print(attn_mask) [1, 8, 5, 5]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context [1, 8, 5, 64]
        # attn [1, 8, 5, 5]

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        output = self.layer_norm(output + residual)
        # print(output.shape)  [1, 5, 512]
        return output, attn


# 感知层
class pos_wise_feed_forward_net(nn.Module):
    def __init__(self):
        super(pos_wise_feed_forward_net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        # print(inputs.shape) [1, 5, 512]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class encoder_layer(nn.Module):
    def __init__(self):
        super(encoder_layer, self).__init__()
        self.enc_self_attn = multi_head_attention()
        self.pos_ffn = pos_wise_feed_forward_net()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # print(enc_inputs.shape) [1, 5, 512]
        # print(enc_self_attn_mask.shape) [1, 5, 5]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class decoder_layer(nn.Module):
    def __init__(self):
        super(decoder_layer, self).__init__()
        self.dec_self_attn = multi_head_attention()
        self.dec_enc_attn = multi_head_attention()
        self.pos_ffn = pos_wise_feed_forward_net()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([decoder_layer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor(dec_inputs))
        # print(dec_outputs.shape) [1, 5, 512]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        # 映射到输出
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # print(enc_outputs.shape) [1, 5, 512]
        # print(enc_self_attns[0].shape) [1, 8, 5, 5]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs)
        dec_logits = dec_logits.view(-1, dec_logits.shape[-1])
        # print(dec_logits.shape)   [5, 7]
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    # 我喜欢喝啤酒
    # sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    sentences = ['i like drink beer P', 'S 我 喜欢 喝 啤酒', '我 喜欢 喝 啤酒 E']

    # src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab = {'P': 0, 'i': 1, 'like': 2, 'drink': 3, 'beer': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, '我': 1, '喜欢': 2, '喝': 3, '啤酒': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = transformer()

    criterion = nn.CrossEntropyLoss()
    # lr设置小一点 防止不收敛
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
