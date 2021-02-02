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

# 判断是否有GPU
USE_CUDA = torch.cuda.is_available()

# 固定随机种子，以防止多次训练结果不一致
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)


def get_words(data_file):
    lines = []
    with open(data_file, encoding="UTF-8") as file:
        for line in file:
            lines.append(line.strip())

    sentence = " ".join(lines)
    words = sentence.split(" ")
    words_set = list(set(words))

    words_freq_num = Counter(words).most_common(len(words_set))
    words_freq_p = [freq for word, freq in words_freq_num]
    words_freq_p = np.array(words_freq_p)
    words_freq_p = words_freq_p ** (3. / 4.)
    words_freq_p = words_freq_p / np.sum(words_freq_p)

    words2id = {w: i for i, w in enumerate(words_set)}
    id2words = {i: w for i, w in enumerate(words_set)}

    return lines, words2id, id2words, words_set, words, np.array(words_freq_num), np.array(words_freq_p)


def get_skip_pairs(lines, context_size):
    skip_grams = []
    for line in lines:
        words = line.split(" ")
        for i, w in enumerate(words):
            context_words = words[max(i - context_size, 0):max(i, 0)] + words[i + 1:i + context_size + 1]
            skip_grams.append((w, context_words))
    return skip_grams


class word_embedding_dataset(tud.Dataset):
    def __init__(self, skip_grams):
        super(word_embedding_dataset, self).__init__()
        self.skip_grams = skip_grams

    def __len__(self):
        return len(self.skip_grams)

    def __getitem__(self, idx):
        return self.skip_grams[idx]


class embedding_model(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(embedding_model, self).__init__()
        self.voc_size = voc_size
        self.emb_size = emb_size
        initrange = 0.5 / self.emb_size

        self.in_embed = nn.Embedding(self.voc_size, self.emb_size)
        self.out_embed = nn.Embedding(self.voc_size, self.emb_size)

        # self.out_embed.weight.data.uniform_(-initrange, initrange)
        # self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        # shape (1,embedding_size)
        input_embedding = self.in_embed(input_labels)
        # shape (context_size,embedding_size)
        pos_embedding = self.out_embed(pos_labels)
        neg_embedding = self.out_embed(neg_labels)

        input_embedding = input_embedding.unsqueeze(2)
        pos_embedding = pos_embedding.unsqueeze(0)
        neg_embedding = neg_embedding.unsqueeze(0)

        pos_dot = torch.bmm(pos_embedding, input_embedding)
        neg_dot = torch.bmm(neg_embedding, input_embedding)

        log_pos = torch.sigmoid(pos_dot).sum(1)
        print(log_pos)
        log_neg = torch.sigmoid(neg_dot).sum(1)

        print(log_neg)
        # print(log_neg)

        # print(pos_dot)
        # print(neg_dot)

        # print(input_embedding.shape)
        # print(pos_embedding.shape)
        # print(neg_embedding.shape)
        # print(input_labels)


def words2id_func(words):
    return np.array([words2id[w] for w in words])


if __name__ == '__main__':
    # 设定超参数（hyper parameters）
    data_file = "../data/zhihu.txt"
    # 负采样个数k
    k = 20
    embedding_size = 8
    context_size = 2
    lines, words2id, id2words, words_set, words, words_freq_num, words_freq_p = \
        get_words(data_file=data_file)

    voc_size = len(words_set)
    # print(Counter(words).most_common(4))

    skip_grams = get_skip_pairs([lines[0]], context_size=2)

    model = embedding_model(voc_size=voc_size, emb_size=embedding_size)

    dataset = word_embedding_dataset(skip_grams=skip_grams)

    indexs = torch.multinomial(torch.Tensor(words_freq_p), k, replacement=True)

    # print(dict(words_freq_num))

    word_freq_indexs = {}
    for i, (word, freq) in enumerate(words_freq_num):
        word_freq_indexs[word] = i

    # print(word_freq_indexs)
    # print(np.array(words_freq_num)[indexs.numpy()])
    # print(indexs.numpy())

    for i, skip_gram in enumerate(dataset):
        center_word = words2id[skip_gram[0]]
        context_words_id = [words2id[word] for word in skip_gram[1]]
        context_words = skip_gram[1]
        p = words_freq_p[:]

        for context in context_words:
            p[word_freq_indexs[context]] = 0

        neg_words_sample = torch.multinomial(torch.Tensor(p), k, replacement=True)
        # print(words_freq_num)
        # neg_words = np.array([words2id[word] for word in words_freq_num[neg_words_sample.numpy()][:, 0]])
        neg_words = words2id_func(words_freq_num[neg_words_sample.numpy()][:, 0])
        context_words = words2id_func(context_words)
        # print(center_word)
        # print(neg_words)
        # print(context_words)

        model(torch.LongTensor([center_word]), torch.LongTensor(context_words), torch.LongTensor(neg_words))
        # model(torch.LongTensor([23]), torch.LongTensor(neg_words), torch.LongTensor(context_words))
        break
        # print(neg_words)
        # print(context_words)
        # print(center_word, context_words)
