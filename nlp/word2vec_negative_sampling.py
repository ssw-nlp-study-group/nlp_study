# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/1/29 2:13 下午

import torch
import torch.nn as nn
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
# 设置plt显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('../runs/word2vec_skip_gram_negative_sampling')

# 判断是否有GPU
USE_CUDA = torch.cuda.is_available()

# 固定随机种子，以防止多次训练结果不一致
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('../runs/word2vec_skip_gram_negative_sampling')


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


def find_nearest_k(word, k):
    wid = words2id[word]
    w_vec = wordvec[wid]

    similarity = wordvec @ w_vec.T
    sort = np.sort(similarity)[::-1]
    sort_arg = np.argsort(similarity)[::-1]

    result = []
    for i in sort_arg:
        result.append(id2words[i])

    print("与 %s 相似度排序" % word, result)

    return result[:k]


class embedding_model(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(embedding_model, self).__init__()
        self.voc_size = voc_size
        self.emb_size = emb_size

        init_range = 0.5 / self.emb_size
        self.in_embed = nn.Embedding(num_embeddings=self.voc_size, embedding_dim=emb_size)
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed = nn.Embedding(num_embeddings=self.voc_size, embedding_dim=emb_size)
        self.out_embed.weight.data.uniform_(-init_range, init_range)

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
        neg_dot = torch.bmm(neg_embedding, -input_embedding)

        log_pos = torch.sigmoid(pos_dot).sum(1)
        log_neg = torch.sigmoid(neg_dot).sum(1)

        loss = (-log_pos - log_neg).squeeze()
        return loss


def words2id_func(words):
    return np.array([words2id[w] for w in words])


if __name__ == '__main__':
    data_file = "../data/zhihu.txt"
    model_file = "../model/word2vec_negative_sampling.pkl"
    # 设定超参数（hyper parameters）
    # 负采样个数k
    k = 5
    embedding_size = 8
    context_size = 2
    lr = 1e-2
    num_epoch = 800

    # 数据处理
    lines, words2id, id2words, words_set, words, words_freq_num, words_freq_p = \
        get_words(data_file=data_file)

    # 不重复单词个数
    voc_size = len(words_set)

    # 根据文本获取 训练数据 pairs [(input,output),...]
    skip_grams = get_skip_pairs(lines, context_size=2)

    model = embedding_model(voc_size=voc_size, emb_size=embedding_size)

    dataset = word_embedding_dataset(skip_grams=skip_grams)

    indexs = torch.multinomial(torch.Tensor(words_freq_p), k, replacement=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    word_freq_indexs = {}
    for i, (word, freq) in enumerate(words_freq_num):
        word_freq_indexs[word] = i

    # index = 0
    # # 词向量训练部分
    # for j in range(num_epoch):
    #     for i, skip_gram in enumerate(dataset):
    #         center_word = words2id[skip_gram[0]]
    #         context_words_id = [words2id[word] for word in skip_gram[1]]
    #         context_words = skip_gram[1]
    #         p = words_freq_p.copy()
    #
    #         for context in context_words:
    #             p[word_freq_indexs[context]] = 0
    #
    #         neg_words_sample = torch.multinomial(torch.Tensor(p), k, replacement=True)
    #         neg_words = words2id_func(words_freq_num[neg_words_sample.numpy()][:, 0])
    #         context_words = words2id_func(context_words)
    #
    #         optimizer.zero_grad()
    #
    #         loss = model(torch.LongTensor([center_word]), torch.LongTensor(context_words), torch.LongTensor(neg_words))
    #         if index % 1000 == 0:
    #             print(loss.item())
    #
    #         loss.backward()
    #         optimizer.step()
    #         writer.add_scalar("loss", loss.item(), index)
    #         index += 1
    #
    # # 保存模型
    # torch.save(model, model_file)

    model = torch.load(model_file)

    for name, param in model.named_parameters():
        # 获取in_embed的参数作为词向量 out_embed可舍弃
        if name == "in_embed.weight":
            wordvec = param.data.numpy()

    pca = PCA(n_components=2)
    result = pca.fit_transform(wordvec)
    plt.scatter(result[:, 0], result[:, 1])
    print(id2words)
    print(words_set)

    for i in range(voc_size):
        plt.annotate(id2words[i], xy=(result[i, 0], result[i, 1]))

    plt.xlabel('奇数')
    plt.ylabel('偶数')
    plt.show()

    # 获取最相近的k个词向量
    result = find_nearest_k('什么', k=4)
    print(result)
