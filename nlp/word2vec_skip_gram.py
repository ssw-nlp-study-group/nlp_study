# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/1/20 9:57 下午
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])
        random_labels.append(skip_grams[i][1])

    return random_inputs, random_labels


class word2vec(nn.Module):
    def __init__(self):
        super(word2vec, self).__init__()
        self.W = nn.Linear(in_features=voc_size, out_features=embedding_size, bias=False)
        self.WT = nn.Linear(in_features=embedding_size, out_features=voc_size, bias=False)

    def forward(self, x):
        hidden_layer = self.W(x)
        output_layer = self.WT(hidden_layer)
        return output_layer


if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    word_seq = " ".join(sentences)
    word_seq_list = word_seq.split(" ")
    word_list = word_seq.split(" ")

    words = list(set(word_list))
    voc_size = len(words)

    words2id = {w: i for i, w in enumerate(words)}
    id2words = {i: w for i, w in enumerate(words)}

    skip_grams = []
    for i in range(1, len(word_seq_list) - 1):
        target = words2id[word_seq_list[i]]
        context = [words2id[word_seq_list[i - 1]], words2id[word_seq_list[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = word2vec()

    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for e in range(1000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        opt.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)

        if (e + 1) % 100 == 0:
            print("epoch:", '%04d' % (e + 1), 'cost:', '{:.6f}'.format(loss))

        loss.backward()
        opt.step()

    for i, label in enumerate(words):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
