# -*- ecoding: utf-8 -*-
# @Author: anyang
# @Time: 2021/2/22 1:15 上午
import util
import numpy as np
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader


class word_seq_datas(Dataset.Dataset):
    def __init__(self):
        super(word_seq_datas, self).__init__()
        self.seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']
        char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']

        self.word_dict = {n: i for i, n in enumerate(char_arr)}
        self.number_dict = {i: w for i, w in enumerate(char_arr)}
        self.dict_len = len(self.word_dict)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        word = self.seq_data[index]
        # item = self.word_dict[word[0]]
        item = []
        for i in word[: -1]:
            # item.append(np.eye(self.dict_len)[self.word_dict[i]])
            item.append(self.word_dict[i])
        label = self.word_dict[word[-1]]
        return np.array(item), label


if __name__ == '__main__':
    datas = word_seq_datas()
    loader = DataLoader.DataLoader(dataset=datas, batch_size=10)
    # for i, (item, label) in enumerate(loader):
    #     print(item, '-------', label)

    print(iter(loader).next())