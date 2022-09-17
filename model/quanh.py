import torch
import torch.nn as nn
import pickle
import numpy as np
import math
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from dataloader import CustomDataset
config = {'data_dir': 'pickle/',
          'l_win': 120,
          'autoencoder_dims': 500,
          'num_epoch': 10}

with np.load('pickle/buzz1_test_data.npz', 'rb') as f:
    for i in f:
        print(f[i].shape)
        for j in f[i]:
            print(len(j))

# print(data.shape, label.shape)
# train_set = CustomDataset(config)
# train_dataloader = DataLoader(train_set, 
#                               batch_size = 128,
#                               shuffle = True)
# for i, batch in enumerate(train_dataloader):
#     print(len(batch['input']))
# rolling_windown = np.lib.stride_tricks.sliding_window_view(data_, 120, axis = 0, writeable = True)
# print(rolling_windown.shape)
# m = nn.Softmax(dim=1)
# input = torch.randn(1, 3)
# output = m(input)
# print(output, np.argmax(output[0]))

# m = np.array([1,2,3,4,5,6,7,8,9,10])
# n = np.array([1,2,3,4,5,7,8,0,4,10])
# v = np.sum(m==n)
# print(v)
# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

# for param in rnn.state_dict():
    # print(param, "\t", rnn.state_dict()[param].shape)