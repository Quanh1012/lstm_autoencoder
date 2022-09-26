from unicodedata import category
import torch
import time
import numpy as np
import pandas as pd
import logging
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_lstm
from trainer_classification import ClassificationTrainer

device = torch.device("cpu")


config = {'data_dir': 'pickle/',
          'num_epoch': 100,
          'd_hidden': 3,
          'num_layers': 2,
          'batch_size': 128,
          'num_filters': 128,
          'kernel_size': 80,
          'stride': (4,1),
          'result_dir': '/home/quanh/Documents/quanh/results/',
          'lr': 0.001,
          'weight_decay': 0.0001
          }

test_set = {}
with np.load('pickle/buzz1_test_data.npz', 'rb') as f:
    for i in f:
        test_set[i] = torch.Tensor(f[i].transpose(0,2,1))
####
classification_model = Conv_lstm(20000, config['num_filters'], config['kernel_size'], config['stride'], config['d_hidden'], config['num_layers'])
classification_model.load_state_dict(torch.load(config['result_dir'] + 'model_2_num_layers_128_filters.pt'))
classification_model.float()
classification_model.eval()
classification_model.to(device)

loss = torch.nn.CrossEntropyLoss()
start = time.perf_counter()
bee = 0               
noise = 0
cricket = 0
num_samples = 0
with torch.no_grad():
    # for i in test_set:
    data = test_set['bee'].float().to(device)
    print(data.size())
    print('{} test: {} samples'.format('bee', len(data)))
    count = 0
    num_samples +=len(data)
    out = classification_model(data)
    classification = np.argmax(out, axis=1)
    for j in classification:
        if j == 0:
            # print(count, 'That is noise sound!')
            noise += 1
        elif j == 1:
            # print(count, 'That is bee sound!')
            bee += 1
        else:
            # print(count, 'That is cricket sound!')
            cricket += 1
        count += 1

   
# acc = (bee + noise + cricket)/num_samples   
print('Bee: {}\t|Noise: {}\t|Cricket: {}'.format(bee, noise, cricket))

