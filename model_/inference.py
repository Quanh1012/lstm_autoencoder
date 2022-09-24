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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = {'data_dir': 'pickle/',
          'l_win': 120,
          'autoencoder_dims': 500}
test_set = {}
with np.load('pickle/buzz1_test_data.npz', 'rb') as f:
    test_set = f

####
classification_model = Conv_lstm(256, 3, [4, 1], 3, 3, 128)
classification_model.load_state_dict()
classification_model.float()
classification_model.eval()
classification_model.to(device)

loss = torch.nn.CrossEntropyLoss()
start = time.perf_counter()
bee = 0               
noise = 0
cricket = 0
num_samples = 0
for i in enumerate(test_set):
    data = test_set[i].float().to(device)
    print('{} test: {} samples'.format(i, len(data)))
    count = 0
    num_samples +=len(data)
    for src in data:
        input = Conv_lstm(src)
        out = classification_model(input)
        
        classification = np.argmax(out)
        if classification == 0:
            print(count, 'That is noise sound!')
            noise += 1
        elif classification == 1:
            print(count, 'That is bee sound!')
            bee += 1
        else:
            print(count, 'That is cricket sound!')
            cricket += 1
        count += 1
acc = (bee + noise + cricket)/num_samples   
print('Bee: {}\t|Noise: {}\t|Cricket: {}\t|Accuracy: {}'.format(bee, noise, cricket, acc))

