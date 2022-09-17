from unicodedata import category
import torch
import time
import numpy as np
import pandas as pd
import logging
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from lstm_Autoencoder import Lstm_autoencoder, Fully_connected
from autoencoder_trainer import AutoencoderTrainer
from trainer_classification import ClassificationTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = {'data_dir': 'pickle/',
          'l_win': 120,
          'autoencoder_dims': 500}
test_set = {}
with np.load('pickle/buzz1_test_data.npz', 'rb') as f:
    test_set = f
autoencoder_model = Lstm_autoencoder(20000, 256, 2, 0.1)
autoencoder_model.load_state_dict()
autoencoder_model.float()
encoder = autoencoder_model.encoder
encoder.eval()
encoder.to(device)
####
classification_model = Fully_connected(20000, 256, 0.1)
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
        h1, h2, h3 = encoder.init_hidden(config['d_hidden'], len(src))
        input = encoder(src, h1, h2, h3)
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

