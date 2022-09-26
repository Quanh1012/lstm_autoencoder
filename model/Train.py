import torch
import time
import numpy as np
import pandas as pd
import logging
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_lstm
from trainer_classification import ClassificationTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cuda:0" if torch.cuda.is_available() else 

    
config = {'data_dir': 'pickle/',
          'num_epoch': 100,
          'd_hidden': 3,
          'num_layers': 3,
          'batch_size': 128,
          'num_filters': 64,
          'kernel_size': 80,
          'stride': (4,1),
          'result_dir': '/home/quanh/Documents/quanh/results/',
          'lr': 0.001,
          'weight_decay': 0.0001
          }

data_set = CustomDataset(config)
lenght_data = data_set.get_lenght()
data_dataloader = DataLoader(data_set, 
                              batch_size = 128,
                              shuffle = True)
train_dataloader = []
val_dataloader = []
for i, batch in enumerate(data_dataloader):
    if i < len(data_dataloader)*0.7:
        train_dataloader.append(batch)
    else:
        val_dataloader.append(batch)
# val_dataloader = DataLoader(val_set,
#                             batch_size = 128,
#                             shuffle = True)
###
classification_model = Conv_lstm(20000, config['num_filters'], config['kernel_size'], config['stride'], config['d_hidden'], config['num_layers'])
classification_model.float()
classification_trainer = ClassificationTrainer(model=classification_model,
                                               train_data=train_dataloader,
                                               val_data=val_dataloader,
                                               device=device,
                                               config=config)
classification_trainer.train()

