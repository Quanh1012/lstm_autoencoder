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
          'l_win': 120,
          'num_epoch': 50,
          'd_hidden': 256,
          'num_layers': 2,
          'batch_size': 128,
          'result_dir': '/home/quanhhh/Documents/QAnh/results/',
          'lr': 0.001,
          'weight_decay': 0.0001
          }

train_set = CustomDataset(config)
train_dataloader = DataLoader(train_set, 
                              batch_size = 128,
                              shuffle = True)
val_set = CustomDataset(config, mode='validate')

val_dataloader = DataLoader(val_set,
                            batch_size = 128,
                            shuffle = True)
###
classification_model = Conv_lstm(20000, 256, 80, (4,1), 3, 3)
classification_model.float()
classification_trainer = ClassificationTrainer(model=classification_model,
                                               train_data=train_dataloader,
                                               val_data=val_dataloader,
                                               device=device,
                                               config=config)
classification_trainer.train()

