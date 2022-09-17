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
          'num_epoch': 10,
          'd_model': 20000,
          'd_hidden': 256,
          'num_layers': 2,
          'batch_size': 128,
          'result_dir': '/home/quanhhh/Documents/QAnh/results/'}

train_set = CustomDataset(config)
train_dataloader = DataLoader(train_set, 
                              batch_size = 128,
                              shuffle = True)
val_set = CustomDataset(config, mode='validate')

val_dataloader = DataLoader(val_set,
                            batch_size = 128,
                            shuffle = True)

autoencoder_model = Lstm_autoencoder(20000, 256, 2, 0.1)
autoencoder_model.float()
autoencoder_trainer = AutoencoderTrainer(autoencoder_model,
                                         train_data=train_dataloader,
                                         val_data=val_dataloader,
                                         device=device,
                                         config=config)
autoencoder_trainer.train()
config = autoencoder_trainer.get_updated_config()

###
classification_model = Fully_connected(20000, 256, 0.1)
classification_model.float()
classification_trainer = ClassificationTrainer(autoencoder_model=autoencoder_model,
                                               fc_function=classification_model,
                                               train_data=train_dataloader,
                                               val_data=val_dataloader,
                                               device=device,
                                               config=config)
classification_trainer.train()

