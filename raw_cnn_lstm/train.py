import torch
import sys
import time
import numpy as np
import pandas as pd
import argparse

from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_lstm
from trainer_classification import ClassificationTrainer
from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("ERROR: Missing or invalid config file.")
        sys.exit(1)

    # train_set = CustomDataset(config, 'buzz1_train_data.npz')

    # data_dataloader = DataLoader(train_set, 
    #                               batch_size = 128,
    #                               shuffle = True)
    # train_dataloader = []
    # val_dataloader = []
    # for i, batch in enumerate(data_dataloader):
    #     if i < len(data_dataloader)*0.7:
    #         train_dataloader.append(batch)
    #     else:
    #         val_dataloader.append(batch)
            
    #buzz2
    train_set = CustomDataset(config, 'buzz2_train_data.npz')
    train_dataloader = DataLoader(train_set, 
                                batch_size = 128,
                                shuffle = True)
    val_set = CustomDataset(config, 'buzz2_test_training_data.npz')
    val_dataloader = DataLoader(val_set,
                                batch_size = 256,
                                shuffle = True)
    ###
    val_set = CustomDataset(config, 'buzz2_val_data.npz')
    val_set_ = DataLoader(val_set,
                                batch_size = 128,
                                shuffle = True)
    torch.manual_seed(10)
    classification_model = Conv_lstm(l_data=20000, 
                                num_filters=config['num_filters'], 
                                kernel_size=(config['kernel_size_1'], config['kernel_size_2']), 
                                strides=(config['stride_1'], config['stride_2']), 
                                maxpooling=(config['maxpool_1'], config['maxpool_2']), 
                                d_hidden=config['d_hidden'], 
                                num_layers= config['num_layers'])
    classification_model.float()
    classification_trainer = ClassificationTrainer(model=classification_model,
                                                train_data=train_dataloader,
                                                val_data=val_dataloader,
                                                val_set=val_set_,
                                                device=device,
                                                config=config)
    classification_trainer.train()
    config = classification_trainer.get_updated_config()
    filename = "{}num_layers_({},{})kernel_size_{}n_filters_({},{})stride_({},{})maxpool_{}lr_{}batch_size".format(
        config["num_layers"],
        config["kernel_size_1"],
        config["kernel_size_2"],
        config["num_filters"],
        config["stride_1"],
        config["stride_2"],
        config["maxpool_1"],
        config["maxpool_2"],
        config['lr'],
        config['batch_size']
    ).replace(".", "_")
    PATH = os.path.join("configs/", "{}.yml".format(filename))
    save_config(PATH, config)
    print('ok')
if __name__ == "__main__":
    main()