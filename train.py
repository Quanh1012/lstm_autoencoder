import torch
import sys
import time
import numpy as np
import pandas as pd
import argparse

from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_gru
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
    # config = {
    #     "batch_size": 128,
    #     "d_hidden": 3,
    #     "data_dir": '/home/quanhhh/Documents/model/data_dir/',
    #     "dropout": 0.08753825463680935,
    #     "kernel_size_1": 3,
    #     "kernel_size_2": 3,
    #     "kernel_size_3": 63,
    #     "kernel_size_4": 15,
    #     "lr": 0.0018402003815178216,
    #     "maxpool_1": 2,
    #     "maxpool_2": 2,
    #     "maxpool_3": 3,
    #     "maxpool_4": 7,
    #     "num_epoch": 15,
    #     "num_filters_1": 82,
    #     "num_filters_2": 84,
    #     "num_layers": 4,
    #     "result_dir": '/home/quanhhh/Documents/model/model_spec/result_wb/',
    #     "stride_1": 3,
    #     "stride_2": 3,
    #     "stride_3": 8,
    #     "stride_4": 11,
    #     "weight_decay": 0.00012632223004014332,
    # }

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
    train_set = CustomDataset(config, 'buzz2_traindata.npz')
    train_dataloader = DataLoader(train_set, 
                                batch_size = 128,
                                shuffle = True)
    shape = train_set.getshape()
    val_set = CustomDataset(config, 'buzz2_testdata.npz')
    val_dataloader = DataLoader(val_set,
                                batch_size = 128,
                                shuffle = True)
    ###
    val_set = CustomDataset(config, 'buzz2_valdata.npz')
    val_set_ = DataLoader(val_set,
                                batch_size = 128,
                                shuffle = True)
    torch.manual_seed(10)
    classification_model = Conv_gru(shape_data=shape, 
                        num_filters=(config['num_filters_1'], config['num_filters_2']),
                        kernel_size=(config['kernel_size_1'], config['kernel_size_2']), 
                        # strides=(config['stride_1'], config['stride_2']), 
                        maxpooling=(config['maxpool_1'], config['maxpool_2']), 
                        d_hidden=config['d_hidden'], 
                        num_layers= config['num_layers'])
    classification_model.float()
    classification_trainer = ClassificationTrainer(model=classification_model,
                                                train_data=train_dataloader,
                                                val_data=val_dataloader,
                                                val_set=val_set_,
                                                device=device,
                                                config=config,
                                                buzz="buzz2")
    classification_trainer.train()
    config = classification_trainer.get_updated_config()
    # filename = "{}num_layers_({},{})kernel_size_{}n_filters_({},{})maxpool_{}lr_{}batch_size".format(
    #     config["num_layers"],
    #     config["kernel_size_1"],
    #     config["kernel_size_2"],
    #     config["num_filters"],
    #     # config["stride_1"],
    #     # config["stride_2"],
    #     config["maxpool_1"],
    #     config["maxpool_2"],
    #     config['lr'],
    #     config['batch_size']
    # ).replace(".", "_")
    # PATH = os.path.join("configs/", "{}.yml".format(filename))
    save_config("/home/quanhhh/Documents/model/config.yaml", config)
    print('ok')
if __name__ == "__main__":
    main()