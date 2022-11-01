import enum
from turtle import update
from unicodedata import category
import torch
import numpy as np
import pandas as pd
import logging
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_lstm
from utils import *
import sys
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    # with np.load('pickle/buzz1_val/buzz1_val_1.npz', 'rb', allow_pickle=True) as f:
    #     print(f['yval'])
    

    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("ERROR: Missing or invalid config file.")
        sys.exit(1)


    train_set = CustomDataset(config, 'buzz2_test_training_data.npz')
    train_dataloader = DataLoader(train_set, 
                                batch_size = 256,
                                shuffle = False)
  
    ####
    classification_model = Conv_lstm(20000, config['num_filters'], (config['kernel_size_1'], config['kernel_size_2']), (config['stride_1'], config['stride_2']), config['d_hidden'], config['num_layers'])
    path = "buzz2_model_{}num_layers_{}filters_({},{})kernel_size_({},{})stride_{}lr_{}epoch_{}batch_size.pt".format(
        config['num_layers'],
        config['num_filters'],
        config["kernel_size_1"],
        config["kernel_size_2"],
        config["stride_1"],
        config["stride_2"],
        config['lr'],
        config['best_epoch'],
        config['batch_size']
    )
    print(path)
    classification_model.load_state_dict(torch.load(config['result_dir'] + path))
    classification_model.float()
    classification_model.eval()
    classification_model.to(device)
    train_list = {
                'conv1_relu': [],
                'batchNorm1': [],
                'maxpooling1': [],
                'conv2_relu': [],
                'batchNorm2': [],
                'maxpooling2': [],
                'lstm':[],
                'output': []
            }
    target_list = []
    print(len(train_dataloader))
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            input = data['input'].float().to(device)
            target = data['labels']
            out = classification_model(input)
            list_output_train = classification_model.get_output_per_layer()
            list_output_train['output'] = out

            for key in list_output_train:
                for data in list_output_train[key]:
                    train_list[key].append(torch.Tensor.numpy(data.to("cpu")))                
            for val in target:
                target_list.append(torch.Tensor.numpy(val))
            if((i + 1)%5 == 0):
                print("Save file")
                np.savez(config['data_dir'] + 'buzz2_test/' + 'buzz2_test_'+ str(int((i + 1)/5))+ '.npz', **train_list, yval=target_list)
                train_list.clear()
                train_list = { 
                                'conv1_relu': [],
                                'batchNorm1': [],
                                'maxpooling1': [],
                                'conv2_relu': [],
                                'batchNorm2': [],
                                'maxpooling2': [],
                                'lstm':[],
                                'output': []
                            }
                target_list.clear()            
            if((i + 1)/5 > int(len(train_dataloader)/5)):
                np.savez(config['data_dir'] + 'buzz2_test/' + 'buzz2_test_'+ str(int((i + 1)/5) + 1) + '.npz', **train_list, yval=target_list)
            
    # print(len(train_list))
if __name__ == "__main__":
    main()
