from turtle import update
from unicodedata import category
import torch
import time
import numpy as np
import pandas as pd
import logging
from dataloader import CustomDataset
from torch.utils.data.dataloader import DataLoader
from model_ import Conv_lstm
from utils import *
import sys
import os

device = torch.device("cpu")

def check_val(model_path, data_path, classification_model, mode):
    data_ = {}
    with np.load(data_path, 'rb') as f:
        for i in f:
            data_[i] = torch.Tensor(f[i].transpose(0,2,1))
    classification_model.load_state_dict(torch.load(model_path))
    classification_model.float()
    classification_model.eval()
    classification_model.to(device)
    sample_list = {}
    bee_ = 0               
    noise_ = 0
    cricket_ = 0
    num_samples = 0
    
    with torch.no_grad():
        for i in data_:
            bee = 0               
            noise = 0
            cricket = 0
            data = data_[i].float().to(device)
            print(data.size())
            print('{} {}: {} samples'.format(i, mode, len(data)))
            count = 0
            num_samples +=len(data)
            out = classification_model(data)
            classification = np.argmax(out, axis=1)
            for j in classification:
                if j == 0:
                    # print(count, 'That is noise sound!')
                    noise += 1
                    if i == 'noise':
                        noise_ += 1
                elif j == 1:
                    # print(count, 'That is bee sound!')
                    bee += 1
                    if i == 'bee':
                        bee_ +=1
                else:
                    # print(count, 'That is cricket sound!')
                    cricket += 1
                    if i == 'cricket':
                        cricket_ += 1
                count += 1
            print('Bee: {}\t|Noise: {}\t|Cricket: {}'.format(bee, noise, cricket))
            sample_list[i + mode] = {'bee': bee, 'noise': noise, 'cricket': cricket, 'num_sample': len(data)}
        acc = (bee_ + noise_ + cricket_)/num_samples     
        print(acc*100) 
    print("=================================================================")
    return acc*100, sample_list
def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("ERROR: Missing or invalid config file.")
        sys.exit(1)

    test_set = {}
    with np.load(config['data_dir'] +'buzz2_test_data.npz', 'rb') as f:
        #test_set = torch.Tensor(f['data'].transpose(0,2,1))
        for i in f:
            test_set[i] = torch.Tensor(f[i].transpose(0,2,1))
    with np.load(config['data_dir'] +'buzz2_val_data.npz', 'rb') as f_val:
        val_set = torch.Tensor(f_val['data'].transpose(0,2,1))

    ####
    classification_model = Conv_lstm(l_data=20000, 
                                    num_filters=config['num_filters'], 
                                    kernel_size=(config['kernel_size_1'], config['kernel_size_2']), 
                                    strides=(config['stride_1'], config['stride_2']), 
                                    maxpooling=(config['maxpool_1'], config['maxpool_2']), 
                                    d_hidden=config['d_hidden'], 
                                    num_layers= config['num_layers'])
#     path = "buzz2_model_{}num_layers_{}filters_({},{})kernel_size_({},{})stride_{}lr_{}epoch_{}batch_size.pt".format(
#         config['num_layers'],
#         config['num_filters'],
#         config["kernel_size_1"],
#         config["kernel_size_2"],
#         config["stride_1"],
#         config["stride_2"],
#         config['lr'],
#         config['best_epoch'],
#         config['batch_size']
#     )
#     print(path)
#     classification_model.load_state_dict(torch.load(config['result_dir'] + path))
#     classification_model.float()
#     classification_model.eval()
#     classification_model.to(device)
#     test_list = {}
#     val_list = {}
#     bee_ = 0               
#     noise_ = 0
#     cricket_ = 0
#     num_samples = 0
    
#     with torch.no_grad():
#         for i in test_set:
#             bee = 0               
#             noise = 0
#             cricket = 0
#             data = test_set[i].float().to(device)
#             print(data.size())
#             print('{} test: {} samples'.format(i, len(data)))
#             count = 0
#             num_samples +=len(data)
#             out = classification_model(data)
#             classification = np.argmax(out, axis=1)
#             for j in classification:
#                 if j == 0:
#                     # print(count, 'That is noise sound!')
#                     noise += 1
#                     if i == 'noise':
#                         noise_ += 1
#                 elif j == 1:
#                     # print(count, 'That is bee sound!')
#                     bee += 1
#                     if i == 'bee':
#                         bee_ +=1
#                 else:
#                     # print(count, 'That is cricket sound!')
#                     cricket += 1
#                     if i == 'cricket':
#                         cricket_ += 1
#                 count += 1
#             print('Bee: {}\t|Noise: {}\t|Cricket: {}'.format(bee, noise, cricket))
#             test_list[i + '_test'] = {'bee': bee, 'noise': noise, 'cricket': cricket, 'num_sample': len(data)}
#         acc_test = (bee_ + noise_ + cricket_)/num_samples     
#         print(acc_test*100) 
# ###

#         bee_ = 0               
#         noise_ = 0
#         cricket_ = 0
#         print('=====================================\nValidation:')
#         name = 'bee'
#         for i in range(3):
#             bee = 0               
#             noise = 0
#             cricket = 0
#             data = val_set[i*1000:1000*(i+1)]
#             out = classification_model(data)
#             classification = np.argmax(out, axis=1)
#             if i == 1:
#                 name='noise'
#             if i == 2:
#                 name='cricket'
#             for j in classification:
#                 if j == 0:
#                     # print(count, 'That is noise sound!')
#                     noise += 1
#                     if i == 1:
#                         noise_ += 1
                        
#                 elif j == 1:
#                     # print(count, 'That is bee sound!')
#                     bee += 1
#                     if i == 0:
#                         bee_ +=1
#                 else:
#                     # print(count, 'That is cricket sound!')
#                     cricket += 1
#                     if i == 2:
#                         cricket_ += 1
#             print('{} validation: {} samples'.format(name, len(data)))
#             print('Bee: {}\t|Noise: {}\t|Cricket: {}'.format(bee, noise, cricket))
            
#             val_list[name + '_val'] = {'bee': bee, 'noise': noise, 'cricket': cricket, 'num_sample': len(data)}
#         print((bee_ + noise_ + cricket_)*100/3000)
        
    acc_test, test_list = check_val(config['model_path'], 'pickle/buzz2_test_data.npz', classification_model, 'test')
    config['test_samples'] = test_list 
    config['accuracy_test'] = acc_test
    acc_val, val_list = check_val(config['model_path'], 'pickle/buzz2_val_data_.npz', classification_model, 'val')
    config['val_samples'] = val_list
    config['accuracy_val'] = acc_val
    if config['best_epoch'] != 0:
        acc_test, test_list = check_val(config['model_path'], 'pickle/buzz2_test_data.npz', classification_model, 'test')
        config['test_samples_best'] = test_list 
        config['accuracy_test_best'] = acc_test
        acc_val, val_list = check_val(config['best_model_path'], 'pickle/buzz2_val_data.npz', classification_model, 'val')
        config['val_samples_best'] = val_list
        config['accuracy_val_best'] = acc_val
    
    
    filename = "{}_num_layers_({},{})kernel_size_{}n_filters_({},{})stride_({},{})maxpool_{}lr_{}batch".format(
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
    PATH = os.path.join("configs/done", "results_{}.yml".format(filename))
    save_config(PATH, config)

    print("DONE.")


if __name__ == "__main__":
    main()
