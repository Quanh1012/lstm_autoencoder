from distutils.command.config import config
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import copy

class ClassificationTrainer():
    def __init__(self, model, train_data, val_data, val_set, device, config, buzz='buzz1'):
        self.model = model
    
        self.train_data = train_data
        self.val_data = val_data
        self.val_set = val_set
        self.device = device
        self.config = config
        self.max_acc = 95.67
        self.max_val = 0
        self.train_loss_list = list()
        self.test_loss_list = list()
        self.train_acc_list = list()
        self.test_acc_list = list()
        self.best_epoch = 0
        self.best_model = None
        self.buzz = buzz
        self.path = self.config['result_dir'] + "{}_model_{}num_layers_({},{})filters_({},{})kernel_size_({},{})maxpool_{}lr_{}epoch_{}batch_size.pt".format(
            self.buzz,
            self.config['num_layers'],
            self.config['num_filters_1'],
            self.config['num_filters_2'],
            self.config["kernel_size_1"],
            self.config["kernel_size_2"],
            # self.config["stride_1"],
            # self.config["stride_2"],
            self.config["maxpool_1"],
            self.config["maxpool_2"],
            self.config['lr'],
            self.config['num_epoch'],
            self.config['batch_size']
        )
    
    def train_epoch(self, criterion, opt, epoch):
        train_loss = 0.0
        train_acc = 0.0
        self.model.train()
        lenght_train = 0
        for i, data in enumerate(self.train_data):
            input = data['input'].float().to(self.device)            
            target = data['labels'].float().to(self.device)
            out = self.model(input)
            lenght_train += len(input)
            opt.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            # print('\rEpoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}'.format(epoch, train_loss/lenght, train_acc/lenght), end='')
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.train_data):
                input = data['input'].float().to(self.device)            
                target = data['labels'].float().to(self.device)
                
                out = self.model(input)
                
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target, axis=1)
                #print(classes, target_)
                train_acc += torch.sum(classes == target_)
        self.train_loss = train_loss/lenght_train
        self.train_acc = train_acc.item()*100/lenght_train
        self.train_loss_list.append(self.train_loss)
        self.train_acc_list.append(self.train_acc)
        
        
        # print('\nEpoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}'.format(epoch, train_loss, train_acc))
        if self.val_data is None:
            if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_epoch = epoch
        else:
            self.validate_epoch(criterion, epoch)
        
    def validate_epoch(self, criterion, epoch):
        test_loss = 0.0
        test_acc = 0.
        val_acc = 0.0
        lenght = 0
        lenghtval = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input = data['input'].float().to(self.device)
                target = data['labels'].float().to(self.device)
                
                out = self.model(input)
                lenght += len(input)
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target, axis=1)
                test_acc += torch.sum(classes == target_)
                loss = criterion(out, target)
                test_loss += loss.item()
            
            for i, data in enumerate(self.val_set):
                input = data['input'].float().to(self.device)
                target = data['labels'].float().to(self.device)
                
                out = self.model(input)
                lenghtval += len(input)
                
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target, axis=1)

                val_acc += torch.sum(classes == target_)
            
        val_acc = val_acc*100/lenghtval  
        test_loss = test_loss/lenght
        test_acc = test_acc.item()*100/lenght
        self.test_loss_list.append(test_loss)
        self.test_acc_list.append(test_acc)
        print('\rEpoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}\t| Test_loss: {:.6f}\t|Test accuracy: {:.6f}\t|Val accuracy: {:.6f} '.format(
                                                                                                                                        epoch, 
                                                                                                                                        self.train_loss, 
                                                                                                                                        self.train_acc, 
                                                                                                                                        test_loss, 
                                                                                                                                        test_acc, 
                                                                                                                                        val_acc), 
                                                                                                                                        end="\r") 
        # wandb.log({'train_loss': self.train_loss, 'test_loss': test_loss})
        if val_acc >= self.max_val:
            self.max_val = val_acc
            self.best_epoch = epoch
            if self.max_val >= 97.5 and test_acc > self.max_acc:
                self.best_model_path = self.config['result_dir'] + "{}_model_{}num_layers_({},{})filters_({},{})kernel_size_({},{})maxpool_{}lr_{}epoch_{}batch_size.pt".format(
                    self.buzz,
                    self.config['num_layers'],
                    self.config['num_filters_1'],
                    self.config['num_filters_2'],
                    self.config["kernel_size_1"],
                    self.config["kernel_size_2"],
                    # self.config["stride_1"],
                    # self.config["stride_2"],
                    self.config["maxpool_1"],
                    self.config["maxpool_2"],
                    self.config['lr'],
                    epoch,
                    self.config['batch_size']
                )
                self.best_model = copy.deepcopy(self.model)
            # torch.save(self.model.state_dict(), path)
    def train(self):
        self.model.to(self.device)  
        
        start = time.perf_counter()
        model_opt = optim.Adam(self.model.parameters(), self.config['lr'], weight_decay = self.config['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        print("-----START TRAINING CLASSIFICATION-----")
        for epoch in range(1, self.config['num_epoch'] + 1):
            self.train_epoch(criterion, model_opt, epoch)  
        print('\n')
        self.config['train_time'] = (time.perf_counter() - start)/60
        print("----COMPLETED TRAINING CLASSIFICATION-----")

        self.config["best_epoch"] = self.best_epoch
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), self.best_model_path)
            self.config['best_model_path'] = self.best_model_path
        else:
            self.config['best_model_path'] = ''

        self.config['val_acc'] = self.max_val
        self.config['train_loss'] = self.train_loss_list
        self.config['train_acc'] = self.train_acc_list
        self.config['test_loss'] = self.test_loss_list
        self.config['test_accuracy'] = self.test_acc_list
        # self.config['model_path'] = self.path 
        # self.save_loss()

    def get_updated_config(self):
        return self.config
    
    def load_model(self, path = None):
        if path is None:
            path = self.config['result_dir'] + "model.pt"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()   
      