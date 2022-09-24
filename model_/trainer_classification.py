import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time

class ClassificationTrainer():
    def __init__(self, model, train_data, val_data, device, config):
        self.model = model
    
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.train_acc_list = list()
        self.val_acc_list = list()
        self.best_epoch = 0
    
    def train_epoch(self, criterion, opt, epoch):
        train_loss = 0.0
        train_acc = 0.0
        self.model.train()
        lenght = 0
        for i, data in enumerate(self.train_data):
            input = data['input'].float().to(self.device)            
            target = data['labels'].float().to(self.device)
            out = self.model(input)
            
            classes = torch.argmax(out, axis=1)
            target_ = torch.argmax(target, axis=1)
            #print(classes, target_)
            train_acc += torch.sum(classes == target_)
            lenght += len(input)
            opt.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            print('\rEpoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}'.format(epoch, loss/len(input), train_acc/lenght), end='')
        
        train_loss = train_loss/lenght
        train_acc = train_acc/lenght
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        
        print('\nEpoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}'.format(epoch, train_loss, train_acc))
        if self.val_data is None:
            if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_epoch = epoch
        else:
            self.validate_epoch(criterion, opt, epoch)
        
    def validate_epoch(self, criterion, opt, epoch):
        val_loss = 0.0
        val_acc = 0.0
        lenght = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input = data['input'].float().to(self.device)
                target = data['labels'].float().to(self.device)
                out = self.model(input)
                lenght += len(input)
                classes = torch.argmax(out, axis=1)
                #print(classes)
                target_ = torch.argmax(target, axis=1)
                #print(target_)
                val_acc += torch.sum(classes == target_)
                loss = criterion(out, target)
                val_loss += loss.item()
                print('\rEpoch: {}\t| Validation_loss: {:.6f}\t|Valdation accuracy: {:.6f}'.format(epoch, loss/len(input), val_acc/lenght), end='') 
            
            val_loss = val_loss/lenght
            val_acc = val_acc/lenght
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
            print('\nEpoch: {}\t| Validation_loss: {:.6f}\t|Valdation accuracy: {:.6f}'.format(epoch, val_loss, val_acc)) 
            
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                self.best_epoch = epoch
            
    def train(self):
        self.model.to(self.device)  
        
        start = time.perf_counter()
        model_opt = optim.Adam(self.model.parameters(), self.config['lr'], weight_decay = self.config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        print("-----START TRAINING CLASSIFICATION-----")
        for epoch in range(1, self.config['num_epoch'] + 1):
            self.train_epoch(criterion, model_opt, epoch)  
        print("----COMPLETED TRAINING CLASSIFICATION-----")
        self.config["best_epoch"] = self.best_epoch
        self.save_loss()
    
    def save_loss(self):
        if self.val_data is not None:
            df_loss = pd.DataFrame([i+1, self.train_loss_list[i], self.val_loss_list[i]] for i in range(len(self.train_loss_list)))
            df_loss.save(self.config['result_dir'] + 'classification_epoch_loss.csv', 
                         index=False, 
                         header=['Epoch', 'Train_loss', 'Validation_loss'])
        else:
            df_loss = pd.DataFrame([i+1, self.train_loss_list[i]] for i in range(len(self.train_loss_list)))
            df_loss.save(self.config['result_dir'] + 'classification_epoch_loss.csv', 
                         index=False, 
                         header=['Epoch', 'Train_loss'])
           
    def get_updated_config(self):
        return self.config
    
    def load_model(self, path = None):
        if path is None:
            path = self.config['result_dir'] + "autoencoder+model.pt"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()        