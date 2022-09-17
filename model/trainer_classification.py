import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time

class ClassificationTrainer():
    def __init__(self, autoencoder_model, fc_function, train_data, val_data, device, config):
        self.model = fc_function
        if autoencoder_model is not None:
            self.encoder = autoencoder_model.encoder
            self.encoder.eval()
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.val_losss_list = list()
        self.train_acc_list = list()
        self.val_acc_list = list()
        self.best_epoch = 0
    
    def train_epoch(self, criterion, opt, epoch):
        train_loss = 0.0
        train_acc = 0.0
        self.model.train()
        
        for i, data in enumerate(self.train_data):
            input = data['input'].float().to(self.device)
            
            h1, h2, h3 = self.encoder.init_hidden(self.config['d_hidden'], len(data))
            input = self.encoder(input, h1,h2,h3)
            
            target = data['labels'].float().to(self.device)
            out = self.model(input)
            
            classes = np.argmax(out, axis=1)
            target_ = np.argmax(target, axis=1)
            
            train_acc += np.sum(classes == target_)
            opt.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        train_loss = train_loss/len(self.train_data)
        train_acc = train_acc/len(self.train_data)
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        
        logging.info('Epoch: {}\t| Train_loss: {:.6f}\t|Train accuracy: {:.6f}'.format(epoch, train_loss, train_acc))
        if self.val_data is None:
            if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_epoch = epoch
        else:
            self.validate_epoch(criterion, opt, epoch)
        
    def validate_epoch(self, criterion, opt, epoch):
        val_loss = 0.0
        val_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input = data['input'].float()
                input = input.to(self.device)
                h1, h2, h3 = self.encoder.init_hidden(self.config['d_hidden'], len(data['input']))
                input = self.encoder(input, h1,h2,h3)
                
                target = data['labels'].float()
                target = target.to(self.device)
                out = self.model(input)
                
                classes = np.argmax(out, axis=1)
                target_ = np.argmax(target, axis=1)
                val_acc += np.sum(classes == target_)
                loss = criterion(out, target)
                val_loss += loss.item()
            
            val_loss = val_loss/len(self.val_data)
            val_acc = val_acc/len(self.val_data)
            self.val_losss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
            logging.info('Epoch: {}\t| Validation_loss: {:.6f}\t|Valdation accuracy: {:.6f}'.format(epoch, val_loss, val_acc)) 
            
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                self.best_epoch = epoch
            
    def train(self):
        self.model.to(self.device)  
        self.encoder.to(self.device)
        
        start = time.perf_counter()
        model_opt = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        logging.info("-----START TRAINING CLASSIFICATION-----")
        for epoch in range(1, self.config['num_epoch'] + 1):
            self.train_epoch(criterion, model_opt, epoch)  
        logging.info("----COMPLETED TRAINING CLASSIFICATION-----")
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