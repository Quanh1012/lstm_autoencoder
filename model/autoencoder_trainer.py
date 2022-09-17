import logging
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
from torch import optim
import time

class AutoencoderTrainer():
    def __init__(self, autoencoder_model, train_data, val_data, device, config):
        self.model = autoencoder_model
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.best_epoch = 0
        self.best_model = None
    
    def train_epoch(self, criterion, opt, epoch):
        train_loss = 0.0
        self.model.train()
        for i, data in enumerate(self.train_data):
            input = data['input'].to(self.device)
            out = self.model(input, len(data))
            
            opt.zero_grad()
            loss = criterion(out, input)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
        train_loss = train_loss/len(self.train_data)
        self.train_loss_list.append(train_loss)
        
        logging.info('Epoch: {}\t| Train_loss: {:.6f}'.format(epoch, train_loss))
        
        self.validate_epoch(criterion, opt, epoch)
    
    def validate_epoch(self, criterion, opt, epoch):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input = data.float().to(self.device)
                out = self.model(input, len(data))
            
                loss = criterion(out, input)
                val_loss += loss.item()
            
            
        val_loss = val_loss/len(self.val_data)
        self.val_loss_list.append(val_loss)
        
        logging.infor('Epoch: {}\t| Val_loss: {:6.f}'.format(epoch, val_loss))
        
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.best_epoch = epoch
            self.best_model = deepcopy(self.model.state_dict())
    
    def train(self):
        self.model.to(self.device)
        
        start = time.perf_counter()
        logging.info("-----START TRAINING THE LSTM_AUTOENCODER-----")
        model_opt = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(1, self.config['num_epoch'] + 1):
            self.train_epoch(criterion, model_opt, epoch)
        logging.info('-----COMPLETED TRAINING-----')
        self.config['train_time'] = (time.perf_counter() - start)/60
        
        self.config['best_epoch'] = self.best_epoch
        torch.save(self.best_model, self.config['result_dir'] + "autoencoder_mode.pt")
        self.save_loss()
    
    def save_loss(self):
        if self.val_data is not None:
            df_loss = pd.DataFrame([i+1, self.train_loss_list[i], self.val_loss_list[i]] for i in range(len(self.train_loss_list)))
            df_loss.save(self.config['result_dir'] + 'autoencoder_epoch_loss.csv', 
                         index=False, 
                         header=['Epoch', 'Train_loss', 'Validation_loss'])
        else:
            df_loss = pd.DataFrame([i+1, self.train_loss_list[i]] for i in range(len(self.train_loss_list)))
            df_loss.save(self.config['result_dir'] + 'autoencoder_epoch_loss.csv', 
                         index=False, 
                         header=['Epoch', 'Train_loss'])
    def get_updated_config(self):
        return self.config
    def load_model(self, path = None):
        if path is None:
            path = self.config['result_dir'] + "autoencoder+model.pt"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
        
        

