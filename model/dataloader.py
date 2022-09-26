import numpy as np
import pickle
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset()
    
    def __len__(self):
        return self.data_.shape[0]
    
    def __getitem__(self, index):
        if self.mode == 'test':
            return self.data_[index]
        else:
            return {'input':self.data_[index], 
                    'labels': self.labels[index]}
    
    def load_dataset(self):
        data = {}
        if self.mode != 'test':
            with np.load('pickle/buzz1_train_data.npz', 'rb') as f:
                data['train'] = f['data']
                data['labels'] = f['labels']
            
            self.data_ = data['train'].transpose(0,2,1)
            # self.data_ = self.split_data(train)
            # self.rolling_windown_data = np.lib.stride_tricks.sliding_window_view(train_split, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)

            self.labels = data['labels']
            # self.rolling_windown_labels= np.lib.stride_tricks.sliding_window_view(labels_split, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)
        else:
            with np.load('pickle/buzz1_test_data.npz', 'rb') as f:
                data = f['data']
            
            self.data_ = data.tranpose(0,2,1)
            # self.rolling_windown_data = np.lib.stride_tricks.sliding_window_view(test, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)
        self.lenght = len(self.data_)
    def get_lenght(self):
        return self.lenght
            
    
    # def split_data(self, data):
    #     if self.mode == 'train':
    #         data = data[: int(data.shape[0]*0.7), :]
    #     elif self.mode == 'validate':
    #         data = data[int(data.shape[0]*0.7):, :]
    #     return data  
        
            