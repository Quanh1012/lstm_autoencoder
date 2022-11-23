import numpy as np
import pickle
import os
from matplotlib import image
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, config, path):
        super().__init__()
        self.config = config
        self.path = path
        self.load_dataset()
    
    def __len__(self):
        return self.data_.shape[0]
    
    def __getitem__(self, index):
        return {'input':self.data_[index], 
                'labels': self.labels[index]}
    
    def load_dataset(self):
        data = {}
        path_ = self.config['data_dir'] + self.path
        with np.load(path_, 'rb') as f:
            data['data'] = f['data']
            data['label'] = f['label']
        self.data_ = data['data'].transpose(0,2,1)
        self.data_ = np.expand_dims(self.data_, axis=1)
        
        
        # self.shape_h = self.data_.shape[2]
        # self.shape_w = self.data_.shape[3]
        self.labels = data['label']
    def getshape(self):
        return self.data_.shape[1:]
            
    def getshape(self):
        return self.data_.shape[1:]
# def split_data(self, data):
#     if self.mode == 'train':
#         data = data[: int(data.shape[0]*0.7), :,:]
#     elif self.mode == 'val':
#         data = data[int(data.shape[0]*0.7):, :,:]
#     return data  
        
            