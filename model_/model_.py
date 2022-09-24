import torch.nn as nn
class Conv_lstm(nn.Module):
    def __init__(self, l_data, num_filters, kernel_size, strides, d_hidden, num_layers, dropout=0.1):
        super().__init__()
        self.l_data = l_data
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1, self.num_filters, self.kernel_size,self.strides[0])
        nn.init.xavier_uniform_(self.conv1.weight)
        d_model1 = int((self.l_data - self.kernel_size)/self.strides[0] + 1)
        d_model2 = int((d_model1 - 4)/4 + 1)
        self.conv2 = nn.Conv1d(self.num_filters, self.num_filters, 3, self.strides[1])
        nn.init.xavier_uniform_(self.conv2.weight)
        d_model3 = int((d_model2 - 3)/self.strides[1] + 1)
        self.f_activation1 = nn.ReLU()
        self.f_activation2 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(self.num_filters)
        self.batchNorm2 = nn.BatchNorm1d(self.num_filters)
        self.maxpooling1 = nn.MaxPool1d(4)
        self.maxpooling2 = nn.MaxPool1d(4)
        d_model4 = int((d_model3 - 4)/4 + 1)
        print(d_model4)
        self.lstm = nn.LSTM(d_model4, self.d_hidden, self.num_layers, batch_first=True, dropout=self.dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        #layer1
        x = self.conv1(x)
        x = self.f_activation1(x)
        
        #layer2
        x = self.batchNorm1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.f_activation2(x)
        
        #layer3
        x = self.batchNorm2(x)
        x = self.maxpooling2(x)
        #layer4
        h_c = self.init_hidden(len(x))
        out, (_, _) = self.lstm(x, h_c)
        return self.softmax(out[:, -1, :])
            
    def init_hidden(self, batch_size):
        weight = next(self.lstm.parameters()).data
        
        hidden = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()        
        cell = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()
        
        return (hidden, cell)

# class Lstm(nn.Module):
#     def __init__(self, d_hidden, num_layers, d_input=0, dropout=0.1):
#         super().__init__()
#         self.d_input = d_input
#         self.d_hidden = d_hidden
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         self.lstm = nn.LSTM(self.d_input, self.d_hidden, self.num_layers, self.dropout)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         h_c = self.init_hidden(len(x))
#         out, (_, _) = self.lstm(x, h_c)
#         return self.softmax(out[:, -1, :])
        
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
        
#         hidden = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()        
#         cell = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()
        
#         return (hidden, cell)
