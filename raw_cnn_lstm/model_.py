import torch.nn as nn
class Conv_lstm(nn.Module):
    def __init__(self, l_data, num_filters, kernel_size, strides, maxpooling, d_hidden, num_layers, dropout=0.1):
        super().__init__()
        self.l_data = l_data
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1, self.num_filters, self.kernel_size[0],self.strides[0])
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        
        d_model1 = int((self.l_data - self.kernel_size[0])/self.strides[0] + 1)
        d_model2 = int((d_model1 - maxpooling[0])/maxpooling[0] + 1)
        
        self.conv2 = nn.Conv1d(self.num_filters, self.num_filters, self.kernel_size[1], self.strides[1])
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        d_model3 = int((d_model2 - self.kernel_size[1])/self.strides[1] + 1)
        
        self.f_activation1 = nn.ReLU()
        self.f_activation2 = nn.ReLU()
        
        self.batchNorm1 = nn.BatchNorm1d(num_features=self.num_filters) # momentum=0.99
        self.batchNorm2 = nn.BatchNorm1d(num_features=self.num_filters)
        
        self.maxpooling1 = nn.MaxPool1d(maxpooling[0])
        self.maxpooling2 = nn.MaxPool1d(maxpooling[1])
        
        d_model4 = int((d_model3 - maxpooling[1])/maxpooling[1] + 1)
        
        self.lstm = nn.LSTM(d_model4, self.d_hidden, self.num_layers, batch_first=True, dropout=self.dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        #layer1
        self.conv1_out = self.conv1(x)
        self.conv1relu_out = self.f_activation1(self.conv1_out)
        
        #layer2
        self.maxpooling1_out = self.maxpooling1(self.conv1relu_out)
        self.batchNorm1_out = self.batchNorm1(self.maxpooling1_out )
        
        self.conv2_out = self.conv2(self.maxpooling1_out)
        self.conv2relu_out = self.f_activation2(self.conv2_out)
        
        #layer3
        self.maxpooling2_out = self.maxpooling2(self.conv2relu_out)
        self.batchNorm2_out = self.batchNorm2(self.maxpooling2_out)
        
        #layer4
        h_c = self.init_hidden(len(self.batchNorm2_out))
        self.out, (_, _) = self.lstm(self.batchNorm2_out, h_c)
        return self.softmax(self.out[:, -1, :])
            
    def init_hidden(self, batch_size):
        weight = next(self.lstm.parameters()).data
        
        hidden = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()        
        cell = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()
        
        return (hidden, cell)
    def get_output_per_layer(self):
        return {
                'conv1_relu': self.conv1relu_out,
                'batchNorm1': self.batchNorm1_out,
                'maxpooling1': self.maxpooling1_out,
                'conv2_relu': self.conv2relu_out,
                'batchNorm2': self.batchNorm2_out,
                'maxpooling2': self.maxpooling2_out,
                'lstm': self.out
            }
        

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
