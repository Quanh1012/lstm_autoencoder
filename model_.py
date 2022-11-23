import torch.nn as nn
import torch
class Conv_gru(nn.Module):
    def __init__(self, shape_data, num_filters, kernel_size, maxpooling, d_hidden, num_layers, dropout=0.1):
        super().__init__()
        self.shape_data = shape_data
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # self.strides = strides
        self.maxpooling = maxpooling
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        # input -> conv2d -> batchnorm -> maxpool2d ->conv2d -> batchnorm -> maxpool 
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv1 = nn.Conv2d(
                            in_channels=self.shape_data[0], 
                            out_channels=self.num_filters[0], 
                            kernel_size=self.kernel_size[0],
                            )

        self.conv2 = nn.Conv2d(
                            in_channels=self.num_filters[0], 
                            out_channels=self.num_filters[1], 
                            kernel_size=self.kernel_size[1],
                            )
        # self.conv3 = nn.Conv2d(
        #                     in_channels=self.num_filters[1], 
        #                     out_channels=1, 
        #                     kernel_size=3,
        #                     )
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.zeros_(self.conv1.bias)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.zeros_(self.conv2.bias)

        self.f_activation1 = nn.ReLU()
        self.f_activation2 = nn.ReLU()


        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batchNorm1 = nn.BatchNorm2d(num_features=self.num_filters[0])
        self.batchNorm2 = nn.BatchNorm2d(num_features=self.num_filters[1])

        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        self.maxpooling1 = nn.MaxPool2d(kernel_size=self.maxpooling[0])
        self.maxpooling2 = nn.MaxPool2d(kernel_size=self.maxpooling[1])

    
        self.h_conv1 = int((self.shape_data[1] - self.kernel_size[0]) + 1)
        self.w_conv1 = int((self.shape_data[2] - self.kernel_size[0]) + 1)

        self.h_maxpool1 = int((self.h_conv1 - self.maxpooling[0])/self.maxpooling[0] + 1)
        self.w_maxpool1 = int((self.w_conv1 - self.maxpooling[0])/self.maxpooling[0] + 1)  

        self.h_conv2 = int((self.h_maxpool1 - self.kernel_size[1]) + 1)
        self.w_conv2 = int((self.w_maxpool1 - self.kernel_size[1]) + 1)

        self.h_maxpool2 = int((self.h_conv2 - self.maxpooling[1])/self.maxpooling[1] + 1)
        self.w_maxpool2 = int((self.w_conv2 - self.maxpooling[1])/self.maxpooling[1] + 1)  

        # self.h_conv3 = int((self.h_maxpool2 - 3) + 1)
        # self.w_conv3 = int((self.w_maxpool2 - 3) + 1)
        
        self.gru = nn.GRU(self.w_maxpool2, self.d_hidden, self.num_layers, batch_first=True, dropout=self.dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        #layer1
        self.conv1_out = self.conv1(x)
        self.conv1relu_out = self.f_activation1(self.conv1_out)
        
        #layer2
        
        self.batchNorm1_out = self.batchNorm1(self.conv1relu_out)
        self.maxpooling1_out = self.maxpooling1(self.batchNorm1_out)
        
        self.conv2_out = self.conv2(self.maxpooling1_out)
        self.conv2relu_out = self.f_activation2(self.conv2_out)
        
        #layer3
        self.batchNorm2_out = self.batchNorm2(self.conv2relu_out)
        self.maxpooling2_out = self.maxpooling2(self.batchNorm2_out)
        # self.outcnn = self.conv3(self.maxpooling2_out)

        # # 
        # self.outcnn = torch.squeeze(self.outcnn, dim=1) 
        self.outcnn = torch.sum(self.maxpooling2_out, dim=1)/self.num_filters[1] #(batch, h, w)
        # self.outcnn = torch.flatten(self.maxpooling2_out, start_dim=1, end_dim=2)
        
        #layer4
        h_c = self.init_hidden(len(self.outcnn))
        self.out, _ = self.gru(self.outcnn, h_c)
        return self.softmax(self.out[:, -1, :])
            
    def init_hidden(self, batch_size):
        self.batch = batch_size
        weight = next(self.gru.parameters()).data
        
        hidden = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()        
        # cell = weight.new(self.num_layers, batch_size, self.d_hidden).zero_()
        
        return hidden #, cell)
        
    def get_output_per_layer(self):
        return {
                'conv1_relu': self.conv1relu_out,
                'batchNorm1': self.batchNorm1_out,
                'maxpooling1': self.maxpooling1_out,
                'conv2_relu': self.conv2relu_out,
                'batchNorm2': self.batchNorm2_out,
                'maxpooling2': self.maxpooling2_out,
                'gru': self.out
            }
    def get_shape(self):
        print('Input: [{}, {}, {}]'.format(
            self.batch,
            self.shape_data[0],
            self.shape_data[1],
        )
        )
        print('Conv1|Batch_norm1: [{}, {}, {}, {}] | Maxpool1: [{}, {}, {}, {}]'.format(
            self.batch,
            self.num_filters[0],
            self.h_conv1,
            self.w_conv1,
            self.batch,
            self.num_filters[0],
            self.h_maxpool1,
            self.w_maxpool1)
            )
        print('Conv2|Batch_norm1: [{}, {}, {}, {}] | Maxpool2: [{}, {}, {}, {}]'.format(
            self.batch,
            self.num_filters[1],
            self.h_conv2,
            self.w_conv2,
            self.batch,
            self.num_filters[0],
            self.h_maxpool2,
            self.w_maxpool2)
            )
        print('Custom layer: [{}, {}, {}]'.format(
            self.batch,
            self.h_maxpool2,
            self.w_maxpool2)
            )
        print('Output: [{}, {}]'.format(
            self.batch,
            self.d_hidden)
            )
        

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
