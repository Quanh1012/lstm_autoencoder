#import torch 
import torch.nn as nn


class Lstm_autoencoder(nn.Module):
    def __init__(self, d_model, d_hidden, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = Encoder(self.d_model, self.d_hidden, self.num_layers, self.dropout)
        self.decoder = Decoder(self.d_model, self.d_hidden, self.num_layers, self.dropout)
        
    def forward(self, input):
        h1, h2, h3 = self.encoder.init_hidden(self.d_hidden)
        h4, h5 = self.decoder.init_hidden(self.d_hidden)
        output = self.decoder(self.encoder(input, h1, h2, h3), h4, h5)
        return output
            
            
class Encoder(nn.Module):
    def __init__(self, d_model, d_hidden, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm1 = nn.LSTM(self.d_model, self.d_hidden*8, self.num_layers, dropout = self.dropout)
        self.lstm2 = nn.LSTM(self.d_hidden*8, self.d_hidden*4, self.num_layers, dropout= self.dropout)
        self.lstm3 = nn.LSTM(self.d_hidden*4, self.d_hidden, self.num_layer, dropout = self.dropout)
    
    def forward(self, input, hidden1, hidden2, hidden3):
        _, (self.h1, _) = self.lstm1(input, hidden1)
        _, (self.h2, _) = self.lstm2(self.h1, hidden2)
        _, (self.h3, _) = self.lstm3(self.h3, hidden3)
        return self.h3
        
    def init_hidden(self, d_hidden):
        weight = next(self.parameters()).data
        
        hidden1 = weight.new(self.n_layers, d_hidden*8).zero_()
        hidden2 = weight.new(self.n_layers, d_hidden*4).zero_()
        hidden3 = weight.new(self.n_layers, d_hidden).zero_()
        
        cell1 = weight.new(self.n_layers, d_hidden*8).zero_()
        cell2 = weight.new(self.n_layers, d_hidden*4).zero_()
        cell3 = weight.new(self.n_layers, d_hidden).zero_()
        
        return (hidden1, cell1), (hidden2, cell2), (hidden3, cell3)
        
        
class Decoder(nn.Module):
    def __init__(self, d_model, d_hidden, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm4 = nn.LSTM(self.d_hidden*4, self.d_hidden*8, self.num_layers, dropout= self.dropout)
        self.lstm5 = nn.LSTM(self.d_hidden*8, self.d_model, self.num_layers, dropout = self.dropout)
    
    def forward(self, input, hidden1, hidden2):
        _, (self.h1, _) = self.lstm4(input, hidden1)
        _, (self.h2, _) = self.lstm5(self.h1, hidden2)
        return self.h2
    
    def init_hidden(self, d_hidden):
        weight = next(self.parameters()).data
        hidden1 = weight.new(self.n_layers, d_hidden*4).zero_()
        hidden2 = weight.new(self.n_layers, d_hidden).zero_()
        
        cell1 = weight.new(self.n_layers, d_hidden*4).zero_()
        cell2 = weight.new(self.n_layers, d_hidden).zero_()
        
        return (hidden1, cell1), (hidden2, cell2)
    
    
class Fully_connected(nn.Module):
    def __init__(self, d_model, d_hidden, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.d_model, 3),   
            nn.Softmax(1)    
        )
    def forward(self, input):
        self.output = self.fully_connected(input)
        return self.output
        
        