
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from ProxLSTM import ProximalLSTMCell


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, input_size,
                 embedding_dim=64, kernel_size=3, stride=3, epsilon=0.1):
        super(LSTMClassifier, self).__init__()
        

        self.output_size = output_size    # should be 9
        self.hidden_size = hidden_size  #the dimension of the LSTM output layer
        self.input_size = input_size      # should be 12
        self.normalize = F.normalize
        self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= embedding_dim, kernel_size= kernel_size, stride= stride) # feel free to change out_channels, kernel_size, stride
        self.relu = nn.ReLU()
        self.lstm = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.eplison = epsilon
        self.proxLSTM = ProximalLSTMCell(self.lstm, self.eplison)
        # self.h_0 = torch.nn.parameter.Parameter(torch.rand(batch_size, self.hidden_size), requires_grad=True)
        # self.c_0 = torch.nn.parameter.Parameter(torch.rand(batch_size, self.hidden_size), requires_grad=True)



        
    def forward(self, inputs, r, batch_size, mode='plain'):
        # do the forward pass
        # pay attention to the order of input dimension.
        # inputs now is of dimension: batch_size * sequence_length * input_size


        '''need to be implemented'''
        if mode == 'plain':
            # chain up the layers
            normalized = self.normalize(inputs, dim=2) # normalize the inputs
            # change dimension to batch_size x channel_size(12) x sequence_length
            normalized = normalized.permute(0,2,1) 
            embedding = self.conv(normalized) # conv layer
            activated = self.relu(embedding) # relu activation
            _,_,num_seq = activated.shape #LSTM layers
            for i in range(num_seq):
                if i == 0:
                    hi, ci = self.lstm(activated[:,:,i])#, (self.h_0, self.c_0)) 
                else:
                    hi, ci = self.lstm(activated[:,:,i], (hi,ci))
                    
            fullyconnected = self.linear(hi) #fully connected from last LSTM layer
            return fullyconnected
                
        if mode == 'AdvLSTM':
            normalized = self.normalize(inputs, dim=2) # normalize the inputs
            # change dimension to batch_size x channel_size(12) x sequence_length
            normalized = normalized.permute(0,2,1) 
            embedding = self.conv(normalized) # conv layer
            self.lstmInput = self.relu(embedding) # relu activation
            self.lstmInput.retain_grad()
            _,_,num_seq = self.lstmInput.shape #LSTM layers
            #print(r)
            self.lstmInput = self.lstmInput + self.eplison * r
            for i in range(num_seq):
                if i == 0:
                    hi, ci = self.lstm(self.lstmInput[:,:,i])#, (self.h_0, self.c_0)) 
                else:
                    hi, ci = self.lstm(self.lstmInput[:,:,i], (hi,ci))
                    
            fullyconnected = self.linear(hi) #fully connected from last LSTM layer
            return fullyconnected

                # chain up the layers
              # different from mode='plain', you need to add r to the forward pass
              # also make sure that the chain allows computing the gradient with respect to the input of LSTM
        if mode == 'ProxLSTM':
            # chain up the layers
            with torch.enable_grad(): #enable grad explictily for using autograd.grad()
                normalized = self.normalize(inputs, dim=2) # normalize the inputs
                # change dimension to batch_size x channel_size(12) x sequence_length
                normalized = normalized.permute(0,2,1) 
                embedding = self.conv(normalized) # conv layer
                activated = self.relu(embedding) # relu activation
                _,_,num_seq = activated.shape #LSTM layers
                # activated.retain_grad()
                # activated.register_hook(print)
                for i in range(num_seq):
                    if i == 0:
                        hi, ci = self.proxLSTM(activated[:,:,i])#, (self.h_0, self.c_0)) 
                    else:
                        if i == num_seq-1:
                            hi, ci = self.proxLSTM(activated[:,:,i], (hi,ci), last=True)
                            # avoiding useless computation for last layer
                        else:
                            hi, ci = self.proxLSTM(activated[:,:,i], (hi,ci))
                fullyconnected = self.linear(hi) #fully connected from last LSTM layer
                return fullyconnected
            # pass
                # chain up layers, but use ProximalLSTMCell here
