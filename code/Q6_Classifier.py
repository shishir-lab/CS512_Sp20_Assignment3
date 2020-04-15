
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from ProxLSTM import ProximalLSTMCell


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, input_size,
                 embedding_dim=64, kernel_size=3, stride=3, epsilon=5):
        super(LSTMClassifier, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.normalize = F.normalize
        self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= embedding_dim, kernel_size= kernel_size, stride= stride) # feel free to change out_channels, kernel_size, stride
        self.relu = nn.ReLU()
        self.lstm = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.eplison = epsilon
        self.proxLSTM = ProximalLSTMCell(self.lstm, self.eplison)
        self.dropout = nn.Dropout2d(0.5)
        self.batchnorm1 = nn.BatchNorm1d(embedding_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, inputs, r, batch_size, mode='plain'):
        if mode == 'ProxLSTM':
            # chain up the layers
            with torch.enable_grad(): #enable grad explictily for using autograd.grad()
                normalized = self.normalize(inputs, dim=2) # normalize the inputs
                # change dimension to batch_size x channel_size(12) x sequence_length
                normalized = normalized.permute(0,2,1) 
                embedding = self.conv(normalized) # conv layer
                embedding = self.batchnorm1(embedding)
                activated = self.relu(embedding) # relu activation
                #activated = self.dropout(activated)  # adding dropout after conv layer
                _,_,num_seq = activated.shape #LSTM layers
                for i in range(num_seq):
                    if i == 0:
                        hi, ci = self.proxLSTM(activated[:,:,i])#, (self.h_0, self.c_0)) 
                    else:
                        if i == num_seq-1:
                            hi, ci = self.proxLSTM(activated[:,:,i], (hi,ci), last=True)
                            # avoiding useless computation for last layer
                        else:
                            hi, ci = self.proxLSTM(activated[:,:,i], (hi,ci))
                #hi = self.dropout(hi)  # adding dropout after prox lstm layer
                #hi = self.batchnorm2(hi)
                fullyconnected = self.linear(hi)  # fully connected from last LSTM layer
                return fullyconnected
