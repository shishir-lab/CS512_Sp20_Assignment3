import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalMapping(ag.Function):
    @staticmethod
    def forward(ctx, inputs, s, epsilon):
        # First order Gradient
        G = ag.grad(s, inputs, create_graph=True, grad_outputs=torch.ones(s.shape))[0]
        # Handle second order derivative
        G2 = ag.grad(G, inputs, create_graph=True, grad_outputs=torch.ones(G.shape))[0]
        ctx.epsilon = epsilon
        gTerm = epsilon * G @ G.t()
        I = torch.eye(gTerm.shape[0])
        invTerm = torch.inverse(I + gTerm)
        c = invTerm @ s
        ctx.save_for_backward(c, G, G2)
        return c
    @staticmethod
    def backward(ctx, grad_outputs):
       # We get: dL/dc, (0 for the last layer) as grad_outpus
       # Need to return dL/ds and dL/dv
       # dL/ds = grad_outputs * (I + G*G.t())^-1 by eq 12
       # dL/dv = dL/dG * dG/dv 
       # dL/dG given by eq(20)
       c, G, G2 = ctx.saved_tensors
       epsilon = ctx.epsilon
       gTerm = epsilon * (G @ G.t())
       I = torch.eye(gTerm.shape[0])
       invTerm = torch.inverse(I + gTerm)
       gradS =  invTerm @ grad_outputs# eqn(12) # transposed.. equal to a term from eq(21)
       gradG = - (gradS @ c.t() + c @ gradS.t()) @ G # eq(21)
       gradInputs = gradG * G2 # G2 is Hessian (By default gradient are accumulated)
       return gradInputs, gradS, None
      

class ProximalLSTMCell(nn.Module):
    def __init__(self,lstm, epsilon):    # feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox
        self.epsilon = epsilon
    
    def forward(self, inputs, hx=None, last=False):
        """
        inputs: Input similar to LSTM
        hx: (pre_h, pre_c) from previous layers
        """
        if hx is None:
            h, s = self.lstm(inputs)
        else:
            h, s = self.lstm(inputs, hx)
            
        if last:
            c = s
        else:
            c = ProximalMapping.apply(inputs, s, self.epsilon)
        
        return h, c
        
        
