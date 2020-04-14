import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

# referenced from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
def jacobian(y, x):
    """Computes the Jacobian of y w.r.t x.
    :param y: function R^M -> R^M
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, M]
    """
    batch_size, x_dim = x.shape
    batch_size, y_dim = y.shape
    # jac = torch.zeros(batch_size, y_dim, x_dim, requires_grad=True)
    der = list()
    for i in range(y_dim):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                       x,
                       grad_outputs=v,
                       create_graph=True)[0]  # shape [B, N]
        # jac[:,i,:] = dy_i_dx
        # print(dy_i_dx)
        der.append(dy_i_dx)
    der = torch.stack(der, dim=2).permute(0,2,1)
    return der

class ProximalMapping(ag.Function):
    @staticmethod
    def forward(ctx, G, s, epsilon):
        # Calculate Gradient and Hessian
        ctx.epsilon = epsilon
        batch_size, y_dim, x_dim = G.shape
        gTerm = epsilon * torch.bmm(G,G.permute(0,2,1))
        I = torch.eye(y_dim)
        I = I.reshape(1, y_dim, y_dim)
        I = I.repeat(batch_size, 1, 1)
        invTerm = torch.inverse(I + gTerm)
        s = s.reshape(batch_size,y_dim,1)
        c = torch.bmm(invTerm, s)
        c = c.reshape(batch_size, y_dim)
        ctx.save_for_backward(c, invTerm, G)
        return c
    
    @staticmethod
    def backward(ctx, grad_c):
       # We get: dL/dc, (0 for the last layer) as grad_outpus
       # Need to return dL/ds and dL/dG
       # dL/ds = grad_c * (I + G*G.t())^-1 by eq 12
       # dL/dG given by eq(21)
       c, invTerm, G = ctx.saved_tensors
       epsilon = ctx.epsilon
       batch_size, y_dim, x_dim = G.shape
       gradS =  torch.bmm(invTerm, grad_c.reshape(batch_size, y_dim, 1))# eqn(12) # transposed.. equal to a term from eq(21)
       c = c.reshape(batch_size, y_dim,1)
       gradG = -1 * (torch.bmm(gradS ,c.permute(0,2,1)) + torch.bmm(c, gradS.permute(0,2,1)))
       gradG =  torch.bmm(gradG, G) # eq(21)
       gradS = gradS.reshape(batch_size, y_dim)
       return gradG, gradS, None
   
      

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
            G = jacobian(s, inputs) # pytorch handles dL/dG * dG/dv, slows computation though
            c = ProximalMapping.apply(G, s, self.epsilon) # we give c, dL/dG and dL/ds
        
        return h, c
        
        
