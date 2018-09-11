# -*- coding: utf-8 -*-
import numpy                                as np
import torch
import torch.nn.functional                  as TNF

#
# PyTorch Convolution Layers
#

class Conv2dBNN(torch.nn.Conv2d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot"):
        #
        # Fan-in/fan-out computation
        #
        num_inputs = in_channels
        num_units  = out_channels
        for x in kernel_size:
            num_inputs *= x
            num_units  *= x
        
        if H == "Glorot":
            self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.H          = H
        
        if W_LR_scale == "Glorot":
            self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.W_LR_scale = self.H
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        Wb = bnn_sign(self.weight/self.H)*self.H
        return TNF.conv2d(x, Wb, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)



#
# PyTorch Dense Layers
#

class LinearBNN(torch.nn.Linear):
    """
    Linear/Dense layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot"):
        #
        # Fan-in/fan-out computation
        #
        num_inputs = in_channels
        num_units  = out_channels
        
        if H == "Glorot":
            self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.H          = H
        
        if W_LR_scale == "Glorot":
            self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.W_LR_scale = self.H
        
        super().__init__(in_channels, out_channels, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, input):
        Wb = bnn_sign(self.weight/self.H)*self.H
        return TNF.linear(input, Wb, self.bias)



#
# PyTorch Non-Linearities
#

class SignBNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return bnn_sign(x)


class BNNSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # import ipdb as pdb; pdb.set_trace()
        ctx.save_for_backward(x)
        return x.sign()
    
    @staticmethod
    def backward(ctx, dx):
        # import ipdb as pdb; pdb.set_trace()
        x, = ctx.saved_variables
        gt1  = x > +1
        lsm1 = x < -1
        gi   = 1-gt1.float()-lsm1.float()
        
        return gi*dx
bnn_sign = BNNSign.apply

