#!/usr/bin/env python

"""
Neural network models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


####### A general feedforward cnn network with spacial invariance #######
class CNNSpaceInv(nn.Module):
    """
      Feedforward Neural Network with spacial invariance.
    """
    def __init__(self, D_cnn=1, C=4, k=3, od=1, activationf='ReLU', L=72, T=30):
        """
        D_cnn: depth, number of cnn layers.
        C: number of channels.
        k: the longer dimension of the kernel (in multiples of 5 deg)
        od: output dimenstion.
        activationf: activation function.
        L: the longer dimension of the input (in multiples of 5 deg)
        T: length in time (in multiples of 10 ms)
        """
        super().__init__()
        self.D_cnn = D_cnn
        self.C = C
        self.k = k
        self.activationf = activationf
        self.L = L

        # cnn layers
        self.cnn_layers = nn.ModuleList([])
        # first cnn layer
        self.cnn_layers.append(nn.Conv2d(T, C, (1, k), padding=(0, int((k-1)/2)), padding_mode='circular'))
        # other cnn layers
        for d in range(D_cnn-1):
            self.cnn_layers.append(nn.Conv2d(C, C, (1, k), padding=(0, int((k-1)/2)), padding_mode='circular'))
        # output layer
        self.output_layer = nn.Conv2d(C, od, (1, 1), padding=(0, 0), padding_mode='circular')
    
    def forward(self, input_data):
        outputs = input_data
        if self.activationf == 'ReLU':
            # cnn layers
            for d in range(self.D_cnn):
                outputs = F.relu(self.cnn_layers[d](outputs))
        elif self.activationf == 'LeakyReLU':
            # cnn layers
            for d in range(self.D_cnn):
                outputs = F.leaky_relu(self.cnn_layers[d](outputs))
        elif self.activationf == 'ELU':
            # cnn layers
            for d in range(self.D_cnn):
                outputs = F.elu(self.cnn_layers[d](outputs))
        # output layer
        outputs = self.output_layer(outputs)
        # sum over space
        outputs = torch.sum(outputs, (-1, -2))
                
        return outputs

    
####### Constraints #######
class general_constraint_positive(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        md = module.data
        md = md.clamp(0., 1e10)
        module.data = md


class weight_constraint_positive(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0., 1e10)
            module.weight.data = w


class bias_constraint_negative(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        if hasattr(module, 'bias'):
            w = module.bias.data
            w = w.clamp(-1e10, 0.)
            module.bias.data = w




    