import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import os
import sys
from datetime import datetime
import h5py as h5

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.in_size      = in_size
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()   #nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)

        o1 = self.norm(torch.matmul(x,mat1)+self.bias1)
        o2 = self.act(o1)
        o3 = torch.matmul(o1,mat2) + self.bias2 + x
        o4 = self.act3(o3)
        return o4

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

class ResTRF(nn.Module):
    def __init__(self, input_dim, output_dim, int_dim_res, int_dim_trf, N_channels):
        super(ResTRF, self).__init__()  
        layers = []

        layers.append(nn.Linear(input_dim, int_dim_res))
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(nn.Linear(int_dim_res, int_dim_trf))
        layers.append(Attention(int_dim_trf, N_channels))
        layers.append(Transformer(int_dim_trf, N_channels))
        layers.append(Attention(int_dim_trf, N_channels))
        layers.append(Transformer(int_dim_trf, N_channels))
        layers.append(Attention(int_dim_trf, N_channels))
        layers.append(Transformer(int_dim_trf, N_channels))
        layers.append(nn.Linear(int_dim_trf,output_dim))
        layers.append(Affine())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

class ResCNN(nn.Module):
    # NOTE that this is different than the one in CMB
    def __init__(self, input_dim, output_dim, int_dim, cnn_dim, kernel_size):
        super(ResCNN, self).__init__()
        assert kernel_size%2 != 0 # kernel size should be odd number so that padding is an int
        self.cnn_dim = cnn_dim   # 1dCNN width needs to be visible by forward pass

        self.input_layer = nn.Linear(input_dim, int_dim)
        
        # Def: By design, a pure block has the equal input and output dimension
        self.Res1 = ResBlock(int_dim, int_dim)
        self.Res2 = ResBlock(int_dim, int_dim)
        self.Res3 = ResBlock(int_dim, int_dim)

        self.Act1 = activation_fcn(int_dim)
        self.Act2 = activation_fcn(int_dim)
        self.Act3 = activation_fcn(int_dim)

        # Transform from the ResMLP dimension to the CNN dimension
        self.CNN_transform = nn.Linear(int_dim, cnn_dim)

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, 
            kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2)

        self.Act4 = activation_fcn(cnn_dim)


        # Def: the transformation from the internal dimension to the output 
        #      dimension of the data vector we intend to emulate
        self.out_layer = nn.Linear(cnn_dim, output_dim)
        self.norm = Affine()

        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model

    def forward(self, x):
        #x is a cosmological and nuisance parameter set you feed in the model
        x = self.input_layer(x)
        x = self.Res1(x)
        x = self.Act1(x)
        x = self.Res2(x)
        x = self.Act2(x)
        x = self.Res3(x)
        x = self.Act3(x)
        x = self.CNN_transform(x)
        x = x.view(x.size(0), 1, -1)
        x = self.conv(x)
        x = x.view(x.size(0), self.cnn_dim)
        x = self.Act4(x)        
        x = self.out_layer(x)
        out = self.norm(x)
        return out
