"""
Copyright 2022 Vanessa Boehm

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

import numpy as np

## computes output shape for both convolutions and pooling layers
def output_shape(in_dim,stride,padding,kernel,dilation=1):
    out_dim = np.floor((in_dim + 2*padding - dilation*(kernel-1)-1)/stride+1).astype(int)
    return out_dim

def output_shape_transpose(in_dim,stride,padding,kernel,output_padding, dilation=1):
    out_dim = (in_dim-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1
    return out_dim

def get_dilation(out_dim,in_dim,stride,padding,kernel,output_padding):
    dilation = np.floor((out_dim-(in_dim-1)*stride+2*padding-output_padding-1)/(kernel-1))
    new_dim  = output_shape_transpose(in_dim,stride,padding,kernel,output_padding, dilation)
    if new_dim == out_dim:
        pass
    else:
        output_padding = (out_dim-new_dim)
    return dilation, output_padding

def get_output_padding(in_dim,out_dim, stride,padding,kernel,dilation=1):
    return out_dim-(in_dim-1)*stride+2*padding-dilation*(kernel-1)-1


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
