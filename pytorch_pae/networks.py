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
import torch.nn.functional as F
import torch.optim as optim

import pytorch_pae.utils as utils

class FCEncoder(nn.Module):
    def __init__(self, params, nparams):
        super(FCEncoder, self).__init__()
        if params['dim'] == '1D':
            self.N    = 1
        elif params['dim'] == '2D':
            self.N    = 2
        else:
            raise Exception("Invalid data dimensionality (must be either 1D or 2D).")
            
            
        if nparams['spec_norm']:
            spec_norm = nn.utils.spectral_norm
        else:
            spec_norm = nn.Identity()
            
        self.model = nn.ModuleList()
    
        self.model.append(nn.Flatten())
        
        current_dim = params['input_dim']**self.N*params['input_c']
        
        for ii in range(nparams['n_layers']):
            
            lin = nn.Linear(current_dim, nparams['out_sizes'][ii],bias=nparams['bias'][ii])
            self.model.append(spec_norm(lin))
            
            current_dim      =  nparams['out_sizes'][ii]
            
            if nparams['layer_norm'][ii]:
                norm = nn.LayerNorm(current_dim,elementwise_affine=nparams['affine'])
                self.model.append(norm)
            
            gate = getattr(nn, nparams['activations'][ii])()
            self.model.append(gate)
            
            dropout = nn.Dropout(nparams['dropout_rate'][ii])
            self.model.append(dropout)
        
        lin = nn.Linear(current_dim,params['latent_dim'])
        self.model.append(spec_norm(lin))
        
        if params['contrastive']:
            self.g = g_network(params['latent_dim'],params['hidden_dim'])
            
    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return x
        
        
class ConvEncoder(nn.Module):
    def __init__(self, params, nparams):
        super(ConvEncoder, self).__init__()
        if params['dim'] == '1D':
            self.conv = nn.Conv1d
            self.pool = nn.AdaptiveMaxPool1d
            self.N    = 1
        elif params['dim'] == '2D':
            self.conv = nn.Conv2d
            self.pool = nn.AdaptiveMaxPool2d
            self.N    = 2
        else:
            raise Exception("Invalid data dimensionality (must be either 1D or 2D).")
            
        if nparams['spec_norm']:
            spec_norm = nn.utils.spectral_norm
        else:
            spec_norm = nn.Identity()
            
        self.model = nn.ModuleList()
    
        current_channels   = params['input_c']
        current_dim        = params['input_dim']
        self.out_dims      = []
        
        for ii in range(nparams['n_layers']):
            
            conv = self.conv(current_channels, nparams['out_channels'][ii], nparams['kernel_sizes'][ii], nparams['strides'][ii], nparams['paddings'][ii], bias=nparams['bias'][ii])
            self.out_dims.append(current_dim)
            self.model.append(spec_norm(conv))
            
            current_channels =  nparams['out_channels'][ii]
            current_dim      =  utils.output_shape(current_dim, nparams['strides'][ii], nparams['paddings'][ii],nparams['kernel_sizes'][ii])
            
            if nparams['layer_norm'][ii]:
                norm = nn.LayerNorm([current_channels]+[current_dim]*self.N,elementwise_affine=nparams['affine'])
                self.model.append(norm)
                
            gate = getattr(nn, nparams['activations'][ii])()
            self.model.append(gate)
            
            pool = self.pool([current_dim//nparams['scale_facs'][ii]]*self.N)
            self.model.append(pool)
            
            current_dim = current_dim//nparams['scale_facs'][ii]

        self.final_dim = current_dim
        self.final_c   = current_channels
        
        self.model.append(nn.Flatten())
        current_shape = current_channels*current_dim**self.N
        linear        = nn.Linear(current_shape,params['latent_dim'])
        self.model.append(spec_norm(linear))
        
        if params['contrastive']:
            self.g = g_network(params['latent_dim'],params['hidden_dim'])
            
            
    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, params, nparams):
        super(ConvDecoder, self).__init__()
        
        if params['dim'] == '1D':
            self.conv = nn.ConvTranspose1d
            self.N    = 1
        elif params['dim'] == '2D':
            self.conv = nn.ConvTranspose2d
            self.N    = 2
        else:
            raise Exception("Invalid data dimensionality (must be either 1D or 2D).")
            
        if nparams['spec_norm']:
            spec_norm = nn.utils.spectral_norm
        else:
            spec_norm = nn.Identity()
        
        self.pool   = nn.Upsample
        
        self.model  = nn.ModuleList()
            
        final_shape = nparams['final_c']*nparams['final_dim']**self.N
    
        self.model.append(nn.Flatten())
        lin         = nn.Linear(params['latent_dim'],final_shape)
        self.model.append(spec_norm(lin))

        if params['dim'] == '1D':
            self.model.append(utils.Reshape((-1, nparams['final_c'],nparams['final_dim'])))
        else:
            self.model.append(utils.Reshape((-1, nparams['final_c'],nparams['final_dim'],nparams['final_dim'])))
                              
        current_dim      = nparams['final_dim']
        current_channels = nparams['final_c']
            
        for jj in range(1,nparams['n_layers']+1):
            ii = nparams['n_layers'] - jj 
            gate = getattr(nn, nparams['activations'][ii])()
            self.model.append(gate)
                  
            upsample    = nn.Upsample(scale_factor=nparams['scale_facs'][ii])
            self.model.append(upsample)
            current_dim = current_dim*nparams['scale_facs'][ii]
                              
            output_padding = utils.get_output_padding(current_dim,nparams['out_dims'][ii],nparams['strides'][ii],nparams['paddings'][ii],nparams['kernel_sizes'][ii],dilation=1)
                
            
            conv           = self.conv(current_channels, nparams['out_channels'][ii], kernel_size=nparams['kernel_sizes'][ii], stride= nparams['strides'][ii], padding=nparams['paddings'][ii], output_padding=output_padding,bias=nparams['bias'][ii])
            
            self.model.append(spec_norm(conv))
            
            current_channels = nparams['out_channels'][ii]
            current_dim      = utils.output_shape_transpose(current_dim, stride=nparams['strides'][ii], padding=nparams['paddings'][ii],kernel=nparams['kernel_sizes'][ii],output_padding=output_padding)
                
            if nparams['layer_norm'][ii]:
                norm = nn.LayerNorm([current_channels]+[current_dim]*self.N,elementwise_affine=nparams['affine'])
                self.model.append(norm)    
                
        
        conv = self.conv(current_channels, 1, kernel_size=1, stride=1)
        self.model.append(spec_norm(conv))
        
        if nparams['final_sigmoid']: 
            self.model.append(getattr(nn, 'Sigmoid')())
        
    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return x
    
    
class FCDecoder(nn.Module):
    def __init__(self, params, nparams):
        super(FCDecoder, self).__init__()
        if params['dim'] == '1D':
            self.N    = 1
        elif params['dim'] == '2D':
            self.N    = 2
        else:
            raise Exception("Invalid data dimensionality (must be either 1D or 2D).")
            
            
        if nparams['spec_norm']:
            spec_norm = nn.utils.spectral_norm
        else:
            spec_norm = nn.Identity()
            
        self.model = nn.ModuleList()
    
        self.model.append(nn.Flatten())
        
        current_dim = params['latent_dim']
        
        for jj in range(1,nparams['n_layers']+1):
            ii = nparams['n_layers'] - jj 
            
            lin = nn.Linear(current_dim, nparams['out_sizes'][ii],bias=nparams['bias'][ii])
            self.model.append(spec_norm(lin))
            
            current_dim      =  nparams['out_sizes'][ii]
            
            if nparams['layer_norm'][ii]:
                norm = nn.LayerNorm(current_dim,elementwise_affine=nparams['affine'])
                self.model.append(norm)
                
            gate = getattr(nn, nparams['activations'][ii])()
            self.model.append(gate)
            
            dropout = nn.Dropout(nparams['dropout_rate'][ii])
            self.model.append(dropout)
        
        lin = nn.Linear(current_dim,params['input_dim']**self.N*params['input_c'])
        self.model.append(spec_norm(lin))
        #gate = getattr(nn, nparams['activations'][ii])()
        #self.model.append(gate)
        
        if nparams['final_sigmoid']: 
            self.model.append(getattr(nn, 'Sigmoid')())
        
        self.model.append(utils.Reshape([-1]+[params['input_c']]+[params['input_dim']]*self.N))
    
    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return x

    
class g_network(nn.Module):
    """
    encoder head for contrastive learning loss
    """
    def __init__(self, input_dim, output_dim):
        super(g_network, self).__init__()
        
        self.model = nn.ModuleList()
        
        self.model.append(nn.ReLU())
        lin = nn.Linear(input_dim, (input_dim+output_dim)//2)
        self.model.append(lin)
        
        self.model.append(nn.ReLU())
        lin = nn.Linear((input_dim+output_dim)//2,output_dim)
        self.model.append(lin)
        
    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return x
