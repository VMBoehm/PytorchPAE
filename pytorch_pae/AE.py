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

import pytorch_pae.networks as nets
import pytorch_pae.custom_losses as cl
from pytorch_pae.data_loader import *


from functools import partial


class Autoencoder(nn.Module):
    def __init__(self, params, dparams, nparams_enc, nparams_dec, tparams, device):
        super(Autoencoder, self).__init__()
        
        if params['encoder_type'] == 'conv':
            self.encoder = nets.ConvEncoder(params, nparams_enc)
            nparams_enc['out_dims']  = self.encoder.out_dims
            nparams_enc['final_dim'] = self.encoder.final_dim
            nparams_enc['final_c']   = self.encoder.final_c
        elif params['encoder_type'] == 'fc':
            self.encoder = nets.FCEncoder(params, nparams_enc)
        else:
            raise Exception('invalid encoder type')
            
        if params['decoder_type'] == 'conv':
            self.decoder = nets.ConvDecoder(params, nparams_dec)
        elif params['decoder_type'] == 'fc':
            self.decoder = nets.FCDecoder(params, nparams_dec)
        else:
            raise Exception('invalid decoder type')
        
        self.optimizer = getattr(optim, tparams['optimizer'])
        self.optimizer = self.optimizer(self.parameters(),tparams['initial_lr'])
        
        self.scheduler = partial(getattr(torch.optim.lr_scheduler, tparams['scheduler']),self.optimizer)
        self.scheduler = self.scheduler(**tparams['scheduler_params'])
        
        if tparams['criterion1'] in dir(nn):
            self.criterion1 = partial(self.loss,getattr(nn, tparams['criterion1'])())
        elif tparams['criterion1'] in dir(cl):
            self.criterion1 = getattr(cl, tparams['criterion1'])
            
        if tparams['criterion2'] in dir(nn):
            self.criterion2 = partial(self.loss,getattr(nn, tparams['criterion2'])())
        elif tparams['criterion2'] in dir(cl):
            self.criterion2 = getattr(cl, tparams['criterion2'])
            
        self.train_loader, self.valid_loader = get_data(dparams['dataset'],dparams['loc'],tparams['batchsize'],256)#,tparams['batchsze_valid'], transforms)
        
        self.device = device
        
        self.to(self.device)
        
        self.params = params
        
        self.ann_epoch = tparams['ann_epoch']
        
        
    def forward(self, x):
        x = self.encoder(x)
        # if params['contrastive']:
        #     x = self.encoder.g.forward(x)
        # else:
        x = self.decoder(x)
        return x
    
    @staticmethod
    def loss(func,recon,features, data, device):
        return func(recon,features)
        
    def update_device(self,device):
        self.device= device
        self.to(self.device)
        return True
    
    def update_lr(self,lr):
    
        self.optimizer = getattr(optim, tparams['optimizer'])
        self.optimizer  = self.optimizer(self.parameters(),lr)
        
        self.scheduler = partial(getattr(torch.optim.lr_scheduler, tparams['scheduler']),self.optimizer)
        self.scheduler = self.scheduler(**tparams['scheduler_params'])
        
        return True
    
    
    def update_scheduler(self,scheduler, scheduler_params):
        
        self.scheduler = partial(getattr(torch.optim.lr_scheduler, scheduler),self.optimizer)
        self.scheduler = self.scheduler(**scheduler_params)
        
        return True
    
    
    def update_optimizer(self,optimizer):
        
        self.optimizer = getattr(optim, optimizer)
        self.optimizer  = self.optimizer(self.parameters(),tparams['initial_lr'])
        
        self.scheduler = partial(getattr(torch.optim.lr_scheduler, tparams['scheduler']),self.optimizer)
        self.scheduler = self.scheduler(**tparams['scheduler_params'])
        
        return True
    
    def train(self, nepochs):
        running_loss    = []
        validation_loss = []
        valid_loader = iter(self.valid_loader)
        for epoch in range(nepochs):
            r_loss = 0
            for ii, data in enumerate(self.train_loader,0):
                if isinstance(data,dict):
                    features  = data['features'].to(self.device).float()
                else:
                    features  = data[0].to(self.device)
                recon = self.forward(features)
                
                if epoch<self.ann_epoch:
                    loss  = self.criterion1(recon, features, data, self.device)
                else:
                    loss  = self.criterion2(recon, features, data, self.device)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                r_loss+=loss.item()
            self.scheduler.step()
            running_loss.append(r_loss/ii)
            try:
                valid_data    = next(valid_loader)
            except:
                valid_loader  = iter(self.valid_loader)
                valid_data    = next(valid_loader)
            if isinstance(data,dict):
                features  = valid_data['features'].to(self.device).float()
            else:
                features  = valid_data[0].to(self.device)
            with torch.no_grad():
                recon      = self.forward(features)
                if epoch<self.ann_epoch:
                    loss       = self.criterion1(recon, features, valid_data, self.device)
                else:
                    loss       = self.criterion2(recon, features, valid_data, self.device)
            validation_loss.append(loss.item())
            print(f'epoch: {epoch:d}, training loss: {running_loss[-1]:.4e}, validation loss: {loss:.4e}, learning rate: {self.scheduler.get_last_lr()[0]:.4e}')
        return running_loss, validation_loss
