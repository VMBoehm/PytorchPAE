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

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import pytorch_pae.custom_datasets as cd

def get_data(data, loc, batchsize, valid_batchsize, transforms=None):
    shuffle = True
    
    if data in dir(datasets):
        dataset = getattr(datasets,data)
    
        training_data = dataset(root=loc,train=True,download=True,transform=transforms)

        valid_data    = dataset(root=loc,train=False,download=True,transform=transforms)
        
    elif data in dir(cd):
        dataset       = getattr(cd,data)
        
        training_data = dataset(root_dir=loc,train=True,transform=transforms)

        valid_data    = dataset(root_dir=loc,train=False,transform=transforms)
    else:
        raise Exception(f'Dataset {data:s} not supported at this time')
    
    if batchsize == -1:
        batchsize = training_data.__len__()
        shuffle   = False
    if valid_batchsize == -1:
        valid_batchsize = valid_data.__len__()
        shuffle   = False
 
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_data, batch_size=valid_batchsize, shuffle=shuffle)
    
    return train_dataloader, valid_dataloader


