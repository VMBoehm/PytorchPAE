import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from torchvision.transforms import ToTensor
import numpy as np
import pickle
import os

class SDSS_DR16(Dataset):
    """De-redshifted and downsampled spectra from SDSS-BOSS DR16"""

    def __init__(self, root_dir='/global/cscratch1/sd/vboehm/Datasets/sdss/by_model/', transform=None, train=True):
        """
        Args:
            root_dir (string): Directory of data file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        if train:
            self.data = pickle.load(open(os.path.join(root_dir,'SDSS_DR16_preprocessed_train.pkl'),'rb'))
        else:
            self.data = pickle.load(open(os.path.join(root_dir,'SDSS_DR16_preprocessed_test.pkl'),'rb'))
            
        self.length           = len(self.data['spec'])
        print(self.data['spec'].shape, self.length)
            
        self.data['features'] = np.swapaxes(self.data['spec'],2,1)
        self.data['mask']     = np.swapaxes(self.data['mask'],2,1)
        self.data['noise']    = np.swapaxes(self.data['noise'],2,1)
        
        print(self.data['features'].shape)
        
        del self.data['mean']
        del self.data['std']
        del self.data['SN']
        del self.data['spec']
        
        self.keys      = list(self.data.keys())
        
        self.transform = transform


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {key: torch.as_tensor(self.data[key][idx]).float() for key in self.keys}
        
        if self.transform != None:
            sample = self.transform(sample['features'])
       
        return sample
    
    
    
class SDSS_DR16_simple(Dataset):
    """De-redshifted and downsampled spectra from SDSS-BOSS DR16"""

    def __init__(self, root_dir='drive/MyDrive/ML_lecture_data/', transform=True, train=True):
        """
        Args:
            root_dir (string): Directory of data file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if train:
            self.data = np.load(open(os.path.join(root_dir,'DR16_denoised_inpainted_train.npy'),'rb'),allow_pickle=True)
        else:
            self.data = np.load(open(os.path.join(root_dir,'DR16_denoised_inpainted_test.npy'),'rb'),allow_pickle=True)
        self.data = torch.as_tensor(self.data)
        self.mean = torch.mean(self.data)
        self.std  = torch.std(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data[idx]-self.mean)/self.std
        
        if self.transform != None:
            sample = self.transform(sample)

        return sample

    
class SDSS_DR16_small(Dataset):
    """De-redshifted and downsampled spectra from SDSS-BOSS DR16, this time with noise and everything"""

    def __init__(self, root_dir='drive/MyDrive/ML_lecture_data/', transform=True, train=True):
        """
        Args:
            root_dir (string): Directory of data file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if train:
            self.data = np.load(open(os.path.join(root_dir,'DR16_train.npy'),'rb'),allow_pickle=True)
        else:
            self.data = np.load(open(os.path.join(root_dir,'DR16_test.npy'),'rb'),allow_pickle=True)
            
        self.length           = len(self.data['spec'])
            
        self.data['features'] = np.swapaxes(self.data['spec'],2,1)
        self.data['mask']     = np.swapaxes(self.data['mask'],2,1)
        self.data['noise']    = np.swapaxes(self.data['noise'],2,1)
        
        self.keys      = list(self.data.keys())
        
        self.transform = transform
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {key: torch.as_tensor(self.data[key][idx]).float() for key in self.keys}
        
        if self.transform != None:
            sample = self.transform(sample['features'])

        return sample