import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from torchvision.transforms import ToTensor
import numpy as np
import pickle
import os

class SDSS_DR16(Dataset):
    """De-redshifted and downsampled spectra from SDSS-BOSS DR16"""

    def __init__(self, root_dir='/global/cscratch1/sd/vboehm/Datasets/sdss/by_model/', transform=True, train=True):
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
            
        self.data['features'] = np.swapaxes(self.data['spec'],2,1)
        self.data['mask']     = np.swapaxes(self.data['mask'],2,1)
        self.data['noise']    = np.swapaxes(self.data['noise'],2,1)
        
        del self.data['mean']
        del self.data['std']
        del self.data['SN']
        del self.data['spec']
        
        self.keys      = list(self.data.keys())
        
        self.transform = transform


    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            sample = {key: torch.as_tensor(self.data[key][idx]) for key in self.keys}
        else:
            sample = {key: self.data[key][idx] for key in self.keys}

        return sample