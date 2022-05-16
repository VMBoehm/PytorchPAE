import torch


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views         = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    




class RandomMask(object):
    """
    adds a random mask to a data point
    """

    def __init__(self, masked_frac):
        """
        masked_frac: float, target fraction of data to be masked
        """
        assert isinstance(masked_frac,(float))
        self.masked_frac = masked_frac
        
    def __call__(self, sample):
        
        rand   = torch.rand(sample.shape,device=sample.device, requires_grad=False, dtype=torch.float32)

        mask   = torch.where(rand<self.masked_frac, False, True)


        sample = torch.where(mask,sample,torch.tensor(0, dtype=torch.float32))
        return sample
    
    
    
class RandomGaussianNoise(object):

    def __init__(self, sigma=None, amplitude=None):
        
        if sigma==None:
            self.invar     = 1./(sigma)**2
            self.amplitude = 1.
            self.constant  = True
        elif amplitude==None:
            self.invar     = 1.
            self.amplitude = amplitude
            self.constant  = False
        else:
            print('Neither noise level nor amplitude were provided. Defaulting to invar=1, amplitude=1')
            self.invar     = 1
            self.amplitude = 1
            self.constant  = True
            
    def set_invar(self,invar):
        self.invar = invar
        return True

    def __call__(self, sample):
        if isinstance(sample, dict):
            data = sample['features']
            keys = sample.keys()
        else:
            raise Exception('Data type not suppported')

        noise = torch.randn(data.shape, layout=torch.strided, device=data.device, requires_grad=False)
        
        if not self.constant:
            self.set_invar(sample['noise'])
    
        noise = torch.sqrt(1./self.invar)*noise*self.amplitude
        sample['features'] = sample['features']+noise
        
        if 'noise' in keys:
            sample['noise'] = 1./((1./self.invar*(self.amplitude)**2)+(1./sample['noise'])**2)

        return sample
    
    
class RandomRedshiftShift(object):
    
    def __init__(self, wl_bins, max_delta_z):
        assert isinstance(max_delta_z,(float))
        self.max_delta_z     = max_delta_z
        self.wl_bins         = wl_bins
        
    def __call__(self, sample):
        if isinstance(sample, dict):
            data = sample['features']
            keys = sample.keys()
        else:
            raise Exception('Data type not suppported')
            
        shift = torch.rand(data.shape[0], device=data.device, requires_grad=False)
        shift = (shift-0.5)*2*max_delta_z
        
        sample['features'] = torch.where(mask,sample['features'],0.0)
        if 'mask' in keys:
            sample['mask'] = torch.logic_or(sample['mask'],mask)
            
        return sample