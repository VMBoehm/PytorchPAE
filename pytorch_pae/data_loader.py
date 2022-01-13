from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_data(data, loc, batchsize):
    
    if data in dir(datasets):
        dataset = getattr(datasets,data)
    
        training_data = dataset(root=loc,train=True,download=True,transform=ToTensor())

        valid_data    = dataset(root=loc,train=False,download=True,transform=ToTensor())
    else:
        pass
    
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=256, shuffle=True)
    
    return train_dataloader, valid_dataloader