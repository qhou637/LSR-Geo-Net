
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split


class SWEDatasetInpTar(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        data: Tensor of shape [N, 2, 1, nlat, nlon]
        """
        assert isinstance(data, torch.Tensor), "Input must be a tensor"
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        inp = self.data[idx, 0]  # [3, nlat, nlon]
        tar = self.data[idx, 1]  # [3, nlat, nlon]
        return inp, tar


def train_val_dataset(dataset, seed=42, test_split=0.2, val_split=0.2,max_samples=None):
  

    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))


    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=test_split, random_state=seed, shuffle=True
    )
    

    train_tem = Subset(dataset, train_idx)
    

    train_sub_idx, val_sub_idx = train_test_split(
        list(range(len(train_tem))), test_size=val_split, random_state=seed, shuffle=True
    )
    

    datasets = {
        'train': Subset(train_tem, train_sub_idx),
        'val': Subset(train_tem, val_sub_idx),
        'test': Subset(dataset, test_idx),
    }
    
    return datasets

