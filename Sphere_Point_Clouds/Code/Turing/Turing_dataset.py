
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


def train_val_dataset(dataset, seed=42, test_split=0.2, val_split=0.2):
    """
    自定义数据集划分方法：
    1. 先从原始数据集中划分出 test
    2. 然后从剩下的 train_tem 划分出 train 和 val

    参数：
        dataset (Dataset): 原始数据集
        seed (int): 随机种子，确保可复现性
        test_split (float): 测试集占比
        val_split (float): 验证集占比（相对于训练集）
    
    返回：
        dict: {'train', 'val', 'test'} 三个子集
    """
    # 第一次划分：从整体中分出测试集
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=test_split, random_state=seed, shuffle=True
    )
    
    # 用临时训练集保存 train + val
    train_tem = Subset(dataset, train_idx)
    
    # 第二次划分：从 train_tem 中划分出 val
    train_sub_idx, val_sub_idx = train_test_split(
        list(range(len(train_tem))), test_size=val_split, random_state=seed, shuffle=True
    )
    
    # 构建实际子集
    datasets = {
        'train': Subset(train_tem, train_sub_idx),
        'val': Subset(train_tem, val_sub_idx),
        'test': Subset(dataset, test_idx),
    }
    
    return datasets
