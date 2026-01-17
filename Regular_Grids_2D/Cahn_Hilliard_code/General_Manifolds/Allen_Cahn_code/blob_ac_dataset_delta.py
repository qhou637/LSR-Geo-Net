

import torch
import os
import h5py
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
import random
import pdb

def create_dataset(mat_file_path, num_samples=None):
    """
    从 .mat 文件创建 PyTorch Geometric Data 对象的列表。
    每个 Data 对象代表一个完整的样本，包含其所有时间步的数据。

    - data.x: 形状为 [num_points, num_time_steps]，包含每个点在所有时间步的 u 值。
    - data.pos: 形状为 [num_points, 3]，包含每个点的固定坐标。
    """
    dataset = []

    print(f"[*] Loading data from {mat_file_path}...")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            history_all = f['save_data']['history_all'][()]
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        raise

    # history_all 在 Python 中的形状: (4, num_points, num_time_steps, num_samples)
    print(f"[*] Raw data shape in Python: {history_all.shape}")

    num_quantities, num_points, num_time_steps, total_samples = history_all.shape
    # pdb.set_trace()

    if num_time_steps < 2:
        raise ValueError("Dataset must have at least 2 time steps.")

    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)

    print(f"[*] Creating dataset with {num_samples} samples, each containing {num_time_steps} time steps...")

    for sample_idx in range(num_samples):
        # --- 修正索引 ---
        # 1. 提取这个样本的所有数据 (x, y, z, u)
        #    结果形状: (4, num_points, num_time_steps)
        sample_data = history_all[:, :, :, sample_idx]

        # 2. 提取 u 值 (物理量索引为 3)
        #    结果形状: (num_points, num_time_steps)
        u_all_time_steps = sample_data[3, :, :]

        # 3. 提取点坐标 (使用 t=0 的坐标，物理量索引为 0, 1, 2)
        #    结果形状: (3, num_points)
        pos_data = sample_data[0:3, :, 0]
        #    转置为 (num_points, 3) 以符合 PyTorch Geometric 的要求
        pos = pos_data.T

        # 转换为 PyTorch 张量
        x = torch.tensor(u_all_time_steps, dtype=torch.float)  # 形状: [num_points, num_time_steps]
        pos = torch.tensor(pos, dtype=torch.float)             # 形状: [num_points, 3]

        data = Data(x=x, pos=pos)
        dataset.append(data)

        # 调试：打印第一个样本的信息
        if sample_idx == 0:
            print(f"[DEBUG] First sample - u_all_time_steps shape: {u_all_time_steps.shape}")
            print(f"[DEBUG] First sample - pos shape: {pos.shape}")

    print(f"[*] Dataset created with {len(dataset)} Data objects.")
    # pdb.set_trace()
    return dataset


# def get_train_val_test_loaders(mat_file_path, batch_size=1, train_split=0.8, val_split=0.1, test_split=0.1, num_samples=None, seed=999):

def get_train_val_test_loaders(
    mat_file_path, 
    train_batch_size=1, 
    val_batch_size=1, 
    test_batch_size=1, 
    train_split=0.8, 
    val_split=0.1, 
    test_split=0.1, 
    num_samples=None, 
    seed=999
):
    """
    创建并返回训练、验证和测试集的 DataLoader。
    通过设置 seed，可以保证每次数据划分的结果一致，便于对比实验。

    Args:
        ... (其他参数同上)
        seed (int, optional): 随机种子。设置为一个固定的整数，可以保证数据划分的可复现性。
                              默认值为 999，这是一个在编程中常用的“魔法数字”。

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # --- 设置随机种子以保证数据划分的可复现性 ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为了更彻底，还可以设置cuDNN的 deterministic 模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[*] Using random seed for data splitting: {seed}")
    # --- 种子设置结束 ---

    full_dataset = create_dataset(mat_file_path, num_samples)

    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # 健壮性检查，确保数据集大小不为零
    if test_size <= 0:
        test_size = 1
        if val_size > total_size - train_size - test_size:
             val_size = total_size - train_size - test_size
    if val_size <= 0:
        val_size = 1
        train_size = total_size - val_size - test_size

    # 现在，random_split 的结果将由上面设置的 seed 决定
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )

    print(f"[*] Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader