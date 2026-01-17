



import sys
import os

# 方式1：绝对路径（推荐，稳定）
delta_conv_root = "/home/qianhou/deltaconv"  # DeltaConv 根目录（包含deltaconv子包）
sys.path.insert(0, delta_conv_root)  # 插入到sys.path最前面，优先加载

import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch.nn as nn
# 注意：根据你的项目结构，调整导入路径
from deltaconv.models.deltanet_base import DeltaNetBase
from deltaconv.nn import MLP

# class DeltaPointPDE(torch.nn.Module):
#     def __init__(self, in_channels,
#                  conv_channels=[64, 128, 256],
#                  mlp_depth=2,
#                  embedding_size=1024,
#                  num_neighbors=20,
#                  grad_regularizer=0.001,
#                  grad_kernel_width=1):
#         """
#         PDE prediction using DeltaConv for point cloud data.
#         The output will have the same shape as the input.

#         Args:
#             in_channels (int): the number of channels provided as input (e.g., 1 for a scalar field).
#             conv_channels (list[int]): the number of output channels of each DeltaConv layer.
#             mlp_depth (int): the depth of the MLPs within each DeltaConv.
#             embedding_size (int): the size of the global feature embedding.
#             num_neighbors (int): the number of neighbors to use in DeltaConv.
#             grad_regularizer (float): the regularizer for gradient estimation in DeltaConv.
#             grad_kernel_width (float): the kernel width for gradient estimation in DeltaConv.
#         """
#         super().__init__()

#         self.in_channels = in_channels

#         # 1. 使用 DeltaNetBase 提取多尺度特征
#         self.deltanet_base = DeltaNetBase(
#             in_channels=in_channels,
#             conv_channels=conv_channels,
#             mlp_depth=mlp_depth,
#             num_neighbors=num_neighbors,
#             grad_regularizer=grad_regularizer,
#             grad_kernel_width=grad_kernel_width
#         )

#         # 2. 全局特征嵌入
#         # 输入是所有 DeltaConv 层输出特征的拼接
#         self.lin_global = MLP([sum(conv_channels), embedding_size])

#         # 3. PDE 预测头
#         # 输入是“全局特征 + 所有局部特征”的拼接
#         self.pde_head = Seq(
#             MLP([embedding_size + sum(conv_channels), 256]), Dropout(0.5),
#             MLP([256, 256]), Dropout(0.5),
#             Linear(256, 128), LeakyReLU(negative_slope=0.2),
#             Linear(128, in_channels)  # 输出通道数与输入一致
#         )

#     def forward(self, data):
#         """
#         Forward pass.

#         Args:
#             data (torch_geometric.data.Data): A Data object containing:
#                 - pos: (N, 3) Tensor of point coordinates.
#                 - x: (N, in_channels) Tensor of input features (PDE initial conditions).
#                 - batch: (N,) Tensor indicating which graph each point belongs to (for batching).

#         Returns:
#             torch.Tensor: (N, in_channels) Tensor of PDE solutions.
#         """
#         # 确保输入是 Data 对象且包含必要的属性
#         if not isinstance(data, Data) or not hasattr(data, 'pos') or not hasattr(data, 'x'):
#             raise ValueError("Input must be a Data object with 'pos' and 'x' attributes.")

#         # 1. 使用 DeltaNetBase 提取多尺度标量特征
#         conv_out = self.deltanet_base(data)

#         # 2. 拼接所有尺度的局部特征
#         local_features = torch.cat(conv_out, dim=1)  # Shape: (N, sum(conv_channels))

#         # 3. 计算全局特征
#         global_embedding_input = self.lin_global(local_features)  # Shape: (N, embedding_size)
        
#         # --- 修正的全局特征广播逻辑 ---
#         # 健壮地处理 batch 张量
#         if hasattr(data, 'batch') and data.batch is not None and data.batch.numel() > 0:
#             batch = data.batch
#         else:
#             # 默认所有点属于同一个样本
#             batch = torch.zeros(global_embedding_input.shape[0], dtype=torch.long, device=global_embedding_input.device)
            
#         # 对每个样本求平均，得到全局特征
#         global_features = global_mean_pool(global_embedding_input, batch)  # Shape: (B, embedding_size)
        
#         # 计算每个样本的点数
#         unique_batch_ids = torch.unique(batch)
#         num_points_per_sample = []
#         for bid in unique_batch_ids:
#             count = torch.sum(batch == bid).item()
#             num_points_per_sample.append(count)
        
#         # 根据每个样本的点数，将对应的全局特征重复相应次数
#         global_expanded = torch.repeat_interleave(
#             global_features, 
#             torch.tensor(num_points_per_sample, device=global_features.device), 
#             dim=0
#         )  # Shape: (N, embedding_size)
#         # --- 修正结束 ---

#         # 4. 拼接全局特征和局部特征
#         combined_features = torch.cat([global_expanded, local_features], dim=1)  # Shape: (N, embedding_size + sum(conv_channels))

#         # 5. 通过预测头得到最终结果
#         pde_output = self.pde_head(combined_features)  # Shape: (N, in_channels)

#         # 输出形状与输入特征 x 的形状一致
#         return pde_output



import sys
import os

# 方式1：绝对路径（推荐，稳定）
delta_conv_root = "/home/qianhou/deltaconv"  # DeltaConv 根目录（包含deltaconv子包）
sys.path.insert(0, delta_conv_root)  # 插入到sys.path最前面，优先加载

import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch.nn as nn
# 注意：根据你的项目结构，调整导入路径
from deltaconv.models.deltanet_base import DeltaNetBase
from deltaconv.nn import MLP

class DeltaPointPDE(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels=1,  # 新增：out_channels参数，默认1
                 conv_channels=[64, 128, 256],
                 mlp_depth=2,
                 embedding_size=1024,
                 num_neighbors=20,
                 grad_regularizer=0.001,
                 grad_kernel_width=1):
        """
        PDE prediction using DeltaConv for point cloud data.
        The output will have the same shape as the input.

        Args:
            in_channels (int): the number of channels provided as input (e.g., 1 for a scalar field).
            out_channels (int): the number of channels for output (default: 1).  # 新增注释
            conv_channels (list[int]): the number of output channels of each DeltaConv layer.
            mlp_depth (int): the depth of the MLPs within each DeltaConv.
            embedding_size (int): the size of the global feature embedding.
            num_neighbors (int): the number of neighbors to use in DeltaConv.
            grad_regularizer (float): the regularizer for gradient estimation in DeltaConv.
            grad_kernel_width (float): the kernel width for gradient estimation in DeltaConv.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels  # 新增：保存out_channels

        # 1. 使用 DeltaNetBase 提取多尺度特征
        self.deltanet_base = DeltaNetBase(
            in_channels=in_channels,
            conv_channels=conv_channels,
            mlp_depth=mlp_depth,
            num_neighbors=num_neighbors,
            grad_regularizer=grad_regularizer,
            grad_kernel_width=grad_kernel_width
        )

        # 2. 全局特征嵌入
        # 输入是所有 DeltaConv 层输出特征的拼接
        self.lin_global = MLP([sum(conv_channels), embedding_size])

        # 3. PDE 预测头（修改：输出通道数改为out_channels）
        # 输入是“全局特征 + 所有局部特征”的拼接
        self.pde_head = Seq(
            MLP([embedding_size + sum(conv_channels), 256]), Dropout(0.5),
            MLP([256, 256]), Dropout(0.5),
            Linear(256, 128), LeakyReLU(negative_slope=0.2),
            Linear(128, self.out_channels)  # 关键修改：从in_channels改为out_channels
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data (torch_geometric.data.Data): A Data object containing:
                - pos: (N, 3) Tensor of point coordinates.
                - x: (N, in_channels) Tensor of input features (PDE initial conditions).
                - batch: (N,) Tensor indicating which graph each point belongs to (for batching).

        Returns:
            torch.Tensor: (N, out_channels) Tensor of PDE solutions.
        """
        # 确保输入是 Data 对象且包含必要的属性
        if not isinstance(data, Data) or not hasattr(data, 'pos') or not hasattr(data, 'x'):
            raise ValueError("Input must be a Data object with 'pos' and 'x' attributes.")

        # 1. 使用 DeltaNetBase 提取多尺度标量特征
        conv_out = self.deltanet_base(data)

        # 2. 拼接所有尺度的局部特征
        local_features = torch.cat(conv_out, dim=1)  # Shape: (N, sum(conv_channels))

        # 3. 计算全局特征
        global_embedding_input = self.lin_global(local_features)  # Shape: (N, embedding_size)
        
        # --- 修正的全局特征广播逻辑 ---
        # 健壮地处理 batch 张量
        if hasattr(data, 'batch') and data.batch is not None and data.batch.numel() > 0:
            batch = data.batch
        else:
            # 默认所有点属于同一个样本
            batch = torch.zeros(global_embedding_input.shape[0], dtype=torch.long, device=global_embedding_input.device)
            
        # 对每个样本求平均，得到全局特征
        global_features = global_mean_pool(global_embedding_input, batch)  # Shape: (B, embedding_size)
        
        # 计算每个样本的点数
        unique_batch_ids = torch.unique(batch)
        num_points_per_sample = []
        for bid in unique_batch_ids:
            count = torch.sum(batch == bid).item()
            num_points_per_sample.append(count)
        
        # 根据每个样本的点数，将对应的全局特征重复相应次数
        global_expanded = torch.repeat_interleave(
            global_features, 
            torch.tensor(num_points_per_sample, device=global_features.device), 
            dim=0
        )  # Shape: (N, embedding_size)
        # --- 修正结束 ---

        # 4. 拼接全局特征和局部特征
        combined_features = torch.cat([global_expanded, local_features], dim=1)  # Shape: (N, embedding_size + sum(conv_channels))

        # 5. 通过预测头得到最终结果
        pde_output = self.pde_head(combined_features)  # Shape: (N, out_channels)

        # 输出形状与out_channels一致
        return pde_output





################################################### SAGEConv ############################################
# import torch
# from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
# from torch_geometric.nn import GCNConv, global_mean_pool, knn_graph
# from torch_geometric.data import Data
# import torch.nn as nn
# from deltaconv.nn import MLP  # 仍可使用 deltaconv 的 MLP
# from torch_geometric.nn import SAGEConv

# class SAGEFeatureExtractor(torch.nn.Module):
#     def __init__(self, in_channels, conv_channels, num_neighbors):
#         super().__init__()
#         self.num_neighbors = num_neighbors
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, conv_channels[0], aggr='mean'))
#         for i in range(1, len(conv_channels)):
#             self.convs.append(SAGEConv(conv_channels[i-1], conv_channels[i], aggr='mean'))

#     def forward(self, data):
#         x, pos, batch = data.x, data.pos, data.batch
#         edge_index = knn_graph(pos, self.num_neighbors, batch, loop=True, flow='target_to_source')
#         features = []
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = LeakyReLU(negative_slope=0.2)(x)
#             features.append(x)
#         return features

# class DeltaPointPDE(torch.nn.Module):
#     def __init__(self, in_channels,
#                  conv_channels=[64, 128, 256],
#                  mlp_depth=2,
#                  embedding_size=1024,
#                  num_neighbors=20,
#                  grad_regularizer=0.001,  # GCN 不需要这些参数，仅为兼容保留
#                  grad_kernel_width=1):
#         super().__init__()

#         self.in_channels = in_channels

#         # 1. 替换为 GCN 特征提取器
#         self.feature_extractor = SAGEFeatureExtractor(
#             in_channels=in_channels,
#             conv_channels=conv_channels,
#             num_neighbors=num_neighbors
#         )

#         # 2. 全局特征嵌入（与原逻辑一致）
#         self.lin_global = MLP([sum(conv_channels), embedding_size])

#         # 3. PDE 预测头（与原逻辑一致）
#         self.pde_head = Seq(
#             MLP([embedding_size + sum(conv_channels), 256]), Dropout(0.5),
#             MLP([256, 256]), Dropout(0.5),
#             Linear(256, 128), LeakyReLU(negative_slope=0.2),
#             Linear(128, in_channels)
#         )

#     def forward(self, data):
#         # 确保输入是 Data 对象且包含必要的属性
#         if not isinstance(data, Data) or not hasattr(data, 'pos') or not hasattr(data, 'x'):
#             raise ValueError("Input must be a Data object with 'pos' and 'x' attributes.")

#         # 1. 使用 GCN 提取多尺度特征
#         conv_out = self.feature_extractor(data)  # 与 DeltaNetBase 输出格式一致

#         # 2. 拼接所有尺度的局部特征
#         local_features = torch.cat(conv_out, dim=1)  # Shape: (N, sum(conv_channels))

#         # 3. 计算全局特征（与原逻辑一致）
#         global_embedding_input = self.lin_global(local_features)
#         batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(local_features.shape[0], dtype=torch.long, device=local_features.device)
#         global_features = global_mean_pool(global_embedding_input, batch)

#         # 4. 广播全局特征到每个点
#         unique_batch_ids = torch.unique(batch)
#         num_points_per_sample = [torch.sum(batch == bid).item() for bid in unique_batch_ids]
#         global_expanded = torch.repeat_interleave(
#             global_features, torch.tensor(num_points_per_sample, device=global_features.device), dim=0
#         )

#         # 5. 拼接全局特征和局部特征
#         combined_features = torch.cat([global_expanded, local_features], dim=1)

#         # 6. 通过预测头得到最终结果
#         pde_output = self.pde_head(combined_features)

#         return pde_output