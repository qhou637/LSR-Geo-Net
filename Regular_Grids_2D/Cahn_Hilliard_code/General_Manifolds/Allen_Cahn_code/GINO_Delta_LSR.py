import torch
import matplotlib.pyplot as plt
import sys
import os
import pdb
import sys
import os

# ====================== 关键：添加 neuraloperator 根目录 ======================
# 你的 neuraloperator 根目录是 /home/qianhou/neuraloperator
neuralop_root = "/home/qianhou/neuraloperator"
if neuralop_root not in sys.path:
    sys.path.insert(0, neuralop_root)  # 插入到 sys.path 最前面，优先加载
# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, '/home/qianhou/torch-harmonics')

import torch
import torch.nn as nn
from NUFFT3D_Layer import NUFFTLayerMultiChannel3D_Param 
from NUFFT3D_NoGuass_Layer import NUFFTLayer_NOGauss_Param

from functools import partial
import torch
import torch.nn.functional as F
import time
import numpy as np

# NeuralOP 相关导入
from neuralop.models.base_model import BaseModel
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import SinusoidalEmbedding
from neuralop.layers.gno_block import GNOBlock

# DeltaConv 相关导入
import torch_geometric
from delta_point import DeltaPointPDE

class GINO_DeltaConv_LSR_MultiLayer(nn.Module):
    """
    多层 DeltaConv + GINO-LSR 堆叠版本
    - 一层 = 1 个 DeltaConv + 1 个 GINO-LSR
    - 通过 num_layers 参数控制堆叠层数
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        delta_pde_params,
        num_layers=1,                  # 多层堆叠数量
        # GINO-LSR 参数
        latent_feature_channels=None,
        projection_channels=256,
        gno_coord_dim=3,
        gno_radius=0.033,
        in_gno_transform_type='linear',
        out_gno_transform_type='linear',
        gno_pos_embed_type='transformer',
        nufft_n_mesh=32,
        nufft_x_lims=(-1.0, 1.0),
        nufft_fno_channel=1,
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.tanh,
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        fno_non_linearity=F.tanh,
        use_mlp_fusion=False,
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GINO_DeltaConv_LSR(
                in_channels=in_channels,
                out_channels=out_channels,
                delta_pde_params=delta_pde_params,
                latent_feature_channels=latent_feature_channels,
                projection_channels=projection_channels,
                gno_coord_dim=gno_coord_dim,
                gno_radius=gno_radius,
                in_gno_transform_type=in_gno_transform_type,
                out_gno_transform_type=out_gno_transform_type,
                gno_pos_embed_type=gno_pos_embed_type,
                nufft_n_mesh=nufft_n_mesh,
                nufft_x_lims=nufft_x_lims,
                nufft_fno_channel=nufft_fno_channel,
                fno_hidden_channels=fno_hidden_channels,
                fno_lifting_channel_ratio=fno_lifting_channel_ratio,
                gno_embed_channels=gno_embed_channels,
                gno_embed_max_positions=gno_embed_max_positions,
                in_gno_channel_mlp_hidden_layers=in_gno_channel_mlp_hidden_layers,
                out_gno_channel_mlp_hidden_layers=out_gno_channel_mlp_hidden_layers,
                gno_channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
                gno_use_open3d=gno_use_open3d,
                gno_use_torch_scatter=gno_use_torch_scatter,
                out_gno_tanh=out_gno_tanh,
                fno_non_linearity=fno_non_linearity,
                use_mlp_fusion=use_mlp_fusion,
                **kwargs
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, pos, batch=None):
        out = x
        for layer in self.layers:
            out = layer(out, pos, batch=batch)
        return out


class GINO_DeltaConv_LSR(BaseModel):
    """
    融合 GINO-LSR（长程依赖）和 DeltaConv（短程依赖）的模型
    - GINO-LSR：通过 NUFFT 学习长程几何依赖
    - DeltaConv：通过 DeltaPointPDE 学习短程局部几何依赖
    - 最终输出：两个路径结果相加（可选MLP融合）
    """
    def __init__(
        self,
        # 基础参数
        in_channels,
        out_channels,
        # DeltaConv 短程参数
        delta_pde_params,  # DeltaPointPDE 初始化参数字典
        # GINO-LSR 长程参数
        latent_feature_channels=None,
        projection_channels=256,
        gno_coord_dim=3,
        gno_radius=0.033,
        in_gno_transform_type='linear',
        out_gno_transform_type='linear',
        gno_pos_embed_type='transformer',
        # NUFFT 核心参数（GINO-LSR 内部）
        nufft_n_mesh=32,          
        nufft_x_lims=(-1.0, 1.0), 
        nufft_fno_channel=1,      
        # GINO-LSR 基础参数
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.tanh, 
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        fno_non_linearity=F.tanh,
        # 融合参数
        use_mlp_fusion=False,  # 是否使用MLP融合，False则直接相加
        **kwargs
        ):
        
        super().__init__()
        # ====================== 1. 初始化 DeltaConv（短程路径） ======================
        self.delta_pde = DeltaPointPDE(**delta_pde_params)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ====================== 2. 初始化 GINO-LSR 原有逻辑（长程路径） ======================
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels
        self.lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels

        # GINO-LSR 输入通道计算
        if in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = 3  # 输出3维特征匹配坐标维度

        self.fno_in_channels = in_gno_out_channels
        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels

        # 警告信息
        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data')
            gno_use_open3d = False
        if self.gno_coord_dim != 3:
            print(f'Warning: NUFFT expects 3-d data while input GNO expects {self.gno_coord_dim}-d data')

        # 维度顺序定义
        self.in_coord_dim = 3  # NUFFT 固定 3D
        self.gno_out_coord_dim = self.in_coord_dim
        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        # AdaIN 嵌入（保留原生逻辑）
        self.adain_pos_embed = None
        self.ada_in_dim = None
        self.act = nn.Tanh()

        # 核心参数
        self.gno_radius = gno_radius
        self.out_gno_tanh = out_gno_tanh

        ### GINO-LSR 原有层
        # 特征投影层
        self.feature_proj = nn.Linear(in_channels, 3).to(torch.float32)
        # 输入 GNO
        self.gno_in = GNOBlock(
            in_channels=3,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=gno_radius,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
        )
        # Lifting 层
        self.lifting = ChannelMLP(in_channels=self.fno_in_channels,
                                  hidden_channels=self.lifting_channels,
                                  out_channels=fno_hidden_channels,
                                  n_layers=3)
        # NUFFT 层（GINO-LSR 核心，长程）
        self.nufft_layer = NUFFTLayerMultiChannel3D_Param(
            nChannels=nufft_fno_channel,
            NpointsMesh=nufft_n_mesh,
            xLims=nufft_x_lims,
            mid_dim=fno_hidden_channels
        )

        self.nufft_nogauss_layer = NUFFTLayer_NOGauss_Param(
            nChannels=nufft_fno_channel,
            NpointsMesh=nufft_n_mesh,
            xLims=nufft_x_lims,
            mid_dim=fno_hidden_channels
        )

        # 输出 GNO
        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.gno_radius,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter
        )
        # 最终投影层
        self.projection = ChannelMLP(in_channels=fno_hidden_channels, 
                              out_channels=self.out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 

        # ====================== 3. 初始化融合层（可选） ======================
        self.use_mlp_fusion = use_mlp_fusion
        if use_mlp_fusion:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels * 2),
                nn.Tanh(),
                nn.Linear(self.out_channels * 2, self.out_channels)
            )

    # --------------------- GINO-LSR 原有 latent_embedding 方法（无修改） ---------------------
    def latent_embedding(self, in_p, coord, ada_in=None):
        """重构 latent_embedding 适配 NUFFT"""
        batch_size = in_p.shape[0]
        # Step 1: 维度调整 (b, n1, n2, n3, c) -> (b, c, n1, n2, n3)
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1, len(in_p.shape)-1)))
        # Step 2: Lifting 层
        in_p = self.lifting(in_p)
        # Step 3: 展平适配 NUFFT 输入
        c = in_p.shape[1] #32
        grid_size = in_p.shape[2] #10
        in_p_flat = in_p.reshape(batch_size, c, -1)  # (b, c, N) torch.Size([1, 32, 1000])
        coord_flat = coord.reshape(batch_size, -1, 3)  # (b, N, 3) torch.Size([1, 1000, 3])
        # Step 4: NUFFT 前向传播
        pdb.set_trace()
        nufft_out, _ = self.nufft_layer(coord_flat, in_p_flat)
        # nufft_out, _ = self.nufft_nogauss_layer(coord_flat, in_p_flat)
        # Step 5: 恢复 3D 形状
        nufft_out = nufft_out.reshape(batch_size, c, grid_size, grid_size, grid_size)
        return nufft_out 

    # --------------------- 核心 forward：融合 GINO-LSR + DeltaConv ---------------------
    def forward(self, x, pos, batch=None, **kwargs):
        """
        前向传播：
        - DeltaConv 路径：短程依赖（输入 x/pos/batch）
        - GINO-LSR 路径：长程依赖（原有逻辑）
        - 输出：两个路径结果相加（或MLP融合）
        Args:
            x: (num_points, in_channels) 输入特征
            pos: (num_points, 3) 3D点云坐标
            batch: (num_points,) 批次信息
        Returns:
            out: (num_points, out_channels) 最终输出
        """
        # ====================== 1. DeltaConv 短程路径 ======================
        # 构造 DeltaConv 所需的 Data 对象
        pdb.set_trace()
        delta_data = torch_geometric.data.Data(x=x.float(), pos=pos.float(), batch=batch)
        delta_output = self.delta_pde(delta_data)  # (num_points, out_channels)

        # ====================== 2. GINO-LSR 长程路径（原有逻辑） ======================
        # 特征投影到3维
        x_proj = self.feature_proj(x.float())  # (num_points, 3)
        
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        # 生成3D潜在网格
        min_pos = pos.min(dim=0)[0]
        max_pos = pos.max(dim=0)[0]
        grid_size = self.nufft_layer.NpointsMesh
        xx = torch.linspace(min_pos[0], max_pos[0], grid_size, device=pos.device)
        yy = torch.linspace(min_pos[1], max_pos[1], grid_size, device=pos.device)
        zz = torch.linspace(min_pos[2], max_pos[2], grid_size, device=pos.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xx, yy, zz, indexing='ij')
        latent_queries = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (grid_size, grid_size, grid_size, 3)
        
        # 输出查询点
        output_queries = pos.float()
        # 调整特征格式
        x_proj = x_proj.unsqueeze(0)  # (1, num_points, 3)

        # 输入GNO处理
        pdb.set_trace()
        in_p = self.gno_in(
            y=pos.float(),
            x=latent_queries.reshape(-1, 3),
            f_y=x_proj
        )
        
        # 形状调整
        grid_shape = latent_queries.shape[:-1]
        in_p = in_p.view((batch_size, *grid_shape, -1))
        
        # 潜在空间嵌入（NUFFT）
        latent_queries_batch = latent_queries.unsqueeze(0)
        latent_embed = self.latent_embedding(in_p=in_p, coord=latent_queries_batch)

        # 输出处理
        c = latent_embed.shape[1]
        latent_embed_flat = latent_embed.permute(0, *list(range(2, latent_embed.ndim)), 1).reshape(batch_size, -1, c)
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed_flat = torch.tanh(latent_embed_flat)
        
        # 输出GNO处理
        gino_out = self.gno_out(
            y=latent_queries.reshape(-1, 3),
            x=output_queries,
            f_y=latent_embed_flat
        )
        gino_out = gino_out.permute(0, 2, 1)
        # 最终投影
        gino_out = self.projection(gino_out).permute(0, 2, 1)
        gino_out = gino_out.squeeze(0)  # (num_points, out_channels)

        # ====================== 3. 融合两个路径的输出 ======================
        if self.use_mlp_fusion:
            # MLP融合（可选）
            combined_output = self.fusion_mlp(gino_out + delta_output)
        else:
            # 直接相加（核心需求）
            combined_output = self.act(gino_out + delta_output)

        return combined_output
