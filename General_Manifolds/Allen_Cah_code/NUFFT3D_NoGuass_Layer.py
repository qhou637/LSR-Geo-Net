import torch
import numpy as np
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
import pdb

class NUFFTLayer_NOGauss_Param(nn.Module):
    def __init__(self, nChannels, NpointsMesh, xLims, mid_dim=1):
        super(NUFFTLayer_NOGauss_Param, self).__init__()
        
        self.mid_dim = mid_dim
        self.nChannels = nChannels
        self.NpointsMesh = NpointsMesh  # GNO输出的规则网格尺寸（如32→32×32×32）
        self.xLims = xLims
        self.L = np.abs(self.xLims[1] - self.xLims[0])

        # ========== 1. 仅保留频域网格（FFT滤波必需），删除所有空间网格/高斯相关代码 ==========
        # 频域k网格初始化
        k_vals = torch.linspace(-(NpointsMesh // 2), NpointsMesh // 2, NpointsMesh)
        k_vals = (2 * np.pi / self.L) * k_vals
        self.register_buffer('kGrid', k_vals)
        ky_grid, kx_grid, kz_grid = torch.meshgrid(
            self.kGrid, self.kGrid, self.kGrid, indexing='ij'
        )
        self.register_buffer('kx_grid', kx_grid)
        self.register_buffer('ky_grid', ky_grid)
        self.register_buffer('kz_grid', kz_grid)

        # 频域k的模长（滤波权重计算必需）
        k = torch.sqrt(torch.square(self.kx_grid) + torch.square(self.ky_grid) + torch.square(self.kz_grid))
        self.register_buffer('k', k)

        # ========== 2. 保留可学习滤波参数（核心逻辑不变） ==========
        self.shift = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.amplitude = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.beta = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.hypera = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))

    def forward(self, x, f, debug=False):
        """
        输入：
            x: [B, N^3, 3] → GNO生成的规则网格展平坐标（仅兼容原代码，实际未使用）
            f: [B, C, N^3] → GNO生成的规则网格展平特征（C = nChannels）
        输出：
            pred: [B, M, N^3] → 频域滤波后的展平特征（M = mid_dim）
            params: 可学习参数字典
        核心修改：
            1. 直接reshape展平特征为规则网格，跳过Mollification+Sample
            2. 完全删除Interpolation步骤，滤波后直接展平输出
        """
        x = x.contiguous()
        f = f.contiguous()
        B, _, _ = x.shape
        _, C, _ = f.shape
        N = self.NpointsMesh
        M = self.mid_dim

        # ========== Step 1: 展平特征 → 规则网格（核心替换：无采样，直接reshape） ==========
        # f: [B, C, N^3] → [B, C, N, N, N]
        grid_feat = f.reshape(B, C, N, N, N)  # 直接恢复3D网格形状

        # ========== Step 2: 3D FFT（原有逻辑保留） ==========
        fft_val = fftshift(fftn(grid_feat, dim=(-3, -2, -1)), dim=(-3, -2, -1))  # [B, C, N, N, N] (complex)

        # ========== Step 3: 频域滤波（移除反卷积，保留参数逻辑） ==========
        # 扩展参数维度：[M, C, 1, 1, 1]
        amp = self.amplitude.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sh = self.shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        be = self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        hy = self.hypera.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        k_expanded = self.k.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N, N]
        
        # 滤波权重计算（原有逻辑不变）
        part1 = amp / ((k_expanded + sh)**2 + amp**2)
        part2 = amp / ((k_expanded - sh)**2 + amp**2)
        filter_k = k_expanded**2 / (k_expanded**2 + 400*hy)
        multiplier = be * (part1 + part2) * filter_k  # [M, C, N, N, N]
        
        # 通道求和，保留mid_dim维度
        total_multiplier = multiplier.sum(dim=1)  # [M, N, N, N]

        # ========== Step 4: 频域乘法（无反卷积项） ==========
        fft_val_agg = fft_val.sum(dim=1, keepdim=True)  # [B, 1, N, N, N]
        total_multiplier_exp = total_multiplier.unsqueeze(0)  # [1, M, N, N, N]
        filtered_fft = fft_val_agg * total_multiplier_exp  # [B, M, N, N, N]

        # ========== Step 5: 3D IFFT（原有逻辑保留） ==========
        inv_fft = ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        inv_fft = torch.real(inv_fft)  # [B, M, N, N, N]

        # ========== Step 6: 规则网格 → 展平输出（完全替代Interpolation） ==========
        # 直接展平，无需插值回点云（因为GNO后续需要的就是展平特征）
        pred = inv_fft.flatten(start_dim=-3)  # [B, M, N^3]

        # 返回结果
        params = {
            "amplitude": self.amplitude,
            "shift": self.shift,
            "beta": self.beta,
            "hypera": self.hypera,
            "total_multiplier": total_multiplier
        }
        return pred, params