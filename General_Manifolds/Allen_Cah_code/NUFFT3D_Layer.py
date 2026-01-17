# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.fft import fftn, ifftn, fftshift, ifftshift
# import pdb

# class NUFFTLayerMultiChannel3D_Param(nn.Module):
#     def __init__(self, nChannels, NpointsMesh, xLims, mid_dim=1):
#         super(NUFFTLayerMultiChannel3D_Param, self).__init__()
        
#         self.mid_dim = mid_dim
#         self.nChannels = nChannels  # 每组mid_dim对应的参数个数
#         self.NpointsMesh = NpointsMesh
#         self.xLims = xLims
#         self.L = np.abs(self.xLims[1] - self.xLims[0])

#         tau_val = 12 * (self.L / (2 * np.pi * NpointsMesh)) ** 2
#         self.register_buffer('tau', torch.tensor(tau_val, dtype=torch.float32))

#         k_vals = torch.linspace(-(NpointsMesh // 2), NpointsMesh // 2, NpointsMesh)
#         k_vals = (2 * np.pi / self.L) * k_vals
#         self.register_buffer('kGrid', k_vals)
#         ky_grid, kx_grid, kz_grid = torch.meshgrid(
#             self.kGrid, self.kGrid, self.kGrid, indexing='ij'
#         )
#         self.register_buffer('kx_grid', kx_grid)
#         self.register_buffer('ky_grid', ky_grid)
#         self.register_buffer('kz_grid', kz_grid)

#         x_vals = torch.linspace(xLims[0], xLims[1], NpointsMesh + 1)[:-1]
#         self.register_buffer('xGrid', x_vals)
#         y_grid, x_grid, z_grid = torch.meshgrid(
#             self.xGrid, self.xGrid, self.xGrid, indexing='ij'
#         )
#         self.register_buffer('x_grid', x_grid)
#         self.register_buffer('y_grid', y_grid)
#         self.register_buffer('z_grid', z_grid)

#         self.grid_points = torch.stack([self.x_grid.flatten(), self.y_grid.flatten(),  self.z_grid.flatten()], dim=-1)
#         k = torch.sqrt(torch.square(self.kx_grid) + torch.square(self.ky_grid) + torch.square(self.kz_grid))
#         self.register_buffer('k', k)

#         deconv_real = self.gaussianDeconv3D(self.kx_grid, self.ky_grid, self.kz_grid, self.tau)
#         self.register_buffer('deconv_real', deconv_real)
#         self.register_buffer('deconv_imag', torch.zeros_like(deconv_real))

#         # 保持参数初始化：[mid_dim, nChannels] → 每个mid_dim有nChannels套参数
#         self.shift = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
#         self.amplitude = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
#         self.beta = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
#         self.hypera = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))

#     def gaussianDeconv3D(self, kx, ky, kz, tau):
#         return torch.sqrt(np.pi / tau)**3 * torch.exp((torch.square(kx) + torch.square(ky) + torch.square(kz))*tau)

#     def forward(self, x, f, debug=False):
#         # x: [B, N, 3], f: [B, C, N]（C=nChannels，输入通道数）
#         x = x.contiguous()
#         f = f.contiguous()
#         B, npoints, _ = x.shape
#         _, C, _ = f.shape  # C = nChannels（输入通道数）
#         N = self.NpointsMesh
#         M = self.mid_dim   # M = mid_dim（输出通道数）

#         # --- Step 2: Mollify (软化) ---
#         x_exp = x.view(B, npoints, 1, 1, 1, 3) # [B, N, 1, 1, 1, 3]
        
#         # 计算每个点到网格点的差值
#         diffx = x_exp[..., 0] - self.x_grid.view(1, 1, N, N, N)
#         diffy = x_exp[..., 1] - self.y_grid.view(1, 1, N, N, N)
#         diffz = x_exp[..., 2] - self.z_grid.view(1, 1, N, N, N)
        
#         # 周期性边界条件的高斯核
#         exp_term = lambda d: torch.exp(-d**2 / (4 * self.tau)) + \
#                             torch.exp(-(d - self.L)**2 / (4 * self.tau)) + \
#                             torch.exp(-(d + self.L)**2 / (4 * self.tau))
        
#         g_x = exp_term(diffx)
#         g_y = exp_term(diffy)
#         g_z = exp_term(diffz)
        
#         kernel = g_x * g_y * g_z # [B, N_points, N_grid, N_grid, N_grid]
        
#         # --- Step 3: Sample (采样到规则网格) ---
#         f_expanded = f.permute(0, 2, 1).view(B, npoints, C, 1, 1, 1) # [B, N, C, 1, 1, 1]
#         weighted = kernel.unsqueeze(2) * f_expanded # [B, N, C, N, N, N]
#         summed = weighted.sum(dim=1) # [B, C, N, N, N] -> fτ(xℓ)
        
#         # --- Step 4: Compute FFT ---
#         fft_val = fftshift(fftn(summed, dim=(-3, -2, -1)), dim=(-3, -2, -1)) # [B, C, N, N, N] (complex)

#         # --- Step 5-7: 频域操作（核心修改）---
#         deconv = torch.stack([self.deconv_real, self.deconv_imag], dim=-1)
#         deconv = torch.view_as_complex(deconv) # [N, N, N]

#         # 扩展参数维度：[M, C, 1, 1, 1] → 每个mid_dim对应nChannels套参数
#         amp = self.amplitude.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [M, C, 1, 1, 1]
#         sh = self.shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)       # [M, C, 1, 1, 1]
#         be = self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)        # [M, C, 1, 1, 1]
#         hy = self.hypera.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)      # [M, C, 1, 1, 1]
        
#         k_expanded = self.k.unsqueeze(0).unsqueeze(0) # [1, 1, N, N, N]
        
#         # 每个mid_dim + nChannels独立计算权重
#         part1 = amp / ((k_expanded + sh)**2 + amp**2)          # [M, C, N, N, N]
#         part2 = amp / ((k_expanded - sh)**2 + amp**2)          # [M, C, N, N, N]
#         filter_k = k_expanded**2 / (k_expanded**2 + 400*hy)    # [M, C, N, N, N]
#         multiplier = be * (part1 + part2) * filter_k            # [M, C, N, N, N]
        
#         # ========== 核心修改1：对nChannels求和，保留mid_dim ==========
#         # 原逻辑：sum(dim=0) → [C, N, N, N]（输出通道=C）
#         # 新逻辑：sum(dim=1) → [M, N, N, N]（输出通道=M=mid_dim）
#         total_multiplier = multiplier.sum(dim=1)  # [M, N, N, N]

#         # ========== 核心修改2：调整频域乘法的广播逻辑 ==========
#         # 1. 调整fft_val维度：[B, C, N, N, N] → 对C求和，适配mid_dim
#         fft_val_agg = fft_val.sum(dim=1, keepdim=True)  # [B, 1, N, N, N]
#         # 2. 扩展total_multiplier维度：[M, N, N, N] → [1, M, N, N, N]
#         total_multiplier_exp = total_multiplier.unsqueeze(0)  # [1, M, N, N, N]
#         # 3. 频域乘法：广播对齐 → [B, M, N, N, N]
#         # filtered_fft = fft_val_agg * deconv.unsqueeze(0).unsqueeze(0) * total_multiplier_exp  ## 对应算法是step5-6. 这是最初比较好的结果


#         # --- Step 8: Compute IFFT ---
#         inv_fft = ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))
#         inv_fft = torch.real(inv_fft) # [B, M, N, N, N]



#         # --- Step 9: Interpolate (插值回点云) ---
#         # 调整kernel维度以适配mid_dim：[B, N_points, N, N, N] → [B, N_points, 1, N, N, N]
#         kernel_exp = kernel.unsqueeze(2)  # [B, N_points, 1, N, N, N]
#         # einsum计算：[B, M, N, N, N] × [B, N_points, 1, N, N, N] → [B, M, N_points]
#         energy = torch.einsum('bmxyz,bnmxyz->bmn', inv_fft, kernel_exp)
        
#         # --- 返回结果 ---
#         pred = energy  # [B, M, N] → M=mid_dim（最终输出通道为mid_dim）
        
#         params = {
#             "amplitude": self.amplitude,
#             "shift": self.shift,
#             "beta": self.beta,
#             "hypera": self.hypera,
#             "total_multiplier": total_multiplier
#         }
#         return pred, params







import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
import pdb

class NUFFTLayerMultiChannel3D_Param(nn.Module):
    def __init__(self, nChannels, NpointsMesh, xLims, mid_dim=1):
        super(NUFFTLayerMultiChannel3D_Param, self).__init__()
        
        self.mid_dim = mid_dim
        self.nChannels = nChannels  # 每组mid_dim对应的参数个数
        self.NpointsMesh = NpointsMesh
        self.xLims = xLims
        self.L = np.abs(self.xLims[1] - self.xLims[0])

        tau_val = 12 * (self.L / (2 * np.pi * NpointsMesh)) ** 2
        self.register_buffer('tau', torch.tensor(tau_val, dtype=torch.float32))

        k_vals = torch.linspace(-(NpointsMesh // 2), NpointsMesh // 2, NpointsMesh)
        k_vals = (2 * np.pi / self.L) * k_vals
        self.register_buffer('kGrid', k_vals)
        ky_grid, kx_grid, kz_grid = torch.meshgrid(
            self.kGrid, self.kGrid, self.kGrid, indexing='ij'
        )
        self.register_buffer('kx_grid', kx_grid)
        self.register_buffer('ky_grid', ky_grid)
        self.register_buffer('kz_grid', kz_grid)

        x_vals = torch.linspace(xLims[0], xLims[1], NpointsMesh + 1)[:-1]
        self.register_buffer('xGrid', x_vals)
        y_grid, x_grid, z_grid = torch.meshgrid(
            self.xGrid, self.xGrid, self.xGrid, indexing='ij'
        )
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('z_grid', z_grid)

        self.grid_points = torch.stack([self.x_grid.flatten(), self.y_grid.flatten(),  self.z_grid.flatten()], dim=-1)
        k = torch.sqrt(torch.square(self.kx_grid) + torch.square(self.ky_grid) + torch.square(self.kz_grid))
        self.register_buffer('k', k)

        deconv_real = self.gaussianDeconv3D(self.kx_grid, self.ky_grid, self.kz_grid, self.tau)
        self.register_buffer('deconv_real', deconv_real)
        self.register_buffer('deconv_imag', torch.zeros_like(deconv_real))

        # 保持参数初始化：[mid_dim, nChannels] → 每个mid_dim有nChannels套参数
        self.shift = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.amplitude = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.beta = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.hypera = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))

    def gaussianDeconv3D(self, kx, ky, kz, tau):
        return torch.sqrt(np.pi / tau)**3 * torch.exp((torch.square(kx) + torch.square(ky) + torch.square(kz))*tau)

    def forward(self, x, f, debug=False):
        # x: [B, N, 3], f: [B, C, N]（C=nChannels，输入通道数）
        x = x.contiguous()
        f = f.contiguous()
        B, npoints, _ = x.shape
        _, C, _ = f.shape  # C = nChannels（输入通道数）
        N = self.NpointsMesh
        M = self.mid_dim   # M = mid_dim（输出通道数）

        # --- Step 2: Mollify (软化) ---
        x_exp = x.view(B, npoints, 1, 1, 1, 3) # [B, N, 1, 1, 1, 3]
        
        # 计算每个点到网格点的差值
        diffx = x_exp[..., 0] - self.x_grid.view(1, 1, N, N, N)
        diffy = x_exp[..., 1] - self.y_grid.view(1, 1, N, N, N)
        diffz = x_exp[..., 2] - self.z_grid.view(1, 1, N, N, N)
        
        # 周期性边界条件的高斯核
        exp_term = lambda d: torch.exp(-d**2 / (4 * self.tau)) + \
                            torch.exp(-(d - self.L)**2 / (4 * self.tau)) + \
                            torch.exp(-(d + self.L)**2 / (4 * self.tau))
        
        g_x = exp_term(diffx)
        g_y = exp_term(diffy)
        g_z = exp_term(diffz)
        
        kernel = g_x * g_y * g_z # [B, N_points, N_grid, N_grid, N_grid]
        
        # --- Step 3: Sample (采样到规则网格) ---
        f_expanded = f.permute(0, 2, 1).view(B, npoints, C, 1, 1, 1) # [B, N, C, 1, 1, 1]
        weighted = kernel.unsqueeze(2) * f_expanded # [B, N, C, N, N, N]
        summed = weighted.sum(dim=1) # [B, C, N, N, N] -> fτ(xℓ)
        
        # --- Step 4: Compute FFT ---
        fft_val = fftshift(fftn(summed, dim=(-3, -2, -1)), dim=(-3, -2, -1)) # [B, C, N, N, N] (complex)

        # --- Step 5-7: 频域操作（核心修改）---
        deconv = torch.stack([self.deconv_real, self.deconv_imag], dim=-1)
        deconv = torch.view_as_complex(deconv) # [N, N, N]

        # 扩展参数维度：[M, C, 1, 1, 1] → 每个mid_dim对应nChannels套参数
        amp = self.amplitude.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [M, C, 1, 1, 1]
        sh = self.shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)       # [M, C, 1, 1, 1]
        be = self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)        # [M, C, 1, 1, 1]
        hy = self.hypera.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)      # [M, C, 1, 1, 1]
        
        k_expanded = self.k.unsqueeze(0).unsqueeze(0) # [1, 1, N, N, N]
        
        # 每个mid_dim + nChannels独立计算权重
        part1 = amp / ((k_expanded + sh)**2 + amp**2)          # [M, C, N, N, N]
        part2 = amp / ((k_expanded - sh)**2 + amp**2)          # [M, C, N, N, N]
        filter_k = k_expanded**2 / (k_expanded**2 + 400*hy)    # [M, C, N, N, N]
        multiplier = be * (part1 + part2) * filter_k            # [M, C, N, N, N]
        
        # ========== 核心修改1：对nChannels求和，保留mid_dim ==========
        # 原逻辑：sum(dim=0) → [C, N, N, N]（输出通道=C）
        # 新逻辑：sum(dim=1) → [M, N, N, N]（输出通道=M=mid_dim）
        total_multiplier = multiplier.sum(dim=1)  # [M, N, N, N]

        # ========== 核心修改2：调整频域乘法的广播逻辑 ==========
        # 1. 调整fft_val维度：[B, C, N, N, N] → 对C求和，适配mid_dim
        fft_val_agg = fft_val.sum(dim=1, keepdim=True)  # [B, 1, N, N, N]
        # 2. 扩展total_multiplier维度：[M, N, N, N] → [1, M, N, N, N]
        total_multiplier_exp = total_multiplier.unsqueeze(0)  # [1, M, N, N, N]
        # 3. 频域乘法：广播对齐 → [B, M, N, N, N]
        filtered_fft = fft_val_agg * deconv.unsqueeze(0).unsqueeze(0) * total_multiplier_exp  ## 对应算法是step5-6


        # filtered_fft = fft_val_agg  * total_multiplier_exp # 没有step5 ，就直接step6

        # --- Step 8: Compute IFFT ---
        inv_fft = ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        inv_fft = torch.real(inv_fft) # [B, M, N, N, N]



        # --- Step 9: Interpolate (插值回点云) ---
        # 调整kernel维度以适配mid_dim：[B, N_points, N, N, N] → [B, N_points, 1, N, N, N]
        kernel_exp = kernel.unsqueeze(2)  # [B, N_points, 1, N, N, N]
        # einsum计算：[B, M, N, N, N] × [B, N_points, 1, N, N, N] → [B, M, N_points]
        energy = torch.einsum('bmxyz,bnmxyz->bmn', inv_fft, kernel_exp)
        
        # --- 返回结果 ---
        pred = energy  # [B, M, N] → M=mid_dim（最终输出通道为mid_dim）
        
        params = {
            "amplitude": self.amplitude,
            "shift": self.shift,
            "beta": self.beta,
            "hypera": self.hypera,
            "total_multiplier": total_multiplier
        }
        return pred, params