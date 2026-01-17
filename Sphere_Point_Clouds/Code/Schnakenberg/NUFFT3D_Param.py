

# import time
# import os
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.fft import fftn, ifftn, fftshift, ifftshift
# import pdb


# class NUFFTLayerMultiChannel3D_Param(nn.Module):
#     def __init__(self, nChannels, NpointsMesh, xLims):
#         super(NUFFTLayerMultiChannel3D_Param, self).__init__()
        
#         self.nChannels = nChannels
#         self.NpointsMesh = NpointsMesh
#         self.xLims = xLims
#         self.L = np.abs(self.xLims[1] - self.xLims[0])
      
 
#         # assert NpointsMesh % 2 == 1

#         # # Register buffers
#         self.register_buffer('tau', torch.tensor(12*(self.L/(2*np.pi*NpointsMesh))**2, 
#                                         dtype=torch.float32))
        

#         # 计算tau并注册为buffer
#         tau_val = 12 * (self.L / (2 * np.pi * NpointsMesh)) ** 2
#         self.register_buffer('tau', torch.tensor(tau_val, dtype=torch.float32))

#         # Frequency grid
#         k_vals = torch.linspace(-(NpointsMesh // 2), NpointsMesh // 2, NpointsMesh)
#         k_vals = (2 * np.pi / self.L) * k_vals
#         self.register_buffer('kGrid', k_vals)

#         ky_grid, kx_grid, kz_grid = torch.meshgrid(
#             self.kGrid, self.kGrid, self.kGrid, indexing='ij'
#         )
#         self.register_buffer('kx_grid', kx_grid)
#         self.register_buffer('ky_grid', ky_grid)
#         self.register_buffer('kz_grid', kz_grid)

#         # Spatial grid
#         x_vals = torch.linspace(xLims[0], xLims[1], NpointsMesh + 1)[:-1]  # [N]
#         self.register_buffer('xGrid', x_vals)

#         y_grid, x_grid, z_grid = torch.meshgrid(
#             self.xGrid, self.xGrid, self.xGrid, indexing='ij'
#         )
#         self.register_buffer('x_grid', x_grid)
#         self.register_buffer('y_grid', y_grid)
#         self.register_buffer('z_grid', z_grid)

#         self.grid_points = torch.stack([self.x_grid.flatten(), self.y_grid.flatten(),  self.z_grid.flatten()], dim=-1)  # [1000, 3]

#         k =  torch.sqrt(torch.square(self.kx_grid) + torch.square(self.ky_grid) + torch.square(self.kz_grid))
#         self.register_buffer('k', k)

#          # 缓存 Deconv 实部
#         deconv_real = self.gaussianDeconv3D(self.kx_grid, self.ky_grid, self.kz_grid, self.tau)
#         self.register_buffer('deconv_real', deconv_real)
#         self.register_buffer('deconv_imag', torch.zeros_like(deconv_real))



#         ########################################################

#         self.build()

        
#     def gaussianPer(self, x, tau, L=2*np.pi):
#         return torch.exp(-torch.square(x)/(4*tau)) + \
#             torch.exp(-torch.square(x-L)/(4*tau)) + \
#             torch.exp(-torch.square(x+L)/(4*tau))




#     def gaussianDeconv3D(self,kx, ky, kz, tau):
#         return torch.sqrt(np.pi / tau)**3 * torch.exp((torch.square(kx) + 
#                                                                 torch.square(ky) + 
#                                                                 torch.square(kz))*tau)
    
    





#     def spherical_to_cartesian(self,lat, lon):
#         """
#         将球面坐标转换为笛卡尔坐标
#         参数:
#             lat: 纬度，范围[-π/2, π/2]
#             lon: 经度，范围[0, 2π]
#         返回:
#             x, y, z: 笛卡尔坐标
#         """
#         # 转换为笛卡尔坐标
#         x = torch.cos(lat) * torch.cos(lon)
#         y = torch.cos(lat) * torch.sin(lon)
#         z = torch.sin(lat)
#         return torch.stack([x, y, z], dim=-1)  # 形状为[... ,3]



#     def prepare_spherical_data(self,f):
#         """
#         准备球面数据用于3D NUFFT
#         参数:
#             f: 输入数据，形状[B, C, nlat, nlon]
#         返回:
#             xyz: 笛卡尔坐标，形状[B, nlat*nlon, 3]
#             f_reshaped: 重整后的数据，形状[B, C, nlat*nlon]
#         """
#         B, C, nlat, nlon = f.shape
#         device = f.device  # 保证所有生成的张量也在这个 device 上
        
#         # 使用 torch 而不是 numpy
#         lat = torch.linspace(-torch.pi/2, torch.pi/2, nlat, device=device)
#         lon = torch.linspace(0, 2*torch.pi, nlon, device=device)
        
#         lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')  # [nlat, nlon]
        
#         # 转换为笛卡尔坐标
#         xyz = self.spherical_to_cartesian(lat_grid, lon_grid)  # [nlat, nlon, 3]
        
#         # 扩展为 batch
#         xyz = xyz.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, nlat*nlon, 3)  # [B, nlat*nlon, 3]
        
#         f_reshaped = f.reshape(B, C, nlat*nlon)  # [B, C, nlat*nlon]
        
#         return f_reshaped, xyz




#     def build(self):
#         # print("building the channels")
#         self.shift = nn.ParameterList([
#             nn.Parameter(torch.randn(1,)) for _ in range(self.nChannels)
#         ])
#         self.amplitude = nn.ParameterList([
#             nn.Parameter(torch.randn(1,)) for _ in range(self.nChannels)
#         ])

#         self.beta = nn.ParameterList([
#             nn.Parameter(torch.randn(1,)) for _ in range(self.nChannels)
#         ])
#         self.hypera =  nn.ParameterList([
#             nn.Parameter(torch.randn(1,)) for _ in range(self.nChannels)
#         ])
#         self.sigma =  nn.ParameterList([
#             nn.Parameter(torch.randn(1,))  for _ in range(self.nChannels)
#         ])

  



#     def forward(self, input, debug=False):
#         """
#         Args:
#             input: shape [B, C, nlat, nlon]
#         Returns:
#             pred: [B, C, nlat, nlon]
#         """
#         f, x = self.prepare_spherical_data(input)
#         x = x.contiguous()
#         f = f.contiguous()

#         B, npoints, _ = x.shape
#         _, C, _ = f.shape
#         N = self.NpointsMesh

#         # [B, npoints, 1, 1, 1, 3]
#         x_exp = x.view(B, npoints, 1, 1, 1, 3)
#         # pdb.set_trace()

#         # Compute Gaussian kernel using broadcasting but lower memory cost
#         diffx = x_exp[..., 0] - self.x_grid[:, 0, 0].view(1, 1, N, 1, 1)
#         diffy = x_exp[..., 1] - self.y_grid[0, :, 0].view(1, 1, 1, N, 1)
#         diffz = x_exp[..., 2] - self.z_grid[0, 0, :].view(1, 1, 1, 1, N)

#         # Compute Gaussian kernel (periodic)
#         exp_term = lambda d: torch.exp(-d**2 / (4 * self.tau)) + \
#                             torch.exp(-(d - self.L)**2 / (4 * self.tau)) + \
#                             torch.exp(-(d + self.L)**2 / (4 * self.tau))

#         g_x = exp_term(diffx)  # [B, npoints, N, 1, 1]
#         g_y = exp_term(diffy)  # [B, npoints, 1, N, 1]
#         g_z = exp_term(diffz)  # [B, npoints, 1, 1, N]

#         # Combined Gaussian kernel
#         kernel = g_x * g_y * g_z  # [B, npoints, N, N, N]

#         # Weight and sum
#         f = f.permute(0, 2, 1).view(B, npoints, C, 1, 1, 1)
#         weighted = kernel.unsqueeze(2) * f  # [B, npoints, C, N, N, N]
#         summed = weighted.sum(dim=1)  # [B, C, N, N, N]

#         # FFT and shift
#         fft_val = fftshift(fftn(summed, dim=(-3, -2, -1)), dim=(-3, -2, -1))

#         # Apply Deconv (cached in __init__)
#         # Deconv = torch.view_as_complex(torch.stack([
#         #     self.deconv_real.unsqueeze(0).unsqueeze(0),
#         #     self.deconv_imag.unsqueeze(0).unsqueeze(0)
#         # ], dim=-1))
#         # fft_val = fft_val * Deconv

#         # real_fft, imag_fft = torch.real(fft_val), torch.imag(fft_val)

#         # Spectral multiplier
#         total_multiplier = 0
#         for i in range(self.nChannels):
#             part1 = self.amplitude[i] / ((self.k+ self.shift[i])**2 + self.amplitude[i]**2)
#             part2 = self.amplitude[i] / ((self.k - self.shift[i])**2 + self.amplitude[i]**2)
#             filter_k = self.k**2 / (self.k**2 + 400*self.hypera[i])
#             total_multiplier += (self.beta[i] * (part1 + part2) * filter_k).unsqueeze(0).unsqueeze(0)

#             # total_multiplier += (self.beta[i] * (part1 + part2)).unsqueeze(0).unsqueeze(0)




#         #############################
#         # pdb.set_trace()
#         filtered_fft = total_multiplier * fft_val
#         ##########################

#         # Inverse FFT
#         # inv_fft = torch.real(ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1)))) / \
#         #         (2*np.pi*self.NpointsMesh/self.L)**3 / (2*np.pi)**3 / 2
#         inv_fft = torch.real(ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1))))


#         energy = torch.einsum('bcxyz,bnxyz->bcn', inv_fft, kernel)
#         pred = energy.permute(0, 2, 1).reshape(input.shape)
#         params = {
#             "amplitude": torch.stack([p for p in self.amplitude]),
#             "shift": torch.stack([p for p in self.shift]),
#             "beta": torch.stack([p for p in self.beta]),
#             "hypera": torch.stack([p for p in self.hypera]),
#             "total_multiplier": total_multiplier
#         }
#         return pred,params

        
        



        










import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
import pdb

class NUFFTLayerMultiChannel3D_Param(nn.Module):
    def __init__(self,  nChannels, NpointsMesh, xLims,mid_dim=1):
        super(NUFFTLayerMultiChannel3D_Param, self).__init__()
        
        self.mid_dim = mid_dim
        self.nChannels = nChannels
        self.NpointsMesh = NpointsMesh
        self.xLims = xLims
        self.L = np.abs(self.xLims[1] - self.xLims[0])

        # tau buffer
        tau_val = 12 * (self.L / (2 * np.pi * NpointsMesh)) ** 2
        self.register_buffer('tau', torch.tensor(tau_val, dtype=torch.float32))

        # Frequency grid
        k_vals = torch.linspace(-(NpointsMesh // 2), NpointsMesh // 2, NpointsMesh)
        k_vals = (2 * np.pi / self.L) * k_vals
        self.register_buffer('kGrid', k_vals)
        ky_grid, kx_grid, kz_grid = torch.meshgrid(
            self.kGrid, self.kGrid, self.kGrid, indexing='ij'
        )
        self.register_buffer('kx_grid', kx_grid)
        self.register_buffer('ky_grid', ky_grid)
        self.register_buffer('kz_grid', kz_grid)

        # Spatial grid
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

        # Cache Deconv
        deconv_real = self.gaussianDeconv3D(self.kx_grid, self.ky_grid, self.kz_grid, self.tau)
        self.register_buffer('deconv_real', deconv_real)
        self.register_buffer('deconv_imag', torch.zeros_like(deconv_real))

        # 参数初始化：保持原始逻辑，不乘0.1
        self.shift = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.amplitude = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.beta = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))
        self.hypera = nn.Parameter(torch.randn(self.mid_dim, self.nChannels))

    def gaussianDeconv3D(self,kx, ky, kz, tau):
        return torch.sqrt(np.pi / tau)**3 * torch.exp((torch.square(kx) + torch.square(ky) + torch.square(kz))*tau)

    def spherical_to_cartesian(self,lat, lon):
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)

    def prepare_spherical_data(self,f):
        B, C, nlat, nlon = f.shape
        device = f.device
        lat = torch.linspace(-torch.pi/2, torch.pi/2, nlat, device=device)
        lon = torch.linspace(0, 2*torch.pi, nlon, device=device)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
        xyz = self.spherical_to_cartesian(lat_grid, lon_grid)
        xyz = xyz.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, nlat*nlon, 3)
        f_reshaped = f.reshape(B, C, nlat*nlon)
        return f_reshaped, xyz

    def forward(self, input, debug=False):
        f, x = self.prepare_spherical_data(input)
        x = x.contiguous()
        f = f.contiguous()
        B, npoints, _ = x.shape
        _, C, _ = f.shape
        N = self.NpointsMesh

        # Gaussian kernel
        x_exp = x.view(B, npoints, 1, 1, 1, 3)
        diffx = x_exp[..., 0] - self.x_grid[:, 0, 0].view(1, 1, N, 1, 1)
        diffy = x_exp[..., 1] - self.y_grid[0, :, 0].view(1, 1, 1, N, 1)
        diffz = x_exp[..., 2] - self.z_grid[0, 0, :].view(1, 1, 1, 1, N)
        exp_term = lambda d: torch.exp(-d**2 / (4 * self.tau)) + \
                            torch.exp(-(d - self.L)**2 / (4 * self.tau)) + \
                            torch.exp(-(d + self.L)**2 / (4 * self.tau))
        g_x = exp_term(diffx)
        g_y = exp_term(diffy)
        g_z = exp_term(diffz)
        kernel = g_x * g_y * g_z

        # Weight and sum
        f = f.permute(0, 2, 1).view(B, npoints, C, 1, 1, 1)
        weighted = kernel.unsqueeze(2) * f
        summed = weighted.sum(dim=1)
        fft_val = fftshift(fftn(summed, dim=(-3, -2, -1)), dim=(-3, -2, -1))

        # Spectral multiplier：每个 mid_dim 独立，每个 mid_dim 内 nChannels 基础 multiplier 相加
        # total_multiplier shape: [1, mid_dim, N, N, N]
        amp = self.amplitude.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [mid_dim, nChannels,1,1,1]
        sh = self.shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        be = self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        hy = self.hypera.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pdb.set_trace()

        part1 = amp / ((self.k.unsqueeze(0) + sh)**2 + amp**2) # [mid_dim, nChannels,NpointsMesh,NpointsMesh,NpointsMesh]
        part2 = amp / ((self.k.unsqueeze(0) - sh)**2 + amp**2) # [mid_dim, nChannels,NpointsMesh,NpointsMesh,NpointsMesh]

        filter_k = self.k.unsqueeze(0)**2 / (self.k.unsqueeze(0)**2 + 400*hy) # [mid_dim, nChannels,NpointsMesh,NpointsMesh,NpointsMesh]
        multiplier = be * (part1 + part2) * filter_k  # [mid_dim, nChannels, N, N, N]
        total_multiplier = multiplier.sum(dim=1)  #  [mid_dim, N, N, N]
        total_multiplier = total_multiplier.unsqueeze(0)  # add batch dim -> [1, mid_dim, N, N, N]


        #### NO High Pass
        # multiplier = be * (part1 + part2) # [mid_dim, nChannels, N, N, N]
        # total_multiplier = multiplier.sum(dim=1)  #  [mid_dim, N, N, N]
        # total_multiplier = total_multiplier.unsqueeze(0)  # add batch dim -> [1, mid_dim, N, N, N]

        # filtered_fft = total_multiplier * fft_val
        ################################################
        # --- Step 5: 频域操作（核心修改）---
        deconv = torch.stack([self.deconv_real, self.deconv_imag], dim=-1)
        deconv = torch.view_as_complex(deconv) # [N, N, N]
        filtered_fft = fft_val*deconv*total_multiplier
#######################################################

        # Inverse FFT
        inv_fft = torch.real(ifftn(ifftshift(filtered_fft, dim=(-3, -2, -1))))

        energy = torch.einsum('bcxyz,bnxyz->bcn', inv_fft, kernel)
        pred = energy.permute(0, 2, 1).reshape(input.shape)

        params = {
            "amplitude": self.amplitude,
            "shift": self.shift,
            "beta": self.beta,
            "hypera": self.hypera,
            "total_multiplier": total_multiplier
        }
        return pred, params
 
    