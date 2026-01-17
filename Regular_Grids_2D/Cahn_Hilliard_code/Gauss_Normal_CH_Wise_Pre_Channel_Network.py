

import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch.nn.functional as F
import pdb




class FourierLayer(nn.Module):
    def __init__(self, fourier_channels, NpointsMesh=10,
                 initial_amplitude=None, initial_shift=None, initial_beta=None, 
                 phase=1, L_phys=16*np.pi, L_math=16*np.pi):  
      
        super(FourierLayer, self).__init__()
        self.fourier_channels = fourier_channels
        self.NpointsMesh = int(NpointsMesh) if not isinstance(NpointsMesh, int) else NpointsMesh

    
        self.L_phys = float(L_phys)
      
        self.L_math = float(L_math)
        self.grid_size = 128

        self.tau = 12 * (self.L_math / (2 * np.pi * NpointsMesh)) ** 2  

 
        mollify_grid_1d = torch.linspace(0, self.L_math, NpointsMesh + 1)[:-1]
        self.x_mollify, self.y_mollify = torch.meshgrid(mollify_grid_1d, mollify_grid_1d, indexing='ij')

       
        point_1d = torch.linspace(0, self.L_phys, self.grid_size + 1)[:-1]
        x_points, y_points = torch.meshgrid(point_1d, point_1d, indexing='ij')
        self.points = torch.stack([x_points.flatten(), y_points.flatten()], dim=-1)

  
        k_lin = torch.linspace(-NpointsMesh//2, NpointsMesh//2 - 1, NpointsMesh)
        kx_grid, ky_grid = torch.meshgrid(k_lin, k_lin, indexing='ij')
        self.kx_grid = (2 * np.pi / self.L_math) * kx_grid 
        self.ky_grid = (2 * np.pi / self.L_math) * ky_grid
        self.k = torch.sqrt(self.kx_grid**2 + self.ky_grid**2)

        self.deconv_real = self.gaussianDeconv2D(self.kx_grid, self.ky_grid, self.tau)
        self.deconv_imag = torch.zeros_like(self.deconv_real)
        self.deconv = torch.view_as_complex(torch.stack([self.deconv_real, self.deconv_imag], dim=-1))

      
        # Amplitude
        if phase == 1:
            if initial_amplitude is None:
                self.amplitude = nn.Parameter(torch.randn(fourier_channels))
            else:
                self.amplitude = nn.Parameter(initial_amplitude.clone().detach().float())
        else:
            if initial_amplitude is None:
                self.amplitude = torch.randn(fourier_channels, dtype=torch.float32)
            else:
                self.amplitude = initial_amplitude.clone().detach().float().requires_grad_(False)

        # Beta 
        if phase == 2:
            if initial_beta is None:
                self.beta = nn.Parameter(torch.randn(fourier_channels))
            else:
                self.beta = nn.Parameter(initial_beta.clone().detach().float())
        else:
            if initial_beta is None:
                self.beta = torch.randn(fourier_channels, dtype=torch.float32)
            else:
                self.beta = initial_beta.clone().detach().float().requires_grad_(False)

        # Shift 
        if phase == 3:
            if initial_shift is None:
                self.shift = nn.Parameter(torch.randn(fourier_channels))
            else:
                self.shift = nn.Parameter(initial_shift.clone().detach().float())
        else:
            if initial_shift is None:
                self.shift = torch.randn(fourier_channels, dtype=torch.float32)
            else:
                self.shift = initial_shift.clone().detach().float().requires_grad_(False)

        # Phase 4 
        if phase == 4:
            self.amplitude = nn.Parameter(
                torch.randn(fourier_channels) if initial_amplitude is None else initial_amplitude.clone().detach().float()
            )
            self.beta = nn.Parameter(
                torch.randn(fourier_channels) if initial_beta is None else initial_beta.clone().detach().float()
            )
            self.shift = nn.Parameter(
                torch.randn(fourier_channels) if initial_shift is None else initial_shift.clone().detach().float()
            )
        
        self.hypera = nn.Parameter(torch.rand(fourier_channels))  

    def gaussianDeconv2D(self, kx, ky, tau):
        return (np.pi / tau) * torch.exp((torch.square(kx) + torch.square(ky)) * tau)


    def forward(self, x, debug=False):
        B, C, H, W = x.shape
        Nspoints = H*W

    
        points = self.points.to(x.device).unsqueeze(0).repeat(B, 1, 1)
        x_mollify = self.x_mollify.to(x.device)
        y_mollify = self.y_mollify.to(x.device)
        k = self.k.to(x.device)
        deconv = self.deconv.to(x.device)


        points_exp = points.view(B, Nspoints, 1, 1, 2)
        diffx = points_exp[..., 0] - x_mollify.view(1, 1, self.NpointsMesh, self.NpointsMesh)
        diffy = points_exp[..., 1] - y_mollify.view(1, 1, self.NpointsMesh, self.NpointsMesh)
        
      
        exp_term = lambda d: torch.exp(-d**2 / (4 * self.tau)) + \
                            torch.exp(-(d - self.L_math)**2 / (4 * self.tau)) + \
                            torch.exp(-(d + self.L_math)**2 / (4 * self.tau))
        
        g_x = exp_term(diffx)
        g_y = exp_term(diffy)
        kernel = g_x * g_y

        x_flat = x.reshape(B, C, Nspoints).permute(0, 2, 1)
        x_expanded = x_flat.unsqueeze(-1).unsqueeze(-1)
        kernel_exp = kernel.unsqueeze(2)
        
        weighted = kernel_exp * x_expanded
        summed = weighted.sum(dim=1)

    
        nfft = self.NpointsMesh * self.NpointsMesh
        fft_val = torch.fft.fft2(summed, dim=(-2, -1)) / nfft
        fft_val = torch.fft.fftshift(fft_val, dim=(-2, -1))

        k_expanded = k.unsqueeze(0).unsqueeze(0)
        amp = self.amplitude.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sh = self.shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        be = self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        hy = self.hypera.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        part1 = amp / ((k_expanded + sh)**2 + amp**2)
        part2 = amp / ((k_expanded - sh)**2 + amp**2)
        filter_k = k_expanded**2 / (k_expanded**2 + 400 * hy)
        multiplier = be * (part1 + part2) * filter_k

        total_multiplier = multiplier.sum(dim=0)
        total_multiplier_exp = total_multiplier.unsqueeze(0)

        filtered_fft = fft_val * deconv.unsqueeze(0).unsqueeze(0) * total_multiplier_exp

        inv_fft = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
        inv_fft = torch.fft.ifft2(inv_fft, dim=(-2, -1))
        inv_fft = torch.real(inv_fft)

        kernel_exp = kernel.unsqueeze(2)
        inv_fft_exp = inv_fft.unsqueeze(1)
        energy = torch.einsum('bpnxy,bcnxy->bpc', kernel_exp, inv_fft_exp)
        pred = energy.permute(0, 2, 1).reshape(B, C, self.grid_size, self.grid_size)

        return pred, fft_val, self.amplitude, self.shift, self.beta, total_multiplier, self.hypera
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
       



class SpectralLayer(nn.Module):
    def __init__(self, in_channels, out_channels, channel_feat, kernel_size,
                 fourier_channels=None,  NpointsMesh=None,initial_amplitude=None, initial_shift=None,
                 initial_beta=None, phase=1):
        super(SpectralLayer, self).__init__()

   
        self.fourier_channels = fourier_channels if fourier_channels is not None else channel_feat
        self.fu = FourierLayer(self.fourier_channels, NpointsMesh,initial_amplitude, initial_shift, initial_beta, phase=phase)
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, channel_feat, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv2 = nn.Conv2d(channel_feat, out_channels, kernel_size, padding=pad, padding_mode='circular',
                               bias=False)
        k2 = 3
        p2 = (k2 - 1) // 2
        self.conv10 = nn.Conv2d(channel_feat, channel_feat, k2, padding=p2, padding_mode='circular',
                                bias=False)
       
        self.tanh = nn.Tanh()
     

        padding_s = (kernel_size - 2) // 2
        padding_up = (kernel_size - 2) // 2
       
        self.stride_conv_d5 = nn.Conv2d(channel_feat, channel_feat, kernel_size=kernel_size, stride=5,
                                        padding=padding_s)
        self.transconv_d5 = nn.ConvTranspose2d(channel_feat, channel_feat, kernel_size=kernel_size, stride=5,
                                               padding=padding_up)


        self.apply(init_weights)



    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.stride_conv_d5(x1)
        x1 = self.conv10(x1)
        x1 = self.transconv_d5(x1)


        x1 = self.conv2(x1)
   
        iffx, ffx, amplitude, shift, beta, multiplier, a_par = self.fu(x)
        # output = (self.tanh(x1+iffx)+1)/2


        output = (self.tanh(x1)+1)/2


        return output, x1, iffx, output, ffx, amplitude, shift, beta, multiplier, a_par







class Wise_SpectralTransform(nn.Module):
    def __init__(self, num_layers, channel_feat, kernel_size,
                 fourier_channels=None, NpointsMesh=None,phase=1, initial_parameters=None):
        super(Wise_SpectralTransform, self).__init__()
        layers = []

        for i in range(num_layers):
            initial_amplitude = (initial_parameters['amplitudes'][i]
                                 if initial_parameters and initial_parameters['amplitudes'] is not None else None)
            initial_shift = (initial_parameters['shifts'][i]
                             if initial_parameters and initial_parameters['shifts'] is not None else None)
            initial_beta = (initial_parameters['betas'][i]
                            if initial_parameters and initial_parameters['betas'] is not None else None)

            layer = SpectralLayer(
                in_channels=1,
                out_channels=1,
                channel_feat=channel_feat,
                kernel_size=kernel_size,
                NpointsMesh = NpointsMesh,
                fourier_channels=fourier_channels,  
                initial_amplitude=initial_amplitude,
                initial_shift=initial_shift,
                initial_beta=initial_beta,
                phase=phase
            )
            layers.append(layer)

        self.snet = nn.Sequential(*layers)
    def forward(self, input):
        x = input
        seq = []
        mid = {}
        fourier = {}
        amplitudes = []
        shifts = []
        betas = []
        multipliers = []
        a_pars = []
        layer_number = 0

        seq.append(input)
        fourier[layer_number] = [input]

        for layer in self.snet:
            x, mid1, iffx, output, ffx, amplitude, shift, beta, multiplier, a_par = layer(x)
            layer_number += 1

            seq.append(x)
            amplitudes.append(amplitude)
            shifts.append(shift)
            betas.append(beta)
            multipliers.append(multiplier)
            a_pars.append(a_par)

            mid[layer_number] = [mid1, iffx, output]
            fourier[layer_number] = [ffx, iffx]

        mid[-1] = [input]
        mid[0] = [input]

        return x, seq, mid, fourier, amplitudes, shifts, betas, multipliers, a_pars
