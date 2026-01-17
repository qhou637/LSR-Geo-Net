import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F

import math
from disco_local import DiscreteContinuousConvS2
from torch_harmonics import RealSHT, InverseRealSHT
from sphere_cnn import SphereConv2D, SphereMaxPool2D
from NUFFT3D_Param import  NUFFTLayerMultiChannel3D_Param as nufft3d_param
from NUFFT2D import  NUFFTLayerMultiChannel2D as nufft2d
from torch_harmonics import ResampleS2
import sys
import os
from functools import partial
import pdb
# pdb.set_trace = lambda: None

class ParallelCoreLayer(nn.Module):
    def __init__(self, mid_dim, fourier_channel,shape, basis_type):
        super().__init__()
        self.nufft = nufft3d_param(nChannels=fourier_channel, NpointsMesh=10, xLims=[-1.0, 1.0],mid_dim=mid_dim)
        self.disco = DISCOBlock(mid_dim, mid_dim, in_shape=shape, out_shape=shape, basis_type=basis_type)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.InstanceNorm2d(mid_dim)
        self.act = nn.Tanh()
        
    def forward(self, x):
        # 保存原始数据类型
        orig_dtype = x.dtype
        
        # 并行处理
        nufft_out,params = self.nufft(x.float())
        disco_out = self.disco(x.float())
        pdb.set_trace()
        
        # 相加结果
        out = nufft_out + disco_out
        out = self.act(out)
        
        # 后续处理
        out = self.dropout(out)
        out = self.norm(out)
        
        return out.to(orig_dtype), params

class DISCOBlock(nn.Module):
    def __init__(self, in_chans, out_chans, in_shape, out_shape, kernel_shape=[5,6],basis_type="piecewise linear"):
        super().__init__()
        self.conv = DiscreteContinuousConvS2(
            in_chans,
            out_chans,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid_in="equiangular",
            grid_out="equiangular",
            groups=1,
            bias=False,
            theta_cutoff=4.0 * torch.pi / float(out_shape[0] - 1),
        )

    def forward(self, x):
        dtype = x.dtype
        with amp.autocast(device_type="cuda", enabled=False):
            x = self.conv(x.float())
            x = x.to(dtype=dtype)
        return x


class ResampleBlock(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.resample = ResampleS2(*in_shape, *out_shape, grid_in="equiangular", grid_out="equiangular")

    def forward(self, x):
        return self.resample(x)



class DoubleConv(nn.Module):
    """经典 U-Net 的 DoubleConv：第一个卷积改变通道数，第二个保持通道数"""
    def __init__(self, in_chans, out_chans, in_shape=None, out_shape=None, basis_type="piecewise linear", act=nn.GELU()):
        super().__init__()
        self.conv1 = DISCOBlock(in_chans, out_chans, in_shape=in_shape, out_shape=out_shape, basis_type=basis_type)
        self.act1 = act
        # self.conv2 = DISCOBlock(out_chans, out_chans, in_shape=in_shape, out_shape=out_shape, basis_type=basis_type)
        # self.act2 = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        # x = self.conv2(x)
        # x = self.act2(x)
        return x



class Stacked_Unet_Params(nn.Module):
    def __init__(self,
                 in_chans=2,
                 out_chans=2,
                 img_shape=(128, 256),
                 embed_dim=64,
                 mid_dim=7,
                 basis_type="piecewise linear",
                 num_down=4,
                 num_layers=3,
                 fourier_channel=1):  # 现在表示堆叠 U-Net 的数量
        super().__init__()

        # 创建 num_layers 个 FullSphericalUNetNUFFT3D
        self.unets = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_chans if i == 0 else out_chans  # 第一个网络输入是原始输入，其余输入是前一层输出
            self.unets.append(
                FullSphericalUNetNUFFT3D(
                    in_chans=in_c,
                    out_chans=out_chans,
                    img_shape=img_shape,
                    embed_dim=embed_dim,
                    mid_dim=mid_dim,
                    basis_type=basis_type,
                    num_down=num_down,
                    fourier_channel=fourier_channel
                )
            )

    # def forward(self, x):
    #     for unet in self.unets:
    #         x = unet(x)
    #     return x

    def forward(self, x):
        all_params = []
        for unet in self.unets:
            x, params = unet(x)
            all_params.append(params)
        return x, all_params


class FullSphericalUNetNUFFT3D(nn.Module):
    def __init__(self,
                 in_chans=2,
                 out_chans=2,
                 img_shape=(128, 256),
                 embed_dim=64,
                 mid_dim=7,
                 basis_type="piecewise linear",
                 num_layers=2,
                 num_down=4,
                 fourier_channel=1):
        super().__init__()

        self.img_shape = img_shape
        self.num_down = num_down

        # 计算各层的形状
        shapes = [img_shape]
        current_shape = img_shape
        for i in range(num_down):
            h = ((current_shape[0] - 1) // 2 + 1, current_shape[1] // 2)
            shapes.append(h)
            current_shape = h

        # 输入卷积层（DoubleConv）
        self.input_conv = DoubleConv(in_chans, embed_dim, in_shape=shapes[0], out_shape=shapes[0])

        # Encoder
        self.enc_down_layers = nn.ModuleList()
        self.enc_conv_layers = nn.ModuleList()
        
        current_ch = embed_dim
        for i in range(num_down):
            next_ch = embed_dim * (2 ** i) if i < num_down - 1 else mid_dim
            self.enc_down_layers.append(ResampleBlock(shapes[i], shapes[i+1]))
            if i < num_down - 1:
                self.enc_conv_layers.append(DoubleConv(current_ch, next_ch, in_shape=shapes[i+1], out_shape=shapes[i+1]))
            else:
                self.enc_conv_layers.append(DISCOBlock(current_ch, next_ch, in_shape=shapes[i+1], out_shape=shapes[i+1], basis_type=basis_type))
            current_ch = next_ch

        self.act = nn.GELU()

        # Core layers
        # self.core_layers = nn.ModuleList()
        # for _ in range(num_layers):
        #     self.core_layers.append(ParallelCoreLayer(mid_dim, fourier_channel,shapes[-1], basis_type))
        
        self.core_layer_unit = ParallelCoreLayer(mid_dim, fourier_channel,shapes[-1], basis_type)

        # Decoder
        self.dec_up_layers = nn.ModuleList()
        self.dec_conv_layers = nn.ModuleList()
        for i in range(num_down - 1, 0, -1):
            in_channels = current_ch + (embed_dim * (2 ** (i-1)) if i > 1 else embed_dim)
            out_channels = embed_dim * (2 ** (i-1)) if i > 1 else embed_dim
            self.dec_up_layers.append(ResampleBlock(shapes[i+1], shapes[i]))
            self.dec_conv_layers.append(DISCOBlock(in_channels, out_channels, in_shape=shapes[i], out_shape=shapes[i], basis_type=basis_type))
            current_ch = out_channels

        # 最后一层 Decoder
        self.dec_up_layers.append(ResampleBlock(shapes[1], shapes[0]))
        self.dec_conv_layers.append(DISCOBlock(embed_dim + embed_dim, embed_dim, in_shape=shapes[0], out_shape=shapes[0], basis_type=basis_type))

        # 输出卷积 1x1
        # self.out_conv = nn.Conv2d(embed_dim, out_chans, kernel_size=1)
        self.out_conv = DISCOBlock(embed_dim, out_chans, in_shape=shapes[0], out_shape=shapes[0], kernel_shape=[1,1],basis_type=basis_type)
     

    def forward(self, x):
        skips = []

        # 输入卷积
        x = self.input_conv(x)
        skips.append(x)

        # Encoder
        for i in range(self.num_down):
            x = self.enc_down_layers[i](x)
            x = self.enc_conv_layers[i](x)
            x = self.act(x)
            if i < self.num_down - 1:
                skips.append(x)

       
        # Core (single layer, only one NUFFT block)
        orig_dtype = x.dtype
        params_list = []

        out, params = self.core_layer_unit(x.float())
        x = out.to(orig_dtype)

        # params is shape: [fourier_channel parameters]
        if params is not None:
            params_list.append(params)

        # Decoder
        for i in range(self.num_down):
            x = self.dec_up_layers[i](x)
            skip = skips[self.num_down - i - 1] if i < self.num_down - 1 else skips[0]
            x = torch.cat([x, skip], dim=1)
            x = self.dec_conv_layers[i](x)
            x = self.act(x)


        # 输出卷积
        x = self.out_conv(x)
        return x,params_list


# # # # ===== 测试 =====
# if __name__ == "__main__":

#     num_layers = 4

#     scale_factor=2
#     hard_thresholding_fraction =1

#     embed_dim = 4
#     fourier_channel = 2
#     num_down = 2


#     model = Stacked_Unet_Params(in_chans=2,out_chans=2,
#                                 img_shape=(256, 512),
#                                 embed_dim=embed_dim, 
#                                 mid_dim=embed_dim*num_down,
#                                 num_layers=num_layers,num_down=num_down,
#                                 fourier_channel=fourier_channel)

#     x = torch.randn(4, 2, 256, 512)
#     y, params = model(x)

#     print("Output shape:", y.shape)
#     print("Num UNet blocks:", len(params))
#     print("Params of UNet 0:", params[0])   # 是 list
#     print("Params keys of UNet 0 layer 0:", params[0][0].keys())



