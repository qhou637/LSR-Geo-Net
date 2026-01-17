import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.optim.lr_scheduler import OneCycleLR,StepLR
from torch.utils.data import DataLoader, Subset
from torch_harmonics.examples.models.lsno import LocalSphericalNeuralOperatorNet as LSFNO
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from unet_test_param import Stacked_Unet_Params
from AC_Autoreg_trainer_LSR import Autoreg_LR_Trainer_LSR

from plot_ac_comp import *
from math import ceil
import pdb
pdb.set_trace = lambda: None
import time
import os
from torch_harmonics.examples import ShallowWaterSolver
from ac_dataset import train_val_dataset
from torch_harmonics.examples import SphereSolver
=
from AC_Autoreg_trainer import Autoreg_LR_Trainer
cmap='twilight_shifted'

enable_amp = False
import torch
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0))


def generate_inference_times(t, total_time):
    times = list(range(t+1, total_time, t))
    if times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times


def generate_test_times(t, total_time):
    times = list(range(t, total_time, t))
    if not times and t <= total_time: 
        times = [t]
    elif times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times

def main(model_times):
    lr = 1e-3
    num_layers = 4
    nepochs =50
    scale_factor=2
    hard_thresholding_fraction =1
    weight_decay = 1e-5
    total_time= 12
    T_all =  [3]
    eps= 0.001
    embed_dim = 7
    fourier_channels_all = [2]
    num_down = 4


  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = torch.load(f'/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/data/256_ac_grf_nu01_equ_eps{eps}_T12s_num400_inp_tar.pt', weights_only=False)
  
    forward_steps = [4]
   

   
    for model_time in model_times:
        for T in T_all:
            for forward_step in forward_steps:
                for  fourier_channel in  fourier_channels_all:
          
                    idx = T
                    # Teaching Forcing

                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/l2_loss/Multistep{forward_step}/LSFNO/model{model_time}/embed{embed_dim}/Infer_T{T}/eps{eps}_alpha{weight_decay}_AC_layer{num_layers}_sf{scale_factor}/ks3_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                


                    outf = f'/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/l2_loss/TFstep{forward_step}/Full_Unet_LSR_Deconv5/model{model_time}/embed{embed_dim}_fourier_ch{fourier_channel}_down{num_down}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks3_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                   os.makedirs(outf, exist_ok=True)



                    ###Auto

                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/l2_loss/Autostep{forward_step}/LSFNO/model{model_time}/embed{embed_dim}/Infer_T{T}/eps{eps}_alpha{weight_decay}_AC_layer{num_layers}_sf{scale_factor}/ks3_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/l2_loss/Autostep{forward_step}/Full_Unet_LSR_Deconv5/model{model_time}/embed{embed_dim}_fourier_ch{fourier_channel}_down{num_down}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks3_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                   # os.makedirs(outf, exist_ok=True)
            
            
            
                
                
                    # 使用自定义方法划分数据集
                    datasets = train_val_dataset(dataset, seed=999, test_split=0.2, val_split=0.2)

                    # 创建 DataLoader
                    batch_size = 4
                    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)


                    #### on the earth 
                    # 初始化参数
                    nlat = 256
                    nlon = 2 * nlat
                    lmax = ceil(128)
                    mmax = lmax
                    dt = 60
                

                    # 初始化 solver
                    solver = ShallowWaterSolver(nlat, nlon, dt, lmax=lmax, mmax=mmax).to(device)

            


                    
                    # model = LSFNO(img_size=(nlat, nlon), grid="equiangular",
                    #                 num_layers=num_layers, scale_factor=scale_factor, embed_dim=embed_dim, big_skip=True,
                    #                 pos_embed="lat", use_mlp=False, normalization_layer="none",  in_chans=1,
                    #                 out_chans=1).to(device)


                    model = Stacked_Unet_Params(in_chans=1,out_chans=1,img_shape=(nlat,nlon),embed_dim=embed_dim, mid_dim=embed_dim*num_down,num_layers=num_layers,num_down=num_down,fourier_channel=fourier_channel).to(device)





                    model = model.to(device)


                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)


                    all_times = generate_test_times(T, total_time)
                    all_steps = [t for t in all_times]
                    inference_times = all_times[1:]




                    # trainer =Autoreg_LR_Trainer( solver=solver,
                    #     model=model,
                    #     train_loader=train_loader,
                    #     val_loader= val_loader,
                    #     test_loader=test_loader,
                    #     nepochs=nepochs,
                    #     optimizer=optimizer,
                    #     scheduler=scheduler, 
                    #     loss_fn='l2',              
                    #     save_path=outf,
                    #     all_steps=all_steps,
                    #     device = device,
                    #     idx=idx,
                    #     forward_step=forward_step
                    # )

                    trainer =Autoreg_LR_Trainer_LSR( solver=solver,
                        model=model,
                        train_loader=train_loader,
                        val_loader= val_loader,
                        test_loader=test_loader,
                        nepochs=nepochs,
                        optimizer=optimizer,
                        scheduler=scheduler, 
                        loss_fn='l2',              
                        save_path=outf,
                        all_steps=all_steps,
                        device = device,
                        idx=idx,
                        forward_step=forward_step
                    )


            

                    trainer.train()
                    prd,test_loss,_,_,= trainer.test(test_loader)

                
                    trainer.test_inference(prd,test_loss=test_loss)
                    solver.lons.data = solver.lons.detach().cpu()
                    solver.lats.data = solver.lats.detach().cpu()


                
                    plot_test_results(solver=solver, save_path=outf)
                    plot_infer_results(solver=solver, inference_times=inference_times,save_path=outf)
                    


if __name__ == "__main__": 
    model_times=[1,2]
    main(model_times)
