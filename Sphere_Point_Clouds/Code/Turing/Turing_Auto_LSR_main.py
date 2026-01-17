import sys
sys.path.append('/home/qianhou/torch-harmonics')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.optim.lr_scheduler import OneCycleLR,StepLR, CosineAnnealingLR

from torch.utils.data import DataLoader, Subset
from torch_harmonics.examples.models.lsno import LocalSphericalNeuralOperatorNet as LSFNO
from torch_harmonics.neuralop.models.fno import FNO2d,FNO

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from unet_test_param import Stacked_Unet_Params
from Turing_Autoreg_trainer_LSR import Autoreg_LR_Trainer_LSR
from plot_turing_comp import * 
from math import ceil
import pdb
import time
import os
from torch_harmonics.examples.Turing_equations import TuringSphereSolver
from Turing_dataset import train_val_dataset


from Turing_Autoreg_trainer import Autoreg_LR_Trainer # For SFNO Training
cmap='twilight_shifted'

enable_amp = False
import torch
# print("PyTorch CUDA version:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU device:", torch.cuda.get_device_name(0))



def generate_test_times(t, total_time):
    times = list(range(t, total_time, t))
    if not times and t <= total_time:  
        times = [t]
    elif times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times

def main(num_downs):
    
    lr = 1e-3
    num_layers = 4
    nepochs =50
    scale_factor=2
    hard_thresholding_fraction =1
    weight_decay = 1e-5
    total_time= 400
    T_all =  [130]
    embed_dim = 7
    fourier_channels_all = [3]
    # num_down = 4
    pdb.set_trace()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  
    dataset = torch.load('/home/qianhou/torch-harmonics/notebooks/Turing/turing_data/turing_sphere_num500_T400.00s.pt', weights_only=False)

    forward_steps = [3]
    model_time = 1
   
    for num_down in num_downs:
        for T in T_all:
            for forward_step in forward_steps:
                for fourier_channel in fourier_channels_all:
             =
                    idx = T//10

                    #Teaching Forcing

                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/Multistep{forward_step}/LSFNO/model{model_time}/embed{embed_dim}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks5_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
        

                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/TFstep{forward_step}/Full_Unet_LSR_Deconv5/model{model_time}/embed{embed_dim}_fourier_ch{fourier_channel}_down{num_down}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks5_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                    
                    # os.makedirs(outf, exist_ok=True)
                


 


                    #Auto

                    # outf = f'/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/Autostep{forward_step}/LSFNO_embed{embed_dim}/model{model_time}/embed{embed_dim}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks5_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'
                   

                    outf = f'/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/Autostep{forward_step}/Full_Unet_LSR_Deconv5/model{model_time}/embed{embed_dim}_fourier_ch{fourier_channel}_down{num_down}/Infer_T{T}/alpha{weight_decay}_layer{num_layers}_sf{scale_factor}/ks5_ratio{hard_thresholding_fraction}_epoch{nepochs}_lr{lr}'

                    os.makedirs(outf, exist_ok=True)
                 
                
                

                    datasets = train_val_dataset(dataset, seed=999, test_split=0.2, val_split=0.2)

                    # DataLoader
                    batch_size = 4
                    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)

                



                    #### on the earth 
             
                    nlat = 256
                    nlon = 2 * nlat
                    lmax = ceil(nlon / 2)
                    mmax = lmax
                    dt = 0.1


                    # Initialize solver

      
                    delta_v = 1e-3
                    delta_u = 0.516 * delta_v
                    alpha = 0.899
                    beta = -0.91
                    gamma = -0.899
                    tau1 = 0.02
                    tau2 = 0.2

                    # ------------------- 初始化 Solver -------------------
                    solver = TuringSphereSolver(
                        nlat, nlon, dt, 
                        delta_u=delta_u, delta_v=delta_v,
                        alpha=alpha, beta=beta, gamma=gamma,
                        tau1=tau1, tau2=tau2,
                        lmax=lmax, mmax=mmax
                    ).to(device)

           


                


  
                    
                    # model = LSFNO(img_size=(nlat, nlon), grid="equiangular",
                    #                 num_layers=num_layers, scale_factor=scale_factor, embed_dim=embed_dim, big_skip=True,
                    #                 pos_embed="lat", use_mlp=False, normalization_layer="none",  in_chans=2,
                    #                 out_chans=2).to(device)

                    

                    # pdb.set_trace()
                    model = Stacked_Unet_Params(in_chans=2,out_chans=2,img_shape=(nlat,nlon),embed_dim=embed_dim, mid_dim=embed_dim*num_down,num_layers=num_layers,num_down=num_down,fourier_channel=fourier_channel).to(device)

                  





                    model = model.to(device)



     
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = StepLR(optimizer, step_size=16, gamma=0.5)
                



   
                    # pdb.set_trace()
                    all_times = generate_test_times(T, total_time)
                    all_steps = [t  // 10 for t in all_times]


                    # pdb.set_trace()
                    inference_steps = all_steps[1:]
        
            
                    # For SFNO Trianer

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
                    plot_infer_results(solver=solver, inference_steps=inference_steps,save_path=outf)
                    


                   
if __name__ == "__main__": 
    num_downs=[3]
    pdb.set_trace()
    main(num_downs)
