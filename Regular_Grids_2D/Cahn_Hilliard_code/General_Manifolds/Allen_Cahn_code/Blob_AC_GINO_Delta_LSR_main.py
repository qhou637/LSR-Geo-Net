


import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, '/home/qianhou/torch-harmonics')

from GINO_Delta_LSR import GINO_DeltaConv_LSR,GINO_DeltaConv_LSR_MultiLayer 
from blob_ac_dataset_delta import get_train_val_test_loaders 
from Blob_AC_Autoreg_trainer_LSR import Autoreg_Trainer_LSR  
from plot_results import plot_test_results_point_cloud, plot_infer_results_point_cloud  


import pdb
pdb.set_trace = lambda: None


def generate_times_and_steps(T, total_time, dt=0.01, save_interval=500):
    """
    根据物理时间间隔 T 和总物理时间 total_time，生成对应的物理时间列表和时间步索引列表。
    参数:
    T (float): 物理时间间隔。
    total_time (float): 总物理时间。
    dt (float): 每个模拟步的时间（来自MATLAB代码）。
    save_interval (int): 每隔多少模拟步保存一次（来自MATLAB代码）。
    返回:
    tuple: (all_times, all_steps, inference_times, inference_steps)
    """

    time_per_step = dt * save_interval

  
    all_times = list(range(T, total_time + 1, T))
    

    if all_times and all_times[-1] > total_time:
        all_times.pop()


    if not all_times:
        print(f"警告: T={T} 大于 total_time={total_time}，无法生成时间列表！")
        return [], [], [], []


    all_steps = [int(t / time_per_step) for t in all_times]
    

    inference_times = all_times[1:] if len(all_times) > 1 else all_times
    inference_steps = all_steps[1:] if len(all_steps) > 1 else all_steps

    return all_times, all_steps, inference_times, inference_steps


def train_models(
    model_times=[1],
    T_all=[10],
    forward_steps=[3],
    nufft_fno_channels_all=[5],
    delta_num_neighbors_all=[20],
    num_layers_all=[1]
    
):
    for model_time in model_times:
        for T in T_all:
            for forward_step in forward_steps:
                for nufft_fno_channel in nufft_fno_channels_all:
                    for delta_num_neighbors in delta_num_neighbors_all:
                        for num_layers in num_layers_all:
                            # ---------------------------
                            # 1.  config
                            # ---------------------------
                            config = {
                                "mat_file_path": "/home/qianhou/neuraloperator/examples/models/Blob_AC/GILR_Delta/blob_history_400samples.mat",
                                "batch_size": 1,
                                "epochs": 25,
                                "lr": 1e-3,
                                "weight_decay": 1e-5,
                                "in_channels": 1,
                                "out_channels": 1,
                                "gno_radius": 0.2,  # 动态
                                "num_layers":2,
                                "projection_channels": 7,
                                "nufft_n_mesh": 10,
                                "nufft_fno_channel": nufft_fno_channel,  # 动态
                                "padding_ratio": 0.2,
                                "delta_conv_channels": [64, 128, 256], # case2
                                # "delta_conv_channels": [32,64,128],
                                "delta_mlp_depth": 2,
                                "delta_embedding_size": 64, #case2
                                #  "delta_embedding_size": 32,
                                "delta_num_neighbors": delta_num_neighbors,  # 动态
                                "delta_grad_reg": 1e-4,
                                "delta_kernel_width": 1.0,
                                "T": T,                         # 动态
                                "total_time": 20,
                                "forward_step": forward_step,    # 动态
                            }

                           
                            config["save_path"] = (
                                f"/home/qianhou/neuraloperator/examples/models/Blob_AC/GILR_Delta/Deconv5_MLP_l2_loss/"
                                f"TFstep{config['forward_step']}/grid_size{config['nufft_n_mesh']}/"
                                # f"Autostep{config['forward_step']}/grid_size{config['nufft_n_mesh']}/"
                                f"Model{model_time}/"
                                f"GINO_DeltaConv_LSR/"
                                f"T{config['T']}_Total{config['total_time']}/"
                                f"layer{num_layers}_nufft_ch{config['nufft_fno_channel']}_case2_delta_neigh{config['delta_num_neighbors']}"
                                f"_alpha{config['weight_decay']}_epoch{config['epochs']}_lr{config['lr']}/"
                            )
                            os.makedirs(config["save_path"], exist_ok=True)

                            # ---------------------------
                            # 2. device
                            # ---------------------------
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            print(f"[*] 使用设备: {device}")
                            if torch.cuda.is_available():
                                print(f"[*] GPU名称: {torch.cuda.get_device_name(0)}")

                            # ---------------------------
                            # 3. load data
                            # ---------------------------
                            train_loader, val_loader, test_loader = get_train_val_test_loaders(
                                mat_file_path=config["mat_file_path"],
                                train_batch_size=config["batch_size"],
                                val_batch_size=config["batch_size"],
                                test_batch_size=1,
                                train_split=0.8,
                                val_split=0.1,
                                test_split=0.1,
                                num_samples=None,
                                seed=999
                            )

                            # ---------------------------
                            # 4. grid calculation
                            # ---------------------------
                            all_pos = []
                            for data in train_loader:
                                all_pos.append(data.pos.cpu())
                            all_pos = torch.cat(all_pos, dim=0)
                            min_pos = all_pos.min(dim=0)[0]
                            max_pos = all_pos.max(dim=0)[0]
                            padding = config["padding_ratio"] * (max_pos - min_pos).max()
                            data_x_lims = [(min_pos - padding).min().item(), (max_pos + padding).max().item()]

                            # ---------------------------
                            # 5. DeltaConv
                            # ---------------------------
                            delta_pde_params = {
                                'in_channels': config["in_channels"],
                                'out_channels': config["out_channels"],
                                'conv_channels': config["delta_conv_channels"],
                                'mlp_depth': config["delta_mlp_depth"],
                                'embedding_size': config["delta_embedding_size"],
                                'num_neighbors': config["delta_num_neighbors"],
                                'grad_regularizer': config["delta_grad_reg"],
                                'grad_kernel_width': config["delta_kernel_width"],
                            }

                         

                            model = GINO_DeltaConv_LSR_MultiLayer(
                                in_channels=config["in_channels"],
                                out_channels=config["out_channels"],
                                num_layers = config['num_layers'],
                                delta_pde_params=delta_pde_params,
                                gno_coord_dim=3,
                                gno_radius=config["gno_radius"],
                                projection_channels=config["projection_channels"],
                                nufft_n_mesh=config["nufft_n_mesh"],
                                nufft_x_lims=data_x_lims,
                                nufft_fno_channel=config["nufft_fno_channel"],
                                gno_embed_channels=32,
                                gno_embed_max_positions=10000,
                                in_gno_transform_type='linear',
                                out_gno_transform_type='linear',
                                gno_pos_embed_type='transformer',
                                # in_gno_channel_mlp_hidden_layers=[80, 80, 80],
                                # out_gno_channel_mlp_hidden_layers=[512, 256],
                                in_gno_channel_mlp_hidden_layers=[64, 64, 64],
                                out_gno_channel_mlp_hidden_layers=[128, 128],
                                gno_channel_mlp_non_linearity=nn.functional.tanh,
                                gno_use_open3d=False,
                                gno_use_torch_scatter=True,
                                out_gno_tanh=None,
                                fno_hidden_channels=32,
                                fno_lifting_channel_ratio=2,
                                fno_non_linearity=nn.functional.tanh,
                                use_mlp_fusion=True
                            ).to(device)

                            # ---------------------------
                            # 7. 优化器+损失+调度器
                            # ---------------------------
                            criterion = nn.MSELoss(reduction='mean').to(device)
                            optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

                            # ---------------------------
                            # 8. 时间步映射
                            # ---------------------------
                            all_times, all_steps, inference_times, inference_steps = generate_times_and_steps(
                                T=config["T"],
                                total_time=config["total_time"],
                                dt=0.01,
                                save_interval=500
                            )

                            # # ---------------------------
                            # # 9. 自回归训练器
                            # # ---------------------------
                            trainer = Autoreg_Trainer_LSR(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                test_loader=test_loader,
                                nepochs=config["epochs"],
                                optimizer=optimizer,
                                criterion=criterion,
                                scheduler=scheduler,
                                save_path=config["save_path"],
                                device=device,
                                all_steps=all_steps,
                                forward_step=config["forward_step"]
                            )

                            # # ---------------------------
                            # # 10. 训练+测试+推理
                            # # ---------------------------
                            print(f"\n[*] 开始训练 Model{model_time}, T={T}, forward_step={forward_step}, "
                                  f"nufft_ch={nufft_fno_channel}, delta_neighbors={delta_num_neighbors}, layer={num_layers}")
                            trainer.train()
                            prd_t1, test_loss_t1, tar_t1, inp_t0, pos = trainer.test()
                            trainer.test_inference(prd_t1,test_loss_t1)

                            # ---------------------------
                            # 11. 可视化
                            # ---------------------------
                            plot_test_results_point_cloud(
                                save_path=config["save_path"],
                                mat_file_path=config["mat_file_path"],
                                device=device, T=T,interval=10
                            )
                            plot_infer_results_point_cloud(
                                save_path=config["save_path"],
                                mat_file_path=config["mat_file_path"],
                                inference_times=inference_times,
                                device=device, interval=10
                            )

if __name__ == "__main__":
    train_models(
        model_times=[1,2,3],
        T_all=[5],
        forward_steps=[4],
        nufft_fno_channels_all=[2],
        delta_num_neighbors_all=[30],
        num_layers_all=[3]  
    )