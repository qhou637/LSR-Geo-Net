import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from CH_GRF_dataset import CH_GRF_Dataset, train_val_dataset
from Gauss_Normal_CH_Wise_Pre_Channel_Network import Wise_SpectralTransform
from torch.utils.data import DataLoader
from torchvision import transforms
from plot_difference_ch import *
from plot_energy_ch_k8_me05 import *
import os
import torch.nn as nn
import numpy as np
import csv
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

pdb.set_trace = lambda: None
# 生成推断时间点的函数
def generate_inference_times(t, total_time):
    times = list(range(t * 2, total_time, t))
    if times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times

# 生成test时间点的函数
def generate_test_times(t, total_time):
    times = list(range(t, total_time, t))
    if times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times

def load_model(model_path, net, device):
    """加载单个模型"""
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    net.eval()
    return net


def load_initial_parameters(model_base_path, model_index, T, total_time, num_layers, features, kernel_size, batch_size, fourier_channels):
    """加载指定模型的初始参数"""
    initial_parameters_path = os.path.join(
        model_base_path, 
        f'model{model_index}', 
        f'CH_T{T}s_T{total_time}', 
        f'fourier_ch_{fourier_channels}Layer{num_layers}_channel{features}_ks_{kernel_size}', 
        '_a_cir_999_pre_epoch_10', 
        f'bs{batch_size}_LR_lr0.001_epoch50', 
        'initial_parameters4.pth'
    )
    
    if os.path.exists(initial_parameters_path):
        # 确保加载时的设备一致性
        initial_parameters = torch.load(initial_parameters_path, map_location='cuda')
        return initial_parameters
    else:
        # 这个分支实际上不会执行，因为文件应该存在
        print(f"Initial parameters path does not exist: {initial_parameters_path}")
        raise FileNotFoundError(f"Initial parameters file not found: {initial_parameters_path}")

def predict_with_models_ensemble(model_paths, nets, data, device):
    """使用所有模型进行预测并对结果取平均"""
    predictions = []
    
    for i, model_path in enumerate(model_paths):
        # 加载模型
        nets[i] = load_model(model_path, nets[i], device)
        
        # 进行预测
        with torch.no_grad():
            out_test, *_ = nets[i](data)
            predictions.append(out_test)
    
    # 对所有模型的预测结果取平均
    if predictions:
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction
    else:
        return None

def compute_test_energy_ensemble(model_paths, test_loader, idx, device, nets):
    """计算测试数据的能量（模型集成版本）"""
    all_energies_pred = []
    all_energies_target = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(device)
            target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to(device)
            
            # 使用集成模型进行预测
            out_test = predict_with_models_ensemble(model_paths, nets, data, device)
            
            # 计算能量
            energy_pred = compute_cahn_hilliard_energy(out_test.cpu().numpy())
            energy_target = compute_cahn_hilliard_energy(target.cpu().numpy())
            
            all_energies_pred.append(energy_pred)
            all_energies_target.append(energy_target)

    return np.mean(all_energies_pred), np.mean(all_energies_target)

def compute_infer_energy_ensemble(model_paths, test_loader, idx, device, prev_data, nets):
    """计算推理数据的能量（模型集成版本）"""
    all_energies_pred = []
    all_energies_target = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if idx < batch.shape[1]:
                target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to(device)
                
                # 使用集成模型进行预测
                out_test = predict_with_models_ensemble(model_paths, nets, prev_data[i], device)
                
                # 计算能量
                energy_pred = compute_cahn_hilliard_energy(out_test.cpu().numpy())
                energy_target = compute_cahn_hilliard_energy(target.cpu().numpy())
                
                all_energies_pred.append(energy_pred)
                all_energies_target.append(energy_target)

    return np.mean(all_energies_pred), np.mean(all_energies_target)

# -------------------------- 新增：计算T0（初始时刻）能量的函数 --------------------------
def compute_t0_energy(test_loader, device):
    """计算初始时刻（T0，即输入数据本身）的能量"""
    t0_energy_list = []
    with torch.no_grad():
        for batch in test_loader:
            # T0数据为batch的第0个时间步
            t0_data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(device)
            # 计算能量并添加到列表
            t0_energy = compute_cahn_hilliard_energy(t0_data.cpu().numpy())
            t0_energy_list.append(t0_energy)
    # 返回T0能量的平均值
    return np.mean(t0_energy_list)

def plot_mse_results(all_mse, T, save_path):
    """绘制MSE结果图"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    
    fig, ax = plt.subplots(figsize=(7.2, 4.8))  # PRL 样式尺寸
    
    num_steps = len(all_mse)
    x_labels = ["T(test)"] + [f"{i+1}T" for i in range(1, num_steps)]
    mse_values = [mse_value if isinstance(mse_value, (int, float)) else mse_value.item() 
                  for mse_value in all_mse]
    
    ax.plot(x_labels, mse_values, marker='o', linestyle='--', color='blue', linewidth=2, markersize=6)
    
    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('MSE Loss', fontsize=font_size)
    ax.set_title(f"MSE Loss: T={T}", fontsize=font_size + 2)
    ax.tick_params(axis='both', labelsize=font_size)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mse_loss_over_time.png'), dpi=300)
    plt.close()

def save_all_energy_snapshots(out, x_labels, save_dir, T, prefix):
    """保存所有时间步对应的小图（尺寸更紧凑）"""
    os.makedirs(save_dir, exist_ok=True)
    sample_idx = 0
    num_steps = len(out)

    for idx in range(num_steps):
        img = out[idx][sample_idx].squeeze()

        fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=300)  # 控制小图尺寸
        ax.imshow(img, cmap='jet', interpolation='none')  # 保留清晰像素
        ax.set_title(x_labels[idx], fontsize=12)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0)  # 减小边距

        clean_label = x_labels[idx].split('(')[0]  # 处理 T1(test)
        filename = f"{prefix}_energy_snapshot_{clean_label}_T{T}s.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def plot_energy_results(all_energy_pred, all_energy_target, T, save_path, all_out_test):
    """绘制能量结果图（横坐标为0, T, 2T...格式，无1T）"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    
    num_steps = len(all_energy_pred)
    # 生成0, T, 2T...格式（i=1时显示T，i≥2时显示2T,3T...）
    x_labels = ["0"]  # 初始时刻固定为0
    for i in range(1, num_steps):
        if i == 1:
            x_labels.append("T")  # 第一个间隔显示为T（而非1T）
        else:
            x_labels.append(f"{i}T")  # 后续为2T,3T...
    
    ax.plot(x_labels, all_energy_pred, marker='o', linestyle='-', color='blue', 
            linewidth=2, markersize=6, label='Predicted Energy')
    ax.plot(x_labels, all_energy_target, marker='o', linestyle='--', color='red', 
            linewidth=2, markersize=6, label='Target Energy')
    
    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Energy', fontsize=font_size)
    ax.set_title(f"Cahn Hiiliard Energy: T={T}", fontsize=font_size + 2)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size-4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'energy_over_time.png'), dpi=300)
    plt.close()
    
    # 保存能量快照（使用修正后的标签）
    full_save_path = os.path.join(save_path, "Energy_PIC")
    os.makedirs(full_save_path, exist_ok=True)
    save_all_energy_snapshots(all_out_test, x_labels, full_save_path, T, "FFT")

def plot_ensemble_test_results(model_paths, nets, test_loader, idx, save_path, layer, sample_indices=[0, 10, 20, 30]):
    """绘制集成模型的测试结果，模仿plot_test函数"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0 or i in sample_indices:
                # 使用0时刻的真实值作为输入
                data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to('cuda')
                target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to('cuda')
                
                # 使用集成模型进行预测
                out_test = predict_with_models_ensemble(model_paths, nets, data, 'cuda')
                
                # 构造mid_list用于可视化
                # 模拟模型输出结构，这里我们直接使用预测值、真实值等构造可视化需要的数据结构
                mid_list = [[data], [out_test], [target]]
                
                if i % 10 == 0:
                    plot_diff_ensemble(mid_list, save_path, layer, i, idx, prefix='ensemble_test')
                
                if i in sample_indices:
                    plot_single_diff_ensemble(mid_list, save_path, layer, sample_idx=i, step_idx=idx, prefix='ensemble_test')

def plot_ensemble_inference_results(model_paths, nets, test_loader, idx, save_path, layer, prev_data, sample_indices=[0, 10, 20, 30]):
    """绘制集成模型的推理结果，模仿plot_infer函数"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
                # 使用前一个时刻的预测值作为输入（这是5个模型平均后的结果）
                data = prev_data[i]  # 前一个时刻5个模型平均后的预测结果（这里是1时刻）
                # pdb.set_trace()
                target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to('cuda')  # 下一个时刻的真实值（这里是2时刻）
                
                # 使用集成模型进行预测（预测idx+1时刻，即2时刻）
                out_test = predict_with_models_ensemble(model_paths, nets, data, 'cuda')
                
                # 构造mid_list用于可视化
                mid_list = [[data], [out_test], [target]]
                
                if i % 10 == 0:
                    plot_diff_ensemble(mid_list, save_path, layer, i, idx, prefix=f'ensemble_step_{idx}')
                
                if i in sample_indices:
                    plot_single_diff_ensemble(mid_list, save_path, layer, sample_idx=i, step_idx=idx, prefix=f'ensemble_step_{idx}')

def plot_diff_ensemble(mid_outputs, save_path, layer, i, idx, prefix='inference'):
    """绘制输入、目标、预测和误差，模仿plot_diff函数"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    difference_path = os.path.join(save_path, 'color_difference')
    os.makedirs(difference_path, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    input_data = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()
    target = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()
    out = mid_outputs[1][0].squeeze(0).squeeze(0).cpu().numpy()  # 预测值在索引1位置

    # 计算颜色范围（全图共享）
    common_vmin = min(input_data.min(), target.min(), out.min())
    common_vmax = max(input_data.max(), target.max(), out.max())
    error_vmin = 0.0
    error_vmax = np.max(np.abs(out - target))

    # 设置字体大小
    font_size = 25

    # 绘图
    im0 = axes[0].imshow(input_data, cmap='jet')
    axes[0].set_title('Initial Condition', fontsize=font_size)

    im1 = axes[1].imshow(target, cmap='jet', vmin=common_vmin, vmax=common_vmax)
    axes[1].set_title('Ground Truth', fontsize=font_size)

    im2 = axes[2].imshow(out, cmap='jet', vmin=common_vmin, vmax=common_vmax)
    axes[2].set_title('Prediction', fontsize=font_size)

    absolute_error = np.abs(out - target)
    im3 = axes[3].imshow(absolute_error, cmap='jet', vmin=error_vmin, vmax=error_vmax)
    axes[3].set_title('Absolute Error', fontsize=font_size)

    # 统一 colorbar 设置
    for ax, im in zip(axes, [im0, im1, im2, im3]):
        if im == im0:  # 初始条件独立的colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=font_size)
        else:  # target和out共享colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=font_size)

    # 增加坐标轴字体大小
    for ax in axes:
        ax.tick_params(axis='both', labelsize=font_size)

    plt.tight_layout()
    filename = f"{prefix}_{idx}_{i}_{layer}_diff.png"
    plt.savefig(os.path.join(difference_path, filename), dpi=300)
    plt.close()

def plot_single_diff_ensemble(mid_outputs, save_path, layer, sample_idx, step_idx, prefix='inference'):
    """绘制单个样本的 Initial Condition、Ground Truth、Prediction 和 Absolute Error，模仿plot_single_diff函数"""
    plt.rcParams['font.family'] = 'Nimbus Roman'

    # 创建目录保存结果图像
    single_difference_path = os.path.join(save_path, 'single_difference_new', f'idx={sample_idx}')
    os.makedirs(single_difference_path, exist_ok=True)

    # 提取数据并 squeeze 成二维数组
    input_data = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()
    target = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()
    out = mid_outputs[1][0].squeeze(0).squeeze(0).cpu().numpy()  # 预测值在索引1位置

    # 设置颜色范围：Prediction 和 Ground Truth 共享
    common_vmin = min(input_data.min(), target.min(), out.min())
    common_vmax = max(input_data.max(), target.max(), out.max())
    error_vmin = 0.0
    error_vmax = np.max(np.abs(out - target))
    font_size = 25

    def save_image(data, title, filename, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
        # ax.set_title(title, fontsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size)

        # 取消坐标轴的网格线
        ax.set_xticks([])  # 去除 x 轴刻度线
        ax.set_yticks([])  # 去除 y 轴刻度线

        # colorbar 放在图下方
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=font_size)

        plt.tight_layout()
        plt.savefig(os.path.join(single_difference_path, filename), dpi=300)
        plt.close()

    # 依次保存每张图
    save_image(input_data, 'Initial Condition', f'{prefix}_{sample_idx}_{step_idx}_initial_condition.png')
    save_image(target, 'Ground Truth', f'{prefix}_{sample_idx}_{step_idx}_ground_truth.png', common_vmin, common_vmax)
    save_image(out, 'Prediction', f'{prefix}_{sample_idx}_{step_idx}_prediction.png', common_vmin, common_vmax)
    save_image(np.abs(out - target), 'Absolute Error', f'{prefix}_{sample_idx}_{step_idx}_absolute_error.png', error_vmin, error_vmax)


def set_deterministic(seed=999):
    """设置完全的确定性运行环境"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(num_layers, kernel_size, features, batch_size=4,  fourier_channels=1, model_source='Test_SOS', NpointsMesh=16):
    # 设置全局确定性
    seed = 999
    set_deterministic(seed)
    # 数据路径保持不变
    data_path = os.path.join(r'/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Case_v2_sos/k8_m0.5_ch_grf_T50_500.h5')
    device = 'cuda'

    forward_step = 1
    T_all = [7]  # 时间间隔列表
    total_time = 49  # 总时长
    
    # 构建模型基础路径
    if 'L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3/Test_SOS' in model_source:
        base_dir = '/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel'
        model_base_path = os.path.join(base_dir, model_source, f'Autostep{forward_step}')
    else:
        model_base_path = f'./{model_source}/Autostep{forward_step}'
    
    # 构建ensemble结果保存的根路径（指定的路径）
    # ensemble_root = '/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3/Test_SOS/Autostep1'

    ensemble_root = '/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3'
    
    
    pdb.set_trace()
    
    for T in T_all:
        idx_initial = T*10-1
        # ========== 核心修改：修改输出路径到指定位置 ==========
        outf = os.path.join(
            ensemble_root,  # 改为指定的根路径
            f'MoE_Res_{forward_step}',
            model_source.split('/')[-1],  # 取Test_SOS部分
            f'CH_T{T}s_T{total_time}',
            f'fourier_ch_{fourier_channels}Layer{num_layers}_channel{features}_ks_{kernel_size}',
            f'bs{batch_size}_ensemble/'
        )
        os.makedirs(outf, exist_ok=True)
        print(f"Ensemble results will be saved to: {outf}")
        
        # 加载数据集（保持不变）
        dataset = CH_GRF_Dataset(data_path, transform=transforms.ToTensor())
        dataset = train_val_dataset(dataset, seed)

        loader_test = DataLoader(dataset['test'], batch_size=1)

        criterion = nn.MSELoss(reduction='mean').to(device)

        all_times = generate_test_times(T, total_time)
        all_steps = [x * 10-1 for x in all_times]

        # 生成推断时间点
        inference_times = generate_inference_times(T, total_time)
        inference_steps = [x * 10-1 for x in inference_times]
        pdb.set_trace()

        # 收集模型路径
        model_paths = []
        for i in range(1, 6):  # model1 to model5
            model_path = os.path.join(
                model_base_path, 
                f'model{i}', 
                f'CH_T{T}s_T{total_time}', 
                f'fourier_ch_{fourier_channels}Layer{num_layers}_channel{features}_ks_{kernel_size}', 
                '_a_cir_999_pre_epoch_10', 
                f'bs{batch_size}_LR_lr0.001_epoch50', 
                'model_best_weight.pth'
            )
            
            if os.path.exists(model_path):
                model_paths.append(model_path)
            else:
                print(f"Model path does not exist: {model_path}")

        print(f"Found {len(model_paths)} models for ensemble prediction")
   
        # 创建模型实例列表，每个模型使用自己的initial_parameters
        nets = []
        for i in range(1, 6):  # model1 to model5
            # 加载每个模型对应的initial_parameters
            initial_parameters = load_initial_parameters(
                model_base_path, i, T, total_time, num_layers, features, kernel_size, batch_size, fourier_channels
            )
            
            # 创建模型实例（传入NpointsMesh）
            net = Wise_SpectralTransform(
                num_layers=num_layers, 
                channel_feat=features, 
                kernel_size=kernel_size,
                fourier_channels=fourier_channels,
                phase=4, 
                NpointsMesh=NpointsMesh,
                initial_parameters=initial_parameters
            ).to(device)
            nets.append(net)
        
        # 测试阶段：使用所有模型对0时刻进行预测，并对结果取平均
        print("Performing ensemble test prediction...")
        test_mse_all = []
        avg_test_predictions = []

        with torch.no_grad():
            for i, batch in enumerate(loader_test):
                # 使用0时刻的真实值作为输入
                data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(device)
                target = batch[:, idx_initial, :, :].unsqueeze(1).to(torch.float32).to(device)

                # 使用统一的集成预测函数
                out_test = predict_with_models_ensemble(model_paths, nets, data, device)
                avg_test_predictions.append(out_test)

                # 计算归一化MSE，使用与参考代码相同的计算方法
                mse = criterion(out_test, target) / (criterion(target, torch.zeros_like(target)) + 1e-8)
                test_mse_all.append(mse.item())
                print(f"Test batch {i}: MSE = {mse.item()}")

        avg_test_mse = np.mean(test_mse_all)
        print(f"Average test MSE: {avg_test_mse}")
        plot_ensemble_test_results(model_paths, nets, loader_test, idx_initial, outf, num_layers)

        # -------------------------- 核心修改：仅处理能量数据，保留MSE原始逻辑 --------------------------
        # 1. 计算T0（初始时刻）能量
        t0_energy = compute_t0_energy(loader_test, device)
        print(f"Initial (T0) energy: {t0_energy}")

        # 2. 计算测试能量（T1）和推理能量（T2及以后）
        test_energy_pred, test_energy_target = compute_test_energy_ensemble(model_paths, loader_test, idx_initial, device, nets)
        print(f"Test energy (T1) - Prediction: {test_energy_pred}, Target: {test_energy_target}")

        print("Performing ensemble inference prediction...")
        inference_mse_all = []
        inference_energy_pred_all = []
        inference_energy_target_all = []
        prev_predictions = avg_test_predictions[:]  # 复制测试预测结果作为推理的初始输入
        all_out_test = []  # 收集所有推理步骤的输出结果
        
        for step_idx in inference_steps:
            step_mse = []
            # 为当前step创建新的预测列表
            current_predictions = []
            
            with torch.no_grad():
                for i, batch in enumerate(loader_test):
                    # 使用集成模型进行预测
                    out_infer = predict_with_models_ensemble(model_paths, nets, prev_predictions[i], device)
                    current_predictions.append(out_infer)
                    
                    target = batch[:, step_idx, :, :].unsqueeze(1).to(torch.float32).to(device)
                    mse = nn.functional.mse_loss(out_infer, target) / (nn.functional.mse_loss(target, torch.zeros_like(target)) + 1e-8)
                    step_mse.append(mse.item())
                    
                    print(f"Inference step {step_idx}, batch {i}: MSE = {mse.item()}")
            
            avg_step_mse = np.mean(step_mse)
            inference_mse_all.append(avg_step_mse)
            print(f"Average MSE for inference step {step_idx}: {avg_step_mse}")
            
            # 保存当前步的结果
            all_out_test.append(np.array([tensor.cpu().numpy() for tensor in current_predictions]))
            
            # 计算推理能量
            infer_energy_pred, infer_energy_target = compute_infer_energy_ensemble(
                model_paths, loader_test, step_idx, device, prev_predictions, nets)
            inference_energy_pred_all.append(infer_energy_pred)
            inference_energy_target_all.append(infer_energy_target)
            print(f"Inference energy for step {step_idx} - Prediction: {infer_energy_pred}, Target: {infer_energy_target}")

            # 绘制推理结果
            plot_ensemble_inference_results(model_paths, nets, loader_test, step_idx, outf, num_layers, prev_predictions)

            # 更新预测值，用于下一步推理
            prev_predictions = current_predictions[:]

        # -------------------------- 整合数据：仅能量添加T0，MSE保持不变 --------------------------
        # MSE列表：保持原始逻辑（无T0）
        all_mse = [avg_test_mse] + inference_mse_all
        
        # 能量列表：添加T0作为第一个元素
        all_energy_pred = [t0_energy, test_energy_pred] + inference_energy_pred_all
        all_energy_target = [t0_energy, test_energy_target] + inference_energy_target_all  # T0目标=自身能量
        
        # 能量快照：插入T0原始数据到最前方，再插入T1测试结果
        t0_data_snapshots = []
        with torch.no_grad():
            for batch in loader_test:
                t0_data = batch[:, 0, :, :].unsqueeze(1).cpu().numpy()
                t0_data_snapshots.append(t0_data)
        t0_data_snapshots = np.array(t0_data_snapshots)
        all_out_test.insert(0, t0_data_snapshots)  # T0快照
        all_out_test.insert(1, np.array([tensor.cpu().numpy() for tensor in avg_test_predictions]))  # T1快照
        
        # -------------------------- 保存结果：仅能量CSV添加T0表头 --------------------------
        # MSE CSV：保持原始表头（无T0）
        csv_path = os.path.join(outf, 'Total_mse_mean.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["T1"] + [f"T{i + 2}" for i in range(len(inference_steps))])
            writer.writerow(all_mse)

        # 在main函数中保存能量CSV的部分
        csv_path = os.path.join(outf, 'FFT_Total_energy_mean.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # CSV表头同步为0, T, 2T...
            csv_headers = ["0"]
            for i in range(1, len(all_energy_pred)):
                if i == 1:
                    csv_headers.append("T")
                else:
                    csv_headers.append(f"{i}T")
            writer.writerow(csv_headers)
            writer.writerow(all_energy_pred)
            writer.writerow(all_energy_target)
        
        # 绘制MSE图表
        plot_mse_results(all_mse, T, outf)
        
        # 绘制能量图表（包含预测值和真实值）
        plot_energy_results(all_energy_pred, all_energy_target, T, outf, all_out_test)
        
        print("Ensemble prediction completed successfully")
        print(f"MSE results saved to {os.path.join(outf, 'Total_mse_mean.csv')}")
        print(f"Energy results saved to {os.path.join(outf, 'FFT_Total_energy_mean.csv')}")
        print(f"MSE plot saved to {os.path.join(outf, 'mse_loss_over_time.png')}")
        print(f"Energy plot saved to {os.path.join(outf, 'energy_over_time.png')}")


if __name__ == '__main__':
    # 使用与原main函数相同的参数
    num_layers = 3
    kernel_size = 7
    features = 7
    batch_size = 4
    fourier_channels = 1
    NpointsMesh = 16
    
    # 调用main函数
    main(
        num_layers=num_layers, 
        kernel_size=kernel_size, 
        features=features, 
        batch_size=batch_size, 
        fourier_channels=fourier_channels,
        model_source='L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3/Test_SOS_SR',
        NpointsMesh=NpointsMesh
    )