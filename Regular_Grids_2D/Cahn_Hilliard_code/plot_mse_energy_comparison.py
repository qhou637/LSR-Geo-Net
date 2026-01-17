import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_existing_mean(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MSE均值文件不存在: {file_path}")
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        mse_mean = [float(x) for x in next(reader)]
    return headers, mse_mean

def calculate_std_from_multiruns(run_paths):

    all_mse_data = []
    valid_paths = []
    
    for path in run_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader) 
                    mse_data = [float(x) for x in next(reader)]
                    all_mse_data.append(mse_data)
                    valid_paths.append(path)
            except Exception as e:
                print(f"警告: 读取文件失败 {path}: {e}")
        else:
            print(f"警告: 文件不存在 {path}")
    
    if len(all_mse_data) == 0:
        raise RuntimeError("没有有效的MSE文件用于计算标准差")
    

    mse_std = np.std(np.array(all_mse_data), axis=0).tolist()
    return mse_std

def plot_mse_comparison(forward_step=1):

    plt.rcParams['font.family'] = 'Nimbus Roman'

    root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"
    

    sos_base_prefix = os.path.join(
        root_dir, 
        f"Test_SOS/Autostep{forward_step}/model{{}}/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/",
        "_a_cir_999_pre_epoch_10/bs4_LR_lr0.001_epoch50/Total_mse_mean.csv"
    )
    
    sos_sr_base_prefix = os.path.join(
        root_dir, 
        f"Test_SOS_SR/Autostep{forward_step}/model{{}}/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/",
        "_a_cir_999_pre_epoch_10/bs4_LR_lr0.001_epoch50/Total_mse_mean.csv"
    )
    

    sos_5runs_paths = [sos_base_prefix.format(i) for i in range(1, 6)]
    sos_sr_5runs_paths = [sos_sr_base_prefix.format(i) for i in range(1, 6)]
    

    sos_mse_mean_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/Total_mse_mean.csv"
    )
    
    sos_sr_mse_mean_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS_SR/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/Total_mse_mean.csv"
    )
    

    headers_sos, mse_sos_mean = read_existing_mean(sos_mse_mean_path)
    mse_sos_std = calculate_std_from_multiruns(sos_5runs_paths)
    

    headers_sos_sr, mse_sos_sr_mean = read_existing_mean(sos_sr_mse_mean_path)
    mse_sos_sr_std = calculate_std_from_multiruns(sos_sr_5runs_paths)

    x_labels = []
    for label in headers_sos:
        if '(test)' in label:
            cleaned = label.replace('(test)', '').strip()
            x_labels.append(cleaned)
        elif label.startswith('T'):
            num = label.replace('T', '').strip()
            if num == '1' or num == '':
                x_labels.append('T')
            else:
                x_labels.append(f'{num}T')
        else:
            x_labels.append(label)
    

    min_len = min(len(mse_sos_mean), len(mse_sos_sr_mean), len(x_labels))
    mse_sos_mean = mse_sos_mean[:min_len]
    mse_sos_sr_mean = mse_sos_sr_mean[:min_len]
    mse_sos_std = mse_sos_std[:min_len]
    mse_sos_sr_std = mse_sos_sr_std[:min_len]
    x_labels = x_labels[:min_len]
    
 
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
 
    ax.plot(x_labels, mse_sos_mean, marker='o', linestyle='-', color='blue', 
            linewidth=2.5, markersize=7, label='LSR-Net')
    ax.plot(x_labels, mse_sos_sr_mean, marker='s', linestyle='-', color='gray', 
            linewidth=2.5, markersize=7, label='SR-Net')

    ax.fill_between(x_labels, 
                    [m - 2*s for m, s in zip(mse_sos_mean, mse_sos_std)],
                    [m + 2*s for m, s in zip(mse_sos_mean, mse_sos_std)],
                    color='blue', alpha=0.2, label='LSR-Net ± 2σ')
    ax.fill_between(x_labels, 
                    [m - 2*s for m, s in zip(mse_sos_sr_mean, mse_sos_sr_std)],
                    [m + 2*s for m, s in zip(mse_sos_sr_mean, mse_sos_sr_std)],
                    color='gray', alpha=0.2, label='SR-Net ± 2σ')
    

    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('RMSE', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size - 4)
    
    # 调整x轴范围
    max_x_idx = len(x_labels) - 1
    ax.set_xlim(left=0, right=max_x_idx + 0.2)
    
    # 保存图片
    save_dir = os.path.join(root_dir, f"MoE_Res_{forward_step}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mse_comparison.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MSE对比图(LSR/SR)已保存: {save_path}")

def plot_energy_comparison(forward_step=1):

    plt.rcParams['font.family'] = 'Nimbus Roman'
    

    root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"

    sos_energy_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/FFT_Total_energy_mean.csv"
    )
    
    sos_sr_energy_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS_SR/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/FFT_Total_energy_mean.csv"
    )
    

    for path in [sos_energy_path, sos_sr_energy_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"能量文件不存在：{path}")


    # LSR-Net (Test_SOS)
    with open(sos_energy_path, 'r') as f:
        reader = csv.reader(f)
        headers_sos = next(reader)
        energy_pred_sos = [float(x) for x in next(reader)]
        energy_target_sos = [float(x) for x in next(reader)]
    
    # SR-Net (Test_SOS_SR)
    with open(sos_sr_energy_path, 'r') as f:
        reader = csv.reader(f)
        headers_sos_sr = next(reader)
        energy_pred_sos_sr = [float(x) for x in next(reader)]
        energy_target_sos_sr = [float(x) for x in next(reader)]

    if headers_sos[0] != '0':
        headers_sos[0] = '0'
    
    max_len = max(
        len(energy_pred_sos), 
        len(energy_pred_sos_sr), 
        len(energy_target_sos),
        len(headers_sos)
    )
    
    def pad_data(data):
        if len(data) < max_len:
            return data + [np.nan] * (max_len - len(data))
        return data[:max_len]
    
    energy_pred_sos = pad_data(energy_pred_sos)
    energy_pred_sos_sr = pad_data(energy_pred_sos_sr)
    energy_target_sos = pad_data(energy_target_sos)
    headers_sos = headers_sos[:max_len]
    x_indices = list(range(len(headers_sos)))
    
   
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    ax.plot(x_indices, energy_pred_sos, marker='o', linestyle='-', color='blue', 
            linewidth=2.5, markersize=7, label='LSR-Net')
    ax.plot(x_indices, energy_pred_sos_sr, marker='s', linestyle='-', color='green', 
            linewidth=2.5, markersize=7, label='SR-Net')
    ax.plot(x_indices, energy_target_sos, marker='o', linestyle='--', color='red', 
            linewidth=2.5, markersize=7, label='Benchmark')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(headers_sos)
    

    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Free Energy', fontsize=font_size)
    ax.set_title("Cahn--Hilliard Energy Comparison", fontsize=font_size + 2)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size - 4)
    
    max_x_idx = x_indices[-1] if x_indices else 0
    ax.set_xlim(left=x_indices[0] if x_indices else 0, right=max_x_idx + 0.2)
    

    save_dir = os.path.join(root_dir, f"MoE_Res_{forward_step}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'energy_comparison.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"能量对比图(LSR/SR)已保存: {save_path}")

def plot_mse_comparison_with_fno(forward_step=1):
    """新增：绘制LSR/SR/FNO三者的MSE对比图"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    

    root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"
    fno_mse_path = "/home/qianhou/FFT_PDE/FNO/Cahn_Hillard/GRF_ICs/Infer_CH_FNO_case/CH_GRF_FNO/Auto_step2/T7_k8_mean03_grad0.3/hidd10_modes_6/ensemble_5models/ensemble_total_mse_mean.csv"
    
  
    sos_mse_mean_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/Total_mse_mean.csv"
    )
    sos_sr_mse_mean_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS_SR/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/Total_mse_mean.csv"
    )
    
    headers_sos, mse_sos_mean = read_existing_mean(sos_mse_mean_path)
    headers_sos_sr, mse_sos_sr_mean = read_existing_mean(sos_sr_mse_mean_path)
    headers_fno, mse_fno_mean = read_existing_mean(fno_mse_path)
    

    x_labels = []
    for label in headers_sos:
        if '(test)' in label:
            cleaned = label.replace('(test)', '').strip()
            x_labels.append(cleaned)
        elif label.startswith('T'):
            num = label.replace('T', '').strip()
            if num == '1' or num == '':
                x_labels.append('T')
            else:
                x_labels.append(f'{num}T')
        else:
            x_labels.append(label)
    

    min_len = min(len(mse_sos_mean), len(mse_sos_sr_mean), len(mse_fno_mean), len(x_labels))
    mse_sos_mean = mse_sos_mean[:min_len]
    mse_sos_sr_mean = mse_sos_sr_mean[:min_len]
    mse_fno_mean = mse_fno_mean[:min_len]
    x_labels = x_labels[:min_len]
    
 
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(x_labels, mse_sos_mean, marker='o', linestyle='-', color='blue', 
            linewidth=2.5, markersize=7, label='LSR-Net')
    ax.plot(x_labels, mse_sos_sr_mean, marker='s', linestyle='-', color='green', 
            linewidth=2.5, markersize=7, label='SR-Net')
    ax.plot(x_labels, mse_fno_mean, marker='^', linestyle='-', color='orange', 
            linewidth=2.5, markersize=7, label='FNO')
    

    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('RMSE', fontsize=font_size)

    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size - 4)
    
    max_x_idx = len(x_labels) - 1
    ax.set_xlim(left=0, right=max_x_idx + 0.2)
    

    save_dir = os.path.join(root_dir, f"MoE_Res_{forward_step}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mse_comparison_with_fno.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MSE对比图(LSR/SR/FNO)已保存: {save_path}")

def plot_energy_comparison_with_fno(forward_step=1):
    """LSR/SR/FNO Energy"""
    plt.rcParams['font.family'] = 'Nimbus Roman'
    
  
    root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"
    fno_energy_path = "/home/qianhou/FFT_PDE/FNO/Cahn_Hillard/GRF_ICs/Infer_CH_FNO_case/CH_GRF_FNO/Auto_step2/T7_k8_mean03_grad0.3/hidd10_modes_6/ensemble_5models/ensemble_FFT_total_energy_mean.csv"
    

    sos_energy_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/FFT_Total_energy_mean.csv"
    )
    sos_sr_energy_path = os.path.join(
        root_dir,
        f"MoE_Res_{forward_step}/Test_SOS_SR/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/FFT_Total_energy_mean.csv"
    )
    

    if not os.path.exists(fno_energy_path):
        raise FileNotFoundError(f"FNO能量文件不存在：{fno_energy_path}")
    
    with open(sos_energy_path, 'r') as f:
        reader = csv.reader(f)
        headers_sos = next(reader)
        energy_pred_sos = [float(x) for x in next(reader)]
        energy_target_sos = [float(x) for x in next(reader)]
    
    with open(sos_sr_energy_path, 'r') as f:
        reader = csv.reader(f)
        headers_sos_sr = next(reader)
        energy_pred_sos_sr = [float(x) for x in next(reader)]
    
    with open(fno_energy_path, 'r') as f:
        reader = csv.reader(f)
        headers_fno = next(reader)
        energy_pred_fno = [float(x) for x in next(reader)]
        energy_target_fno = [float(x) for x in next(reader)]
    

    if headers_sos[0] != '0':
        headers_sos[0] = '0'
    
    max_len = max(
        len(energy_pred_sos), 
        len(energy_pred_sos_sr), 
        len(energy_pred_fno),
        len(energy_target_sos),
        len(headers_sos)
    )
    
    def pad_data(data):
        if len(data) < max_len:
            return data + [np.nan] * (max_len - len(data))
        return data[:max_len]
    
    energy_pred_sos = pad_data(energy_pred_sos)
    energy_pred_sos_sr = pad_data(energy_pred_sos_sr)
    energy_pred_fno = pad_data(energy_pred_fno)
    energy_target_sos = pad_data(energy_target_sos)
    headers_sos = headers_sos[:max_len]
    x_indices = list(range(len(headers_sos)))
    
  
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    

    ax.plot(x_indices, energy_pred_sos, marker='o', linestyle='-', color='blue', 
            linewidth=2.5, markersize=7, label='LSR-Net')
    ax.plot(x_indices, energy_pred_sos_sr, marker='s', linestyle='-', color='green', 
            linewidth=2.5, markersize=7, label='SR-Net')
    ax.plot(x_indices, energy_pred_fno, marker='^', linestyle='-', color='orange', 
            linewidth=2.5, markersize=7, label='FNO')
    ax.plot(x_indices, energy_target_sos, marker='o', linestyle='--', color='red', 
            linewidth=2.5, markersize=7, label='Benchmark')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(headers_sos)
    
  
    font_size = 20
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Free Energy', fontsize=font_size)
    ax.set_title("Cahn--Hilliard Energy Comparison (LSR/SR/FNO)", fontsize=font_size + 2)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(fontsize=font_size - 4)
    
    max_x_idx = x_indices[-1] if x_indices else 0
    ax.set_xlim(left=x_indices[0] if x_indices else 0, right=max_x_idx + 0.2)
    

    save_dir = os.path.join(root_dir, f"MoE_Res_{forward_step}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'energy_comparison_with_fno.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"能量对比图(LSR/SR/FNO)已保存: {save_path}")

def main():
    forward_step = 1
    

    root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"
    save_dir = os.path.join(root_dir, f"MoE_Res_{forward_step}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        plot_mse_comparison(forward_step=forward_step)
        plot_energy_comparison(forward_step=forward_step)
        

        plot_mse_comparison_with_fno(forward_step=forward_step)
        plot_energy_comparison_with_fno(forward_step=forward_step)
        
        print("\n所有对比图生成完成！")
        print(f"输出目录: {save_dir}")
        print("\n生成的文件列表：")
        print(f"1. {save_dir}/mse_comparison.png (LSR/SR对比)")
        print(f"2. {save_dir}/energy_comparison.png (LSR/SR对比)")
        print(f"3. {save_dir}/mse_comparison_with_fno.png (LSR/SR/FNO对比)")
        print(f"4. {save_dir}/energy_comparison_with_fno.png (LSR/SR/FNO对比)")
        
    except Exception as e:
        print(f"错误: {e}")
        raise

if __name__ == "__main__":
    main()


