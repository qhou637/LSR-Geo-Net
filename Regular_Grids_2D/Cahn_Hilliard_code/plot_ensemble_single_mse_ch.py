

#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


root_dir = "/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Gauss_Molification/Deconv5/Gauss_MolifyMesh16/Fourier_Channel/L16pi_High_Tik_k2_400h2_OneTanhNew_k8_m05_Grad0.3"

models_base = os.path.join(root_dir, "Test_SOS/Autostep1")


ensemble_path = os.path.join(
    root_dir,
    "MoE_Res_1/Test_SOS/CH_T7s_T49/",
    "fourier_ch_1Layer3_channel7_ks_7/bs4_ensemble/Total_mse_mean.csv"
)


model_rel_pattern = os.path.join(
    models_base,
    "model{idx}/CH_T7s_T49/fourier_ch_1Layer3_channel7_ks_7/"
    "_a_cir_999_pre_epoch_10/bs4_LR_lr0.001_epoch50/Total_mse_mean.csv"
)


out_dir = os.path.join(root_dir, "MoE_Res_1/Test_SOS")
os.makedirs(out_dir, exist_ok=True)



def load_mse_csv(csv_path):
 
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    vals = df.iloc[0].values.astype(float)
    return vals



model_mse_list = []
for i in range(1, 6):  # model1 to model5
    path = model_rel_pattern.format(idx=i)
    if os.path.exists(path):
        try:
            vals = load_mse_csv(path)
            model_mse_list.append(vals)
            print(f"成功加载 model{i}: 数据长度={len(vals)}")
        except Exception as e:
            print(f"读取model{i}失败: {e}")
    else:
        print(f"警告: model{i}文件不存在 - {path}")

if len(model_mse_list) == 0:
    raise RuntimeError("未找到任何模型的MSE文件，请检查路径是否正确。")


ensemble_mse = None
if os.path.exists(ensemble_path):
    ensemble_mse = load_mse_csv(ensemble_path)
    print(f"成功加载Ensemble: 数据长度={len(ensemble_mse)}")
else:
    print(f"警告: Ensemble文件不存在 - {ensemble_path}")

    moe_dir = os.path.dirname(ensemble_path)
    if os.path.exists(moe_dir):
        print(f"MoE_Res_1目录下的文件: {os.listdir(moe_dir)}")
    else:
        print(f"MoE_Res_1目录不存在: {moe_dir}")


lengths = [len(v) for v in model_mse_list]
if ensemble_mse is not None:
    lengths.append(len(ensemble_mse))
min_len = min(lengths)
print(f"\n对齐数据长度为: {min_len} (原始长度: {lengths})")


model_mse_list = [v[:min_len] for v in model_mse_list]
if ensemble_mse is not None:
    ensemble_mse = ensemble_mse[:min_len]


steps = np.arange(min_len)

T_labels = [f"{i}T" for i in range(1, min_len + 1)]

if len(T_labels) > 0: 
    T_labels[0] = "T"
model_mse_array = np.stack(model_mse_list, axis=0)
steps = np.arange(min_len)
model_mse_array = np.stack(model_mse_list, axis=0)


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "font.family": "Nimbus Roman" 
})

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
linestyles = ["--", "--", "--", "--", "--"] 


plt.figure(figsize=(7.5, 5.5))
for i in range(model_mse_array.shape[0]):
    plt.plot(
        steps,
        model_mse_array[i],
        label=f"Model{i+1}",
        linewidth=2.0,
        alpha=0.9,
        color=colors[i],
        linestyle=linestyles[i],
    )
# Ensemble
if ensemble_mse is not None:
    plt.plot(
        steps,
        ensemble_mse,
        label="Ensemble",
        color="black",
        linewidth=3.0,
        linestyle="-",
        marker = 's'
    )


plt.xticks(steps, T_labels)
plt.xlabel("Time")
plt.ylabel("RMSE")
plt.grid(True, linestyle="--", alpha=0.4)


plt.xticks(steps, T_labels)
plt.xlabel("Time")
plt.ylabel("RMSE")
plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend(loc="lower left", frameon=False)
plt.legend(
    loc="lower left",        
    bbox_to_anchor=(0, 0.05),
    frameon=False
)
plt.tight_layout()
plt.tight_layout()


fnA = os.path.join(out_dir, "mse_all_models_and_ensemble.png")
plt.savefig(fnA, dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {fnA}")


last_vals = model_mse_array[:, -1]
best_idx = int(np.argmin(last_vals))
worst_idx = int(np.argmax(last_vals))

print(f"\n最佳模型: Model{best_idx+1} (最终MSE={last_vals[best_idx]:.4e})")
print(f"最差模型: Model{worst_idx+1} (最终MSE={last_vals[worst_idx]:.4e})")

plt.figure(figsize=(6.5, 5.5))

plt.plot(
    steps, 
    model_mse_array[best_idx], 
    'o-', 
    label=f"Best (Model{best_idx+1})", 
    linewidth=2.3,
    color=colors[best_idx]
)

plt.plot(
    steps, 
    model_mse_array[worst_idx], 
    's--', 
    label=f"Worst (Model{worst_idx+1})", 
    linewidth=2.3,
    color=colors[worst_idx]
)

if ensemble_mse is not None:
    plt.plot(
        steps,
        ensemble_mse,
        'd-.',
        label="Ensemble",
        linewidth=2.8,
        color='black',
        markersize=6
    )


plt.xticks(steps, T_labels)
plt.xlabel("Time")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper left", frameon=False)
plt.tight_layout()


fnB = os.path.join(out_dir, "mse_ensemble_best_worst.png")
plt.savefig(fnB, dpi=300, bbox_inches='tight')
plt.close()  
print(f"已保存: {fnB}")

print(f"\n所有图表已保存到: {out_dir}")
print("所有图表生成完成！")
