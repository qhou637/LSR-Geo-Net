


# import os
# import sys
# import math
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib.animation as animation
# from matplotlib.colors import Normalize
# from math import ceil
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.insert(0, '/home/qianhou/torch-harmonics')

# # 禁用cartopy避免冲突
# # import cartopy.crs as ccrs

# # ===================== 核心修复：独立的3D球面可视化函数（不依赖solver） =====================
# def plot_3d_sphere(data, fig, cmap='twilight_shifted', vmin=None, vmax=None, azim=0, elev=30):
#     """
#     独立实现3D球面可视化，完全不依赖solver的plot_griddata，避免GPU/投影错误
#     """
#     # 确保数据是NumPy数组
#     if isinstance(data, torch.Tensor):
#         data_np = data.detach().cpu().numpy()
#     else:
#         data_np = np.array(data)
    
#     # 自动计算vmin/vmax
#     if vmin is None:
#         vmin = data_np.min()
#     if vmax is None:
#         vmax = data_np.max()
    
#     # 创建3D轴（兼容所有matplotlib版本）
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 生成经纬度网格
#     nlat, nlon = data_np.shape
#     lats = np.linspace(-90, 90, nlat)
#     lons = np.linspace(0, 360, nlon)
#     Lons, Lats = np.meshgrid(lons, lats)
    
#     # 转换为3D笛卡尔坐标
#     R = 1.0
#     x = R * np.cos(np.radians(Lats)) * np.cos(np.radians(Lons))
#     y = R * np.cos(np.radians(Lats)) * np.sin(np.radians(Lons))
#     z = R * np.sin(np.radians(Lats))
    
#     # 归一化数据用于颜色映射
#     norm_data = (data_np - vmin) / (vmax - vmin)
#     norm_data = np.clip(norm_data, 0, 1)  # 防止超出范围
    
#     # 获取颜色映射
#     cmap_obj = plt.get_cmap(cmap)
#     colors = cmap_obj(norm_data)
    
#     # 绘制3D球面
#     surf = ax.plot_surface(
#         x, y, z,
#         rstride=1, cstride=1,
#         facecolors=colors,
#         linewidth=0,
#         antialiased=False,
#         shade=False
#     )
    
#     # 设置3D视角
#     ax.view_init(elev=elev, azim=azim)
    
#     # 设置坐标轴
#     ax.set_xlim([-R, R])
#     ax.set_ylim([-R, R])
#     ax.set_zlim([-R, R])
#     ax.set_aspect('equal')
#     ax.axis('off')  # 关闭坐标轴
    
#     # 添加颜色条
#     mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=vmin, vmax=vmax))
#     mappable.set_array(data_np)
#     cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
#     cbar.ax.tick_params(labelsize=12)
#     formatter = mticker.FormatStrFormatter('%.2f')
#     cbar.ax.yaxis.set_major_formatter(formatter)
    
#     # 移除标题
#     ax.set_title('')
#     fig.suptitle('')
    
#     return ax

# # ===================== 多角度可视化核心函数 =====================
# def plot_3d_sphere_rotating(data, cmap='twilight_shifted', vmin=None, vmax=None, 
#                            angles=None, save_dir=None, prefix="sphere", figsize=(8,7), dpi=300):
#     """
#     生成球面多角度静态图或旋转动画
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 1. 生成多角度静态图片
#     if angles is not None:
#         for idx, (azim, elev) in enumerate(angles):
#             fig = plt.figure(figsize=figsize)
#             # 使用独立的3D球面绘制函数
#             plot_3d_sphere(data, fig, cmap=cmap, vmin=vmin, vmax=vmax, azim=azim, elev=elev)
            
#             # 保存图片
#             save_path = os.path.join(save_dir, f"{prefix}_azim{azim}_elev{elev}.png")
#             fig.savefig(
#                 save_path, 
#                 dpi=dpi, 
#                 bbox_inches='tight', 
#                 pad_inches=0.1, 
#                 facecolor='white'
#             )
#             plt.close(fig)
#         return

#     # 2. 生成旋转动画（可选）
#     fig = plt.figure(figsize=figsize)
#     plot_3d_sphere(data, fig, cmap=cmap, vmin=vmin, vmax=vmax)
    
#     # 定义动画更新函数
#     def update(frame):
#         ax = fig.gca(projection='3d')
#         ax.view_init(elev=30, azim=frame)
#         return [ax]
    
#     # 创建动画
#     ani = animation.FuncAnimation(
#         fig, 
#         update, 
#         frames=np.arange(0, 360, 5),
#         interval=50, 
#         blit=False,  # 关闭blit避免兼容问题
#         repeat=True
#     )
    
#     # 保存动画
#     gif_path = os.path.join(save_dir, f"{prefix}_rotating.gif")
#     ani.save(gif_path, writer='pillow', fps=10, dpi=dpi)
#     plt.close(fig)
#     print(f"旋转动画已保存至: {gif_path}")

# # ===================== 改造后的plot_sample_comparison =====================
# def plot_sample_comparison(result_list, sample_id=0, save_dir="test_plots_raotation", 
#                            cmap='twilight_shifted', t=None, rotate_mode="static"):
#     """
#     生成对比可视化（无solver依赖）
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     var_names = ['u', 'v']
#     data, pred, target = result_list
    
#     # 确保所有数据移到CPU
#     data = data.squeeze(0).cpu()
#     pred = pred.squeeze(0).cpu()
#     target = target.squeeze(0).cpu()

#     # 定义要展示的视角
#     view_angles = [
#         (0, 30),    # 正面
#         (90, 30),   # 右侧
#         (180, 30),  # 背面
#         (270, 30),  # 左侧
#         (0, 90)     # 顶部
#     ]

#     for i, var in enumerate(var_names):
#         # 准备数据
#         d_in = data[i]
#         d_gt = target[i]
#         d_pred = pred[i]
#         d_err = torch.abs(d_pred - d_gt)

#         # 计算vmin/vmax
#         vmin_input, vmax_input = d_in.min().item(), d_in.max().item()
#         vmin_target, vmax_target = d_gt.min().item(), d_gt.max().item()
#         vmin_pred, vmax_pred = d_pred.min().item(), d_pred.max().item()

#         # 构造文件名前缀
#         t_str = f"_t{t:02d}" if t is not None else ""
#         base_prefix = f"{sample_id}{t_str}_{var}"

#         # 生成保存目录
#         var_save_dir = os.path.join(save_dir, var)
#         os.makedirs(var_save_dir, exist_ok=True)

#         # 1. Input 数据可视化
#         if rotate_mode in ["static", "both"]:
#             plot_3d_sphere_rotating(
#                 d_in, cmap=cmap, vmin=vmin_input, vmax=vmax_input,
#                 angles=view_angles, save_dir=var_save_dir, prefix=f"{base_prefix}_input"
#             )
#         if rotate_mode in ["animation", "both"]:
#             plot_3d_sphere_rotating(
#                 d_in, cmap=cmap, vmin=vmin_input, vmax=vmax_input,
#                 angles=None, save_dir=var_save_dir, prefix=f"{base_prefix}_input"
#             )

#         # 2. Ground Truth 数据可视化
#         if rotate_mode in ["static", "both"]:
#             plot_3d_sphere_rotating(
#                 d_gt, cmap=cmap, vmin=vmin_target, vmax=vmax_target,
#                 angles=view_angles, save_dir=var_save_dir, prefix=f"{base_prefix}_target"
#             )
#         if rotate_mode in ["animation", "both"]:
#             plot_3d_sphere_rotating(
#                 d_gt, cmap=cmap, vmin=vmin_target, vmax=vmax_target,
#                 angles=None, save_dir=var_save_dir, prefix=f"{base_prefix}_target"
#             )

#         # 3. Prediction 数据可视化
#         if rotate_mode in ["static", "both"]:
#             plot_3d_sphere_rotating(
#                 d_pred, cmap=cmap, vmin=vmin_pred, vmax=vmax_pred,
#                 angles=view_angles, save_dir=var_save_dir, prefix=f"{base_prefix}_pred"
#             )
#         if rotate_mode in ["animation", "both"]:
#             plot_3d_sphere_rotating(
#                 d_pred, cmap=cmap, vmin=vmin_pred, vmax=vmax_pred,
#                 angles=None, save_dir=var_save_dir, prefix=f"{base_prefix}_pred"
#             )

#         # 4. Error 数据可视化
#         if rotate_mode in ["static", "both"]:
#             plot_3d_sphere_rotating(
#                 d_err, cmap='Reds', vmin=None, vmax=None,
#                 angles=view_angles, save_dir=var_save_dir, prefix=f"{base_prefix}_error"
#             )
#         if rotate_mode in ["animation", "both"]:
#             plot_3d_sphere_rotating(
#                 d_err, cmap='Reds', vmin=None, vmax=None,
#                 angles=None, save_dir=var_save_dir, prefix=f"{base_prefix}_error"
#             )

# # ===================== 修复后的plot_infer_results（整合自定义时间步） =====================
# def plot_infer_results(save_path, inference_steps, interval=10, cmap='twilight_shifted'):
#     print('Plot inference results with rotation...')
#     # 加载数据
#     data = torch.load(os.path.join(save_path, "infer_results.pt"), weights_only=False)
#     save_dir = os.path.join(save_path, "infer_plots_rotating")
#     os.makedirs(save_dir, exist_ok=True)

#     # 所有数据移到CPU
#     inputs = data["inputs"].cpu()          
#     predictions = data["predictions"].cpu()
#     targets = data["targets"].cpu()

#     # 打印数据信息
#     print(f"Inputs shape: {inputs.shape}")
#     print(f"Predictions shape: {predictions.shape}")
#     print(f"Targets shape: {targets.shape}")
#     print(f"Custom inference_steps: {inference_steps}")

#     n_samples, nfuture = predictions.shape[:2]
    
#     # 适配时间步长度
#     if len(inference_steps) > nfuture:
#         inference_steps = inference_steps[:nfuture]
#         print(f"自定义时间步长度超过数据时间维度，截断为: {inference_steps}")
#     elif len(inference_steps) < nfuture:
#         supplement_steps = list(range(len(inference_steps), nfuture))
#         inference_steps += supplement_steps
#         print(f"自定义时间步长度不足，补全后为: {inference_steps}")

#     # 只处理第一个样本（避免生成过多文件，可根据需要调整）
#     sample_id = 0
#     for t in range(nfuture):
#         input_sample = inputs[sample_id:sample_id+1, t]
#         pred_sample = predictions[sample_id:sample_id+1, t]
#         target_sample = targets[sample_id:sample_id+1, t]

#         result_list = [input_sample, pred_sample, target_sample]
#         t_real = 10 * inference_steps[t]  

#         timestep_dir = os.path.join(save_dir, f"t{t_real:02d}")
#         plot_sample_comparison(
#             result_list,
#             sample_id=sample_id,
#             save_dir=timestep_dir,
#             cmap=cmap,
#             t=t_real,
#             rotate_mode="static"  # 仅生成静态图，更快更稳定
#         )

# # ===================== 工具函数 =====================
# def generate_test_times(t, total_time):
#     times = list(range(t, total_time, t))
#     if not times and t <= total_time:  
#         times = [t]
#     elif times and times[-1] + t <= total_time:
#         times.append(times[-1] + t)
#     return times

# # ===================== 主函数 =====================
# if __name__ == "__main__":
#     # 1. 你的推理结果路径
#     infer_path = "/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/TFstep3/Full_Unet_LSR_Deconv5/model1/good_embed7_fourier_ch3_down3/Infer_T130/alpha1e-05_layer4_sf2/ks5_ratio1_epoch50_lr0.001"
    
#     # 2. 自定义时间步逻辑
#     total_time= 400
#     T =  130
#     all_times = generate_test_times(T, total_time)
#     all_steps = [t  // 10 for t in all_times]
#     inference_steps = all_steps[1:]
    
#     # 打印时间步信息
#     print(f"all_times: {all_times}")
#     print(f"all_steps: {all_steps}")
#     print(f"final inference_steps: {inference_steps}")
    
#     # 3. 生成可视化结果（不再需要solver）
#     plot_infer_results(infer_path, inference_steps, interval=1, cmap='twilight_shifted')


import os
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from math import ceil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, '/home/qianhou/torch-harmonics')

# ===================== 核心修复：独立的3D球面可视化函数（子图版） =====================
def plot_3d_sphere_subplot(data, ax, cmap='twilight_shifted', vmin=None, vmax=None, azim=0, elev=30, angle_label=""):
    """
    在子图中绘制3D球面，并标注视角角度
    """
    # 确保数据是NumPy数组
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.array(data)
    
    # 自动计算vmin/vmax
    if vmin is None:
        vmin = data_np.min()
    if vmax is None:
        vmax = data_np.max()
    
    # 生成经纬度网格
    nlat, nlon = data_np.shape
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon)
    Lons, Lats = np.meshgrid(lons, lats)
    
    # 转换为3D笛卡尔坐标
    R = 1.0
    x = R * np.cos(np.radians(Lats)) * np.cos(np.radians(Lons))
    y = R * np.cos(np.radians(Lats)) * np.sin(np.radians(Lons))
    z = R * np.sin(np.radians(Lats))
    
    # 归一化数据用于颜色映射
    norm_data = (data_np - vmin) / (vmax - vmin)
    norm_data = np.clip(norm_data, 0, 1)
    
    # 获取颜色映射
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm_data)
    
    # 绘制3D球面
    surf = ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False
    )
    
    # 设置3D视角
    ax.view_init(elev=elev, azim=azim)
    
    # 设置坐标轴
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加角度标注（左上角，白色字体带黑色描边）
    ax.text(
        -0.9, 0.9, 0.9, angle_label,
        color='white', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
        transform=ax.transAxes, ha='left', va='top'
    )
    
    return ax, vmin, vmax

# ===================== 多视角同图可视化函数 =====================
def plot_multi_angle_sphere(data, cmap='twilight_shifted', vmin=None, vmax=None, 
                           save_dir=None, prefix="sphere", figsize=(20, 12), dpi=300):
    """
    将不同角度的球面图整合到一张画布上，并标注角度
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义要展示的视角及标签
    view_configs = [
        ((0, 30), "Front\n(azim=0°, elev=30°)"),    # 正面
        ((90, 30), "Right\n(azim=90°, elev=30°)"),   # 右侧
        ((180, 30), "Back\n(azim=180°, elev=30°)"),  # 背面
        ((270, 30), "Left\n(azim=270°, elev=30°)"),  # 左侧
        ((0, 90), "Top\n(azim=0°, elev=90°)")        # 顶部
    ]
    
    # 创建画布（2行3列，最后一个位置放颜色条）
    fig = plt.figure(figsize=figsize)
    
    # 绘制各个视角的子图
    axes = []
    for idx, ((azim, elev), label) in enumerate(view_configs):
        # 前5个子图位置：(0,0), (0,1), (0,2), (1,0), (1,1)
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        ax, data_vmin, data_vmax = plot_3d_sphere_subplot(
            data, ax, cmap=cmap, vmin=vmin, vmax=vmax, 
            azim=azim, elev=elev, angle_label=label
        )
        axes.append(ax)
    
    # 在最后一个位置添加统一的颜色条
    cbar_ax = fig.add_subplot(2, 3, 6)
    cmap_obj = plt.get_cmap(cmap)
    mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=data_vmin, vmax=data_vmax))
    mappable.set_array(data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data)
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=14)
    formatter = mticker.FormatStrFormatter('%.2f')
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar_ax.set_title('Value', fontsize=14, pad=10)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存组合图
    save_path = os.path.join(save_dir, f"{prefix}_multi_angle.png")
    fig.savefig(
        save_path, 
        dpi=dpi, 
        bbox_inches='tight', 
        pad_inches=0.2, 
        facecolor='white'
    )
    plt.close(fig)
    print(f"多视角组合图已保存至: {save_path}")

# ===================== 改造后的plot_sample_comparison（多视角同图） =====================
def plot_sample_comparison(result_list, sample_id=0, save_dir="test_plots_rotation", 
                           cmap='twilight_shifted', t=None):
    """
    生成对比可视化（多视角同图展示）
    """
    os.makedirs(save_dir, exist_ok=True)

    var_names = ['u', 'v']
    data, pred, target = result_list
    
    # 确保所有数据移到CPU
    data = data.squeeze(0).cpu()
    pred = pred.squeeze(0).cpu()
    target = target.squeeze(0).cpu()

    for i, var in enumerate(var_names):
        # 准备数据
        d_in = data[i]
        d_gt = target[i]
        d_pred = pred[i]
        d_err = torch.abs(d_pred - d_gt)

        # 计算vmin/vmax
        vmin_input, vmax_input = d_in.min().item(), d_in.max().item()
        vmin_target, vmax_target = d_gt.min().item(), d_gt.max().item()
        vmin_pred, vmax_pred = d_pred.min().item(), d_pred.max().item()

        # 构造文件名前缀
        t_str = f"_t{t:02d}" if t is not None else ""
        base_prefix = f"{sample_id}{t_str}_{var}"

        # 生成保存目录
        var_save_dir = os.path.join(save_dir, var)
        os.makedirs(var_save_dir, exist_ok=True)

        # 1. Input 数据多视角可视化
        plot_multi_angle_sphere(
            d_in, cmap=cmap, vmin=vmin_input, vmax=vmax_input,
            save_dir=var_save_dir, prefix=f"{base_prefix}_input"
        )

        # 2. Ground Truth 数据多视角可视化
        plot_multi_angle_sphere(
            d_gt, cmap=cmap, vmin=vmin_target, vmax=vmax_target,
            save_dir=var_save_dir, prefix=f"{base_prefix}_target"
        )

        # 3. Prediction 数据多视角可视化
        plot_multi_angle_sphere(
            d_pred, cmap=cmap, vmin=vmin_pred, vmax=vmax_pred,
            save_dir=var_save_dir, prefix=f"{base_prefix}_pred"
        )

        # 4. Error 数据多视角可视化
        plot_multi_angle_sphere(
            d_err, cmap='Reds', vmin=None, vmax=None,
            save_dir=var_save_dir, prefix=f"{base_prefix}_error"
        )

# ===================== 修复后的plot_infer_results =====================
def plot_infer_results(save_path, inference_steps, interval=10, cmap='twilight_shifted'):
    print('Plot inference results with multi-angle visualization...')
    # 加载数据
    data = torch.load(os.path.join(save_path, "infer_results.pt"), weights_only=False)
    save_dir = os.path.join(save_path, "infer_plots_multi_angle")
    os.makedirs(save_dir, exist_ok=True)

    # 所有数据移到CPU
    inputs = data["inputs"].cpu()          
    predictions = data["predictions"].cpu()
    targets = data["targets"].cpu()

    # 打印数据信息
    print(f"Inputs shape: {inputs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Custom inference_steps: {inference_steps}")

    n_samples, nfuture = predictions.shape[:2]
    
    # 适配时间步长度
    if len(inference_steps) > nfuture:
        inference_steps = inference_steps[:nfuture]
        print(f"自定义时间步长度超过数据时间维度，截断为: {inference_steps}")
    elif len(inference_steps) < nfuture:
        supplement_steps = list(range(len(inference_steps), nfuture))
        inference_steps += supplement_steps
        print(f"自定义时间步长度不足，补全后为: {inference_steps}")

    # 只处理第一个样本
    sample_id = 0
    for t in range(nfuture):
        input_sample = inputs[sample_id:sample_id+1, t]
        pred_sample = predictions[sample_id:sample_id+1, t]
        target_sample = targets[sample_id:sample_id+1, t]

        result_list = [input_sample, pred_sample, target_sample]
        t_real = 10 * inference_steps[t]  

        timestep_dir = os.path.join(save_dir, f"t{t_real:02d}")
        plot_sample_comparison(
            result_list,
            sample_id=sample_id,
            save_dir=timestep_dir,
            cmap=cmap,
            t=t_real
        )

# ===================== 工具函数 =====================
def generate_test_times(t, total_time):
    times = list(range(t, total_time, t))
    if not times and t <= total_time:  
        times = [t]
    elif times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 推理结果路径
    infer_path = "/home/qianhou/torch-harmonics/notebooks/Turing/l2_loss/TFstep3/Full_Unet_LSR_Deconv5/model1/good_embed7_fourier_ch3_down3/Infer_T130/alpha1e-05_layer4_sf2/ks5_ratio1_epoch50_lr0.001"
    
    # 2. 自定义时间步逻辑
    total_time= 400
    T =  130
    all_times = generate_test_times(T, total_time)
    all_steps = [t  // 10 for t in all_times]
    inference_steps = all_steps[1:]
    
    # 打印时间步信息
    print(f"all_times: {all_times}")
    print(f"all_steps: {all_steps}")
    print(f"final inference_steps: {inference_steps}")
    
    # 3. 生成多视角可视化结果
    plot_infer_results(infer_path, inference_steps, interval=1, cmap='twilight_shifted')