

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import pyvista as pv
# import h5py
# from torch_geometric.data import Data
# from tempfile import NamedTemporaryFile
# import pdb

# # --- 全局配置 (针对远程服务器环境) ---
# os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# if 'DISPLAY' in os.environ:
#     del os.environ['DISPLAY']

# # --- 通用渲染函数 (强制关闭single_plot的colorbar) ---
# def render_single_mesh(mesh, title=None, cmap='turbo', image_size=(600, 500), is_combined=False):
#     """
#     渲染网格：
#     - is_combined=False (single_plot)：强制关闭colorbar（无任何颜色条）
#     - is_combined=True (拼接图)：显示colorbar（无'u'标签+居中）
#     """
#     plotter = pv.Plotter(window_size=image_size, off_screen=True)
    
#     # 核心修复：single_plot强制关闭colorbar
#     if not is_combined:
#         # single_plot：完全不显示colorbar
#         plotter.add_mesh(
#             mesh,
#             scalars="u",
#             cmap=cmap,
#             smooth_shading=True,
#             specular=0.2,
#             diffuse=0.9,
#             ambient=0.3,
#             opacity=1.0,
#             show_scalar_bar=False  # 强制关闭colorbar（关键！）
#         )
#     else:
#         # 拼接图：显示colorbar（无'u'标签+居中）
#         plotter.add_mesh(
#             mesh,
#             scalars="u",
#             cmap=cmap,
#             smooth_shading=True,
#             specular=0.2,
#             diffuse=0.9,
#             ambient=0.3,
#             opacity=1.0,
#             show_scalar_bar=True,
#             scalar_bar_args={
#                 "title": "",          # 移除'u'描述
#                 "vertical": False,    # 水平放置
#                 "position_x": 0.1,    # 左移对齐（解决偏右）
#                 "position_y": 0.02,   # 底部显示
#                 "title_font_size": 0,
#                 "label_font_size": 12,
#                 "width": 0.8,
#                 "height": 0.05
#             }
#         )
    
#     plotter.set_background('white')
#     plotter.reset_camera()
#     plotter.camera.zoom(1.1)
    
#     try:
#         with NamedTemporaryFile(suffix='.png', delete=False) as tmp:
#             plotter.screenshot(tmp.name)
#             img = plt.imread(tmp.name)
#         os.unlink(tmp.name)
#     except Exception as e:
#         print(f"[!] Error during rendering or image reading: {e}")
#         return None
#     finally:
#         plotter.close()
        
#     return img

# # --- 网格创建函数 (完全保留原有逻辑) ---
# def create_point_cloud_mesh(pos_np, u_np, mat_params, vertical_shift=-0.2, scale_factor=0.5):
#     n_patches = mat_params['n_patches']
#     patch_size = mat_params['patch_size']
#     nx, ny = patch_size
#     points_per_patch = nx * ny
#     total_points = mat_params['total_points']
    
#     pos_np = pos_np * scale_factor
    
#     if n_patches * points_per_patch != total_points:
#         print(f"警告：面片数×每个面片点数 ({n_patches}×{points_per_patch}={n_patches*points_per_patch}) 不等于总点数 ({total_points})")
#         n_patches = total_points // points_per_patch
    
#     all_faces = []
#     print(f"1. Shape of u_input_np: {u_np.shape}")
#     print(f"2. Number of points (pos_np.shape[0]): {pos_np.shape[0]}")
  
#     for p in range(n_patches):
#         start_idx = p * points_per_patch
#         end_idx = start_idx + points_per_patch
        
#         if end_idx > len(pos_np):
#             print(f"警告：面片{p}的索引超出范围（{end_idx} > {len(pos_np)}），跳过！")
#             continue
        
#         temp_grid = pv.StructuredGrid()
#         temp_grid.dimensions = [nx, ny, 1]
#         temp_grid.points = pos_np[start_idx:end_idx]
        
#         temp_tri = temp_grid.extract_surface().triangulate()
#         local_faces = temp_tri.faces
#         num_local_faces = len(local_faces) // 4
#         global_offset = start_idx
        
#         for i in range(num_local_faces):
#             face_start = i * 4
#             n = local_faces[face_start]
#             local_points = local_faces[face_start + 1: face_start + 1 + n]
#             global_points = local_points + global_offset
            
#             if all(p < len(pos_np) for p in global_points):
#                 all_faces.extend([n] + global_points.tolist())

#     all_faces_np = np.array(all_faces, dtype=np.int64)
#     full_mesh = pv.PolyData(pos_np, all_faces_np)
#     full_mesh.points[:, 2] += vertical_shift
    
#     if len(u_np) == len(pos_np):
#         full_mesh['u'] = u_np
#     else:
#         print(f"错误：u_np长度({len(u_np)})与点云数量({len(pos_np)})不匹配！")
#         return None
    
#     full_mesh.compute_normals(
#         cell_normals=False, 
#         point_normals=True, 
#         inplace=True,
#         consistent_normals=True
#     )
    
#     print(f"网格创建完成：面数={len(all_faces_np)//4}，点数={len(pos_np)}")
#     return full_mesh

# # --- 单样本绘图函数 (single_plot：强制无colorbar) ---
# def plot_single_sample_results(u_input_np, u_pred_np, u_target_np, pos_np, mat_params, save_dir, prefix, vertical_shift=-0.2, scale_factor=0.5, mse=None):
#     os.makedirs(save_dir, exist_ok=True)
    
#     full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
#     if full_mesh is None:
#         print(f"[!] 网格创建失败，跳过样本 {prefix}")
#         return
    
#     # single_plot核心：is_combined=False → 强制无colorbar
#     img_input = render_single_mesh(full_mesh.copy(), is_combined=False)
#     if img_input is not None:
#         input_save_path = os.path.join(save_dir, f"{prefix}_input.png")
#         plt.imsave(input_save_path, img_input)
#         print(f"[*] Saved single plot: {input_save_path}")

#     full_mesh['u'] = u_target_np
#     img_target = render_single_mesh(full_mesh.copy(), is_combined=False)
#     if img_target is not None:
#         target_save_path = os.path.join(save_dir, f"{prefix}_target.png")
#         plt.imsave(target_save_path, img_target)
#         print(f"[*] Saved single plot: {target_save_path}")

#     full_mesh['u'] = u_pred_np
#     img_pred = render_single_mesh(full_mesh.copy(), is_combined=False)
#     if img_pred is not None:
#         pred_save_path = os.path.join(save_dir, f"{prefix}_pred.png")
#         plt.imsave(pred_save_path, img_pred)
#         print(f"[*] Saved single plot: {pred_save_path}")

# # ==============================
# # 批量测试结果绘图函数 (完全保留逻辑)
# # ==============================
# def plot_test_results_point_cloud(save_path, mat_file_path, device, T,
#                                   vertical_shift=-0.2, scale_factor=0.5, interval=1):
#     # 1. 读取测试结果
#     data = torch.load(os.path.join(save_path, "test_results.pt"), weights_only=False)
#     inputs = data["inputs"]
#     predictions = data["predictions"]
#     targets = data["targets"]

#     if len(inputs.shape) == 3:
#         inputs = inputs.unsqueeze(1)
#         predictions = predictions.unsqueeze(1)
#         targets = targets.unsqueeze(1)

#     n_samples, n_time_steps = predictions.shape[:2]

#     # 2. 读取MATLAB数据
#     print("Loading MATLAB data for mesh reconstruction...")
#     try:
#         with h5py.File(mat_file_path, 'r') as f:
#             history_all = f['save_data']['history_all'][()]
#             params = f['save_data']['params']
            
#             pos_np = np.vstack([
#                 history_all[0, :, 0, 0],
#                 history_all[1, :, 0, 0],
#                 history_all[2, :, 0, 0]
#             ]).T
            
#             mat_params = {
#                 'n_patches': int(params['n_patches'][()].item()),
#                 'patch_size': (int(params['patch_size'][0,0].item()), int(params['patch_size'][1,0].item())),
#                 'total_points': int(params['total_points'][()].item()),
#                 'num_samples': int(params['num_samples'][()].item()),
#                 'kend': int(params['kend'][()].item()),
#                 'save_interval': int(params['save_interval'][()].item())
#             }
#         print(f"MATLAB参数解析完成：{mat_params}")
#     except Exception as e:
#         print(f"[!] 读取MAT文件失败：{e}")
#         return

#     # 3. 创建保存目录
#     single_root_dir = os.path.join(save_path, f"test_single_plots_T{T:04d}")
#     combined_root_dir = os.path.join(save_path, f"test_plots_T{T:04d}")
#     os.makedirs(single_root_dir, exist_ok=True)
#     os.makedirs(combined_root_dir, exist_ok=True)

#     # 4. 遍历绘图
#     for sample_id in range(0, n_samples, interval):
#         for t_idx in range(n_time_steps):
#             u_input_np = inputs[sample_id, t_idx].cpu().numpy().squeeze()
#             u_pred_np = predictions[sample_id, t_idx].cpu().numpy().squeeze()
#             u_target_np = targets[sample_id, t_idx].cpu().numpy().squeeze()
#             mse = np.mean((u_pred_np - u_target_np) ** 2)

#             # --- single_plot：无colorbar ---
#             single_dir = os.path.join(single_root_dir, f"t{T:04d}", f"sample_{sample_id:04d}")
#             os.makedirs(single_dir, exist_ok=True)
#             plot_single_sample_results(
#                 u_input_np, u_pred_np, u_target_np, pos_np, mat_params,
#                 save_dir=single_dir,
#                 prefix=f"sample_{sample_id:04d}_t{T:04d}",
#                 vertical_shift=vertical_shift,
#                 scale_factor=scale_factor,
#                 mse=mse
#             )

#             # --- 拼接图：有colorbar ---
#             combined_dir = os.path.join(combined_root_dir, f"t{T:04d}")
#             os.makedirs(combined_dir, exist_ok=True)
#             combined_save_path = os.path.join(combined_dir, f"sample_{sample_id:04d}_comparison.png")

#             full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
#             if full_mesh is None:
#                 print(f"[!] 网格创建失败，跳过样本 {sample_id} t={T} 的拼接图")
#                 continue
            
#             img_input = render_single_mesh(full_mesh.copy(), is_combined=True)
#             full_mesh['u'] = u_target_np
#             img_target = render_single_mesh(full_mesh.copy(), is_combined=True)
#             full_mesh['u'] = u_pred_np
#             img_pred = render_single_mesh(full_mesh.copy(), is_combined=True)

#             if img_input is None or img_target is None or img_pred is None:
#                 print(f"[!] 渲染失败，跳过样本 {sample_id} t={T} 的拼接图")
#                 continue

#             # 拼接图片：大标题
#             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
#             ax1.imshow(img_input); ax1.axis('off'); ax1.set_title("Input", fontsize=20, pad=20, loc='center')
#             ax2.imshow(img_target); ax2.axis('off'); ax2.set_title("Target", fontsize=20, pad=20, loc='center')
#             ax3.imshow(img_pred); ax3.axis('off'); ax3.set_title(f"Prediction (MSE: {mse:.4f})", fontsize=20, pad=20, loc='center')
#             plt.subplots_adjust(wspace=0.05, hspace=0.1)

#             try:
#                 plt.savefig(combined_save_path, dpi=200, bbox_inches='tight', facecolor='white')
#                 plt.close()
#                 print(f"[*] Saved combined plot: {combined_save_path}")
#             except Exception as e:
#                 print(f"[!] 保存拼接图失败：{e}")

# # ==============================
# # 批量推理结果绘图函数 (同步修改)
# # ==============================
# def plot_infer_results_point_cloud(save_path, mat_file_path, inference_times, device,
#                                    vertical_shift=-0.2, scale_factor=0.5, interval=1):
#     # 1. 读取推理结果
#     data = torch.load(os.path.join(save_path, "infer_results.pt"), weights_only=False)
#     inputs = data["inputs"]
#     predictions = data["predictions"]
#     targets = data["targets"]

#     n_samples, n_time_steps = predictions.shape[:2]
#     assert len(inference_times) == n_time_steps, f"Mismatch: len(inference_times)={len(inference_times)} vs n_time_steps={n_time_steps}"

#     # 2. 读取MATLAB数据
#     print("Loading MATLAB data for mesh reconstruction...")
#     try:
#         with h5py.File(mat_file_path, 'r') as f:
#             history_all = f['save_data']['history_all'][()]
#             params = f['save_data']['params']
            
#             pos_np = np.vstack([
#                 history_all[0, :, 0, 0],
#                 history_all[1, :, 0, 0],
#                 history_all[2, :, 0, 0]
#             ]).T
            
#             mat_params = {
#                 'n_patches': int(params['n_patches'][()].item()),
#                 'patch_size': (int(params['patch_size'][0,0].item()), int(params['patch_size'][1,0].item())),
#                 'total_points': int(params['total_points'][()].item()),
#                 'num_samples': int(params['num_samples'][()].item()),
#                 'kend': int(params['kend'][()].item()),
#                 'save_interval': int(params['save_interval'][()].item())
#             }
#         print(f"MATLAB参数解析完成：{mat_params}")
#     except Exception as e:
#         print(f"[!] 读取MAT文件失败：{e}")
#         return

#     # 3. 创建保存目录
#     single_root_dir = os.path.join(save_path, "infer_single_plots")
#     combined_root_dir = os.path.join(save_path, "infer_plots")
#     os.makedirs(single_root_dir, exist_ok=True)
#     os.makedirs(combined_root_dir, exist_ok=True)

#     # 4. 遍历绘图
#     for sample_id in range(0, n_samples, interval):
#         for t_idx in range(n_time_steps):
#             t_real = inference_times[t_idx]
            
#             u_input_np = inputs[sample_id, t_idx].cpu().numpy().squeeze()
#             u_pred_np = predictions[sample_id, t_idx].cpu().numpy().squeeze()
#             u_target_np = targets[sample_id, t_idx].cpu().numpy().squeeze()
#             mse = np.mean((u_pred_np - u_target_np) ** 2)

#             # --- single_plot：无colorbar ---
#             single_dir = os.path.join(single_root_dir, f"t{t_real:04d}", f"sample_{sample_id:04d}")
#             os.makedirs(single_dir, exist_ok=True)
#             plot_single_sample_results(
#                 u_input_np, u_pred_np, u_target_np, pos_np, mat_params,
#                 save_dir=single_dir,
#                 prefix=f"sample_{sample_id:04d}_t{t_real:04d}",
#                 vertical_shift=vertical_shift,
#                 scale_factor=scale_factor,
#                 mse=mse
#             )

#             # --- 拼接图：有colorbar ---
#             combined_dir = os.path.join(combined_root_dir, f"t{t_real:04d}")
#             os.makedirs(combined_dir, exist_ok=True)
#             combined_save_path = os.path.join(combined_dir, f"sample_{sample_id:04d}_comparison.png")

#             full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
#             if full_mesh is None:
#                 print(f"[!] 网格创建失败，跳过样本 {sample_id} t={t_real} 的拼接图")
#                 continue
            
#             img_input = render_single_mesh(full_mesh.copy(), is_combined=True)
#             full_mesh['u'] = u_target_np
#             img_target = render_single_mesh(full_mesh.copy(), is_combined=True)
#             full_mesh['u'] = u_pred_np
#             img_pred = render_single_mesh(full_mesh.copy(), is_combined=True)

#             if img_input is None or img_target is None or img_pred is None:
#                 print(f"[!] 渲染失败，跳过样本 {sample_id} t={t_real} 的拼接图")
#                 continue

#             # 拼接图片：大标题
#             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
#             ax1.imshow(img_input); ax1.axis('off'); ax1.set_title("Input", fontsize=20, pad=20, loc='center')
#             ax2.imshow(img_target); ax2.axis('off'); ax2.set_title("Target", fontsize=20, pad=20, loc='center')
#             ax3.imshow(img_pred); ax3.axis('off'); ax3.set_title(f"Prediction (MSE: {mse:.4f})", fontsize=20, pad=20, loc='center')
#             plt.subplots_adjust(wspace=0.05, hspace=0.1)

#             try:
#                 plt.savefig(combined_save_path, dpi=200, bbox_inches='tight', facecolor='white')
#                 plt.close()
#                 print(f"[*] Saved combined plot: {combined_save_path}")
#             except Exception as e:
#                 print(f"[!] 保存拼接图失败：{e}")






import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import h5py
from torch_geometric.data import Data
from tempfile import NamedTemporaryFile
import pdb

# --- 全局配置 (针对远程服务器环境) ---
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# # --- 通用渲染函数 (强制关闭single_plot的colorbar) ---
# def render_single_mesh(mesh, title=None, cmap='turbo', image_size=(600, 500), is_combined=False):
#     """
#     渲染网格：
#     - is_combined=False (single_plot)：强制关闭colorbar（无任何颜色条）
#     - is_combined=True (拼接图)：显示colorbar（无'u'标签+居中）
#     """
#     plotter = pv.Plotter(window_size=image_size, off_screen=True)
    
#     # 核心修复：single_plot强制关闭colorbar
#     if not is_combined:
#         # single_plot：完全不显示colorbar
#         plotter.add_mesh(
#             mesh,
#             scalars="u",
#             cmap=cmap,
#             smooth_shading=True,
#             specular=0.2,
#             diffuse=0.9,
#             ambient=0.3,
#             opacity=1.0,
#             show_scalar_bar=False  # 强制关闭colorbar（关键！）
#         )
#     else:
#         # 拼接图：显示colorbar（无'u'标签+居中）
#         plotter.add_mesh(
#             mesh,
#             scalars="u",
#             cmap=cmap,
#             smooth_shading=True,
#             specular=0.2,
#             diffuse=0.9,
#             ambient=0.3,
#             opacity=1.0,
#             show_scalar_bar=True,
#             scalar_bar_args={
#                 "title": "",          # 移除'u'描述
#                 "vertical": False,    # 水平放置
#                 "position_x": 0.1,    # 左移对齐（解决偏右）
#                 "position_y": 0.02,   # 底部显示
#                 "title_font_size": 0,
#                 "label_font_size": 12,
#                 "width": 0.8,
#                 "height": 0.05
#             }
#         )
    
#     plotter.set_background('white')
#     plotter.reset_camera()
#     plotter.camera.zoom(1.1)
    
#     try:
#         with NamedTemporaryFile(suffix='.png', delete=False) as tmp:
#             plotter.screenshot(tmp.name)
#             img = plt.imread(tmp.name)
#         os.unlink(tmp.name)
#     except Exception as e:
#         print(f"[!] Error during rendering or image reading: {e}")
#         return None
#     finally:
#         plotter.close()
        
#     return img

def render_single_mesh(mesh, title=None, cmap='turbo', image_size=(600, 500), is_combined=False):
    """
    渲染网格：
    - is_combined=False: 完全关闭colorbar
    - is_combined=True: 显示精简 colorbar
    - 新增: 自动裁剪 PyVista 截图白边，以符合 JCP 紧凑格式
    """
    plotter = pv.Plotter(window_size=image_size, off_screen=True)

    if not is_combined:
        plotter.add_mesh(
            mesh, scalars="u", cmap=cmap,
            smooth_shading=True, specular=0.2, diffuse=0.9, ambient=0.3,
            opacity=1.0, show_scalar_bar=False
        )
    else:
        plotter.add_mesh(
            mesh, scalars="u", cmap=cmap,
            smooth_shading=True, specular=0.2, diffuse=0.9, ambient=0.3,
            opacity=1.0, show_scalar_bar=True,
            scalar_bar_args={
                "title": "",
                "vertical": False,
                "position_x": 0.1,
                "position_y": 0.02,
                "title_font_size": 0,
                "label_font_size": 10,
                "width": 0.8,
                "height": 0.05
            }
        )

    plotter.set_background('white')
    plotter.reset_camera()
    plotter.camera.zoom(1.15)

    try:
        with NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plotter.screenshot(tmp.name, return_img=False)
            img = plt.imread(tmp.name)

        # === 关键改动：自动裁剪白边（PyVista 截图默认留白） ===
        # 1. 找非白像素
        gray = np.mean(img[:, :, :3], axis=2)
        mask = gray < 0.98
        coords = np.argwhere(mask)

        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img[y0:y1, x0:x1]     # 裁剪白边

        os.unlink(tmp.name)

    except Exception as e:
        print(f"[!] Error during rendering or image reading: {e}")
        return None
    finally:
        plotter.close()

    return img


# --- 网格创建函数 (完全保留原有逻辑) ---
def create_point_cloud_mesh(pos_np, u_np, mat_params, vertical_shift=-0.2, scale_factor=0.5):
    n_patches = mat_params['n_patches']
    patch_size = mat_params['patch_size']
    nx, ny = patch_size
    points_per_patch = nx * ny
    total_points = mat_params['total_points']
    
    pos_np = pos_np * scale_factor
    
    if n_patches * points_per_patch != total_points:
        print(f"警告：面片数×每个面片点数 ({n_patches}×{points_per_patch}={n_patches*points_per_patch}) 不等于总点数 ({total_points})")
        n_patches = total_points // points_per_patch
    
    all_faces = []
    print(f"1. Shape of u_input_np: {u_np.shape}")
    print(f"2. Number of points (pos_np.shape[0]): {pos_np.shape[0]}")
  
    for p in range(n_patches):
        start_idx = p * points_per_patch
        end_idx = start_idx + points_per_patch
        
        if end_idx > len(pos_np):
            print(f"警告：面片{p}的索引超出范围（{end_idx} > {len(pos_np)}），跳过！")
            continue
        
        temp_grid = pv.StructuredGrid()
        temp_grid.dimensions = [nx, ny, 1]
        temp_grid.points = pos_np[start_idx:end_idx]
        
        temp_tri = temp_grid.extract_surface().triangulate()
        local_faces = temp_tri.faces
        num_local_faces = len(local_faces) // 4
        global_offset = start_idx
        
        for i in range(num_local_faces):
            face_start = i * 4
            n = local_faces[face_start]
            local_points = local_faces[face_start + 1: face_start + 1 + n]
            global_points = local_points + global_offset
            
            if all(p < len(pos_np) for p in global_points):
                all_faces.extend([n] + global_points.tolist())

    all_faces_np = np.array(all_faces, dtype=np.int64)
    full_mesh = pv.PolyData(pos_np, all_faces_np)
    full_mesh.points[:, 2] += vertical_shift
    
    if len(u_np) == len(pos_np):
        full_mesh['u'] = u_np
    else:
        print(f"错误：u_np长度({len(u_np)})与点云数量({len(pos_np)})不匹配！")
        return None
    
    full_mesh.compute_normals(
        cell_normals=False, 
        point_normals=True, 
        inplace=True,
        consistent_normals=True
    )
    
    print(f"网格创建完成：面数={len(all_faces_np)//4}，点数={len(pos_np)}")
    return full_mesh

# --- 单样本绘图函数 (single_plot：强制无colorbar) ---
def plot_single_sample_results(u_input_np, u_pred_np, u_target_np, pos_np, mat_params, save_dir, prefix, vertical_shift=-0.2, scale_factor=0.5, mse=None):
    os.makedirs(save_dir, exist_ok=True)
    
    full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
    if full_mesh is None:
        print(f"[!] 网格创建失败，跳过样本 {prefix}")
        return
    
    # single_plot核心：is_combined=False → 强制无colorbar
    img_input = render_single_mesh(full_mesh.copy(), is_combined=False)
    if img_input is not None:
        input_save_path = os.path.join(save_dir, f"{prefix}_input.png")
        plt.imsave(input_save_path, img_input)
        print(f"[*] Saved single plot: {input_save_path}")

    full_mesh['u'] = u_target_np
    img_target = render_single_mesh(full_mesh.copy(), is_combined=False)
    if img_target is not None:
        target_save_path = os.path.join(save_dir, f"{prefix}_target.png")
        plt.imsave(target_save_path, img_target)
        print(f"[*] Saved single plot: {target_save_path}")

    full_mesh['u'] = u_pred_np
    img_pred = render_single_mesh(full_mesh.copy(), is_combined=False)
    if img_pred is not None:
        pred_save_path = os.path.join(save_dir, f"{prefix}_pred.png")
        plt.imsave(pred_save_path, img_pred)
        print(f"[*] Saved single plot: {pred_save_path}")

# ==============================
# 批量测试结果绘图函数 (完全保留逻辑)
# ==============================
def plot_test_results_point_cloud(save_path, mat_file_path, device, T,
                                  vertical_shift=-0.2, scale_factor=0.5, interval=1):
    # 1. 读取测试结果
    data = torch.load(os.path.join(save_path, "test_results.pt"), weights_only=False)
    inputs = data["inputs"]
    predictions = data["predictions"]
    targets = data["targets"]

    if len(inputs.shape) == 3:
        inputs = inputs.unsqueeze(1)
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)

    n_samples, n_time_steps = predictions.shape[:2]

    # 2. 读取MATLAB数据
    print("Loading MATLAB data for mesh reconstruction...")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            history_all = f['save_data']['history_all'][()]
            params = f['save_data']['params']
            
            pos_np = np.vstack([
                history_all[0, :, 0, 0],
                history_all[1, :, 0, 0],
                history_all[2, :, 0, 0]
            ]).T
            
            mat_params = {
                'n_patches': int(params['n_patches'][()].item()),
                'patch_size': (int(params['patch_size'][0,0].item()), int(params['patch_size'][1,0].item())),
                'total_points': int(params['total_points'][()].item()),
                'num_samples': int(params['num_samples'][()].item()),
                'kend': int(params['kend'][()].item()),
                'save_interval': int(params['save_interval'][()].item())
            }
        print(f"MATLAB参数解析完成：{mat_params}")
    except Exception as e:
        print(f"[!] 读取MAT文件失败：{e}")
        return

    # 3. 创建保存目录
    single_root_dir = os.path.join(save_path, f"test_single_plots_T{T:04d}")
    combined_root_dir = os.path.join(save_path, f"test_plots_T{T:04d}")
    os.makedirs(single_root_dir, exist_ok=True)
    os.makedirs(combined_root_dir, exist_ok=True)

    # 4. 遍历绘图
    for sample_id in range(0, n_samples, interval):
        for t_idx in range(n_time_steps):
            u_input_np = inputs[sample_id, t_idx].cpu().numpy().squeeze()
            u_pred_np = predictions[sample_id, t_idx].cpu().numpy().squeeze()
            u_target_np = targets[sample_id, t_idx].cpu().numpy().squeeze()
            mse = np.mean((u_pred_np - u_target_np) ** 2)

            # --- single_plot：无colorbar ---
            single_dir = os.path.join(single_root_dir, f"t{T:04d}", f"sample_{sample_id:04d}")
            os.makedirs(single_dir, exist_ok=True)
            plot_single_sample_results(
                u_input_np, u_pred_np, u_target_np, pos_np, mat_params,
                save_dir=single_dir,
                prefix=f"sample_{sample_id:04d}_t{T:04d}",
                vertical_shift=vertical_shift,
                scale_factor=scale_factor,
                mse=mse
            )

            # --- 拼接图：有colorbar ---
            combined_dir = os.path.join(combined_root_dir, f"t{T:04d}")
            os.makedirs(combined_dir, exist_ok=True)
            combined_save_path = os.path.join(combined_dir, f"sample_{sample_id:04d}_comparison.png")

            full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
            if full_mesh is None:
                print(f"[!] 网格创建失败，跳过样本 {sample_id} t={T} 的拼接图")
                continue
            
            img_input = render_single_mesh(full_mesh.copy(), is_combined=True)
            full_mesh['u'] = u_target_np
            img_target = render_single_mesh(full_mesh.copy(), is_combined=True)
            full_mesh['u'] = u_pred_np
            img_pred = render_single_mesh(full_mesh.copy(), is_combined=True)

            if img_input is None or img_target is None or img_pred is None:
                print(f"[!] 渲染失败，跳过样本 {sample_id} t={T} 的拼接图")
                continue

            # 拼接图片：大标题
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            ax1.imshow(img_input); ax1.axis('off'); ax1.set_title("Input", fontsize=20, pad=20, loc='center')
            ax2.imshow(img_target); ax2.axis('off'); ax2.set_title("Target", fontsize=20, pad=20, loc='center')
            ax3.imshow(img_pred); ax3.axis('off'); ax3.set_title(f"Prediction (MSE: {mse:.4f})", fontsize=20, pad=20, loc='center')
            plt.subplots_adjust(wspace=0.05, hspace=0.1)

            try:
                plt.savefig(combined_save_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"[*] Saved combined plot: {combined_save_path}")
            except Exception as e:
                print(f"[!] 保存拼接图失败：{e}")

# ==============================
# 批量推理结果绘图函数 (同步修改)
# ==============================
def plot_infer_results_point_cloud(save_path, mat_file_path, inference_times, device,
                                   vertical_shift=-0.2, scale_factor=0.5, interval=1):
    # 1. 读取推理结果
    data = torch.load(os.path.join(save_path, "infer_results.pt"), weights_only=False)
    inputs = data["inputs"]
    predictions = data["predictions"]
    targets = data["targets"]

    n_samples, n_time_steps = predictions.shape[:2]
    assert len(inference_times) == n_time_steps, f"Mismatch: len(inference_times)={len(inference_times)} vs n_time_steps={n_time_steps}"

    # 2. 读取MATLAB数据
    print("Loading MATLAB data for mesh reconstruction...")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            history_all = f['save_data']['history_all'][()]
            params = f['save_data']['params']
            
            pos_np = np.vstack([
                history_all[0, :, 0, 0],
                history_all[1, :, 0, 0],
                history_all[2, :, 0, 0]
            ]).T
            
            mat_params = {
                'n_patches': int(params['n_patches'][()].item()),
                'patch_size': (int(params['patch_size'][0,0].item()), int(params['patch_size'][1,0].item())),
                'total_points': int(params['total_points'][()].item()),
                'num_samples': int(params['num_samples'][()].item()),
                'kend': int(params['kend'][()].item()),
                'save_interval': int(params['save_interval'][()].item())
            }
        print(f"MATLAB参数解析完成：{mat_params}")
    except Exception as e:
        print(f"[!] 读取MAT文件失败：{e}")
        return

    # 3. 创建保存目录
    single_root_dir = os.path.join(save_path, "infer_single_plots")
    combined_root_dir = os.path.join(save_path, "infer_plots")
    os.makedirs(single_root_dir, exist_ok=True)
    os.makedirs(combined_root_dir, exist_ok=True)

    # 4. 遍历绘图
    for sample_id in range(0, n_samples, interval):
        for t_idx in range(n_time_steps):
            t_real = inference_times[t_idx]
            
            u_input_np = inputs[sample_id, t_idx].cpu().numpy().squeeze()
            u_pred_np = predictions[sample_id, t_idx].cpu().numpy().squeeze()
            u_target_np = targets[sample_id, t_idx].cpu().numpy().squeeze()
            mse = np.mean((u_pred_np - u_target_np) ** 2)

            # --- single_plot：无colorbar ---
            single_dir = os.path.join(single_root_dir, f"t{t_real:04d}", f"sample_{sample_id:04d}")
            os.makedirs(single_dir, exist_ok=True)
            plot_single_sample_results(
                u_input_np, u_pred_np, u_target_np, pos_np, mat_params,
                save_dir=single_dir,
                prefix=f"sample_{sample_id:04d}_t{t_real:04d}",
                vertical_shift=vertical_shift,
                scale_factor=scale_factor,
                mse=mse
            )

            # --- 拼接图：有colorbar ---
            combined_dir = os.path.join(combined_root_dir, f"t{t_real:04d}")
            os.makedirs(combined_dir, exist_ok=True)
            combined_save_path = os.path.join(combined_dir, f"sample_{sample_id:04d}_comparison.png")

            full_mesh = create_point_cloud_mesh(pos_np, u_input_np, mat_params, vertical_shift, scale_factor)
            if full_mesh is None:
                print(f"[!] 网格创建失败，跳过样本 {sample_id} t={t_real} 的拼接图")
                continue
            
            img_input = render_single_mesh(full_mesh.copy(), is_combined=True)
            full_mesh['u'] = u_target_np
            img_target = render_single_mesh(full_mesh.copy(), is_combined=True)
            full_mesh['u'] = u_pred_np
            img_pred = render_single_mesh(full_mesh.copy(), is_combined=True)

            if img_input is None or img_target is None or img_pred is None:
                print(f"[!] 渲染失败，跳过样本 {sample_id} t={t_real} 的拼接图")
                continue

            # 拼接图片：大标题
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            ax1.imshow(img_input); ax1.axis('off'); ax1.set_title("Input", fontsize=20, pad=20, loc='center')
            ax2.imshow(img_target); ax2.axis('off'); ax2.set_title("Target", fontsize=20, pad=20, loc='center')
            ax3.imshow(img_pred); ax3.axis('off'); ax3.set_title(f"Prediction (MSE: {mse:.4f})", fontsize=20, pad=20, loc='center')
            plt.subplots_adjust(wspace=0.05, hspace=0.1)

            try:
                plt.savefig(combined_save_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"[*] Saved combined plot: {combined_save_path}")
            except Exception as e:
                print(f"[!] 保存拼接图失败：{e}")