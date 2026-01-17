


import os
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, '/home/qianhou/torch-harmonics')

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pdb
import numpy as np
from torch_harmonics.examples import ShallowWaterSolver
import cartopy.crs as ccrs


def plot_sample_comparison(result_list, solver, sample_id=0, save_dir="test_plots", cmap='twilight_shifted', t=None):
    import os
    os.makedirs(save_dir, exist_ok=True)

    var_names = ['u', 'v']

    data, pred, target = result_list  #
    data = data.squeeze(0)     # [2, H, W]
    pred = pred.squeeze(0)
    target = target.squeeze(0)


    fig_size = (8, 7) 
    
    for i, var in enumerate(var_names):
        # --- Prepare data for plotting ---
        d_in = data[i]
        d_gt = target[i]
        d_pred = pred[i]
        d_err = torch.abs(d_pred - d_gt)

        # Individual vmin/vmax for each plot
        vmin_input, vmax_input = d_in.min().item(), d_in.max().item()
        vmin_target, vmax_target = d_gt.min().item(), d_gt.max().item()
        vmin_pred, vmax_pred = d_pred.min().item(), d_pred.max().item()

        def plot_with_cbar(data, cmap, vmin, vmax, save_path):
            fig = plt.figure(figsize=fig_size)
            
            # 调用plot_griddata（仅返回im对象，无cbar）
            data_cpu = data.detach().cpu()
            im = solver.plot_griddata(
                data_cpu, 
                fig, 
                cmap=cmap, 
                vmax=vmax, 
                vmin=vmin, 
                projection='3d',
                title='' 
            )
            

            cbar = None
 
            for ax in fig.axes:
                if hasattr(ax, 'colorbar') or 'colorbar' in str(ax):
                    cbar = ax
                    break
   
            if cbar is None:
                try:
                  
                    cbar = fig.axes[-1] if fig.axes else None
                except:
                    pass
            
     
            if cbar is not None:
               
                cbar.tick_params(labelsize=12)
                
            
                formatter = mticker.FormatStrFormatter('%.2f')
                cbar.xaxis.set_major_formatter(formatter)
                cbar.yaxis.set_major_formatter(formatter)                  
          
                cbar.figure.canvas.draw()
  
            plt.title('')
            plt.suptitle('')
            for ax in fig.axes:
                ax.set_title('')
            
           
            plt.tight_layout()
            
 
            fig.savefig(
                save_path, 
                dpi=300,  ]
                bbox_inches='tight', 
                pad_inches=0.1,      
                facecolor='white'     
            )
            plt.close(fig)

 
        t_str = f"_t{t:02d}" if t is not None else ""
        prefix = f"{sample_id}{t_str}_{var}"
        

        plot_with_cbar(d_in, cmap, vmin_input, vmax_input,
                       os.path.join(save_dir, f"{prefix}_input.png"))
        plot_with_cbar(d_gt, cmap, vmin_target, vmax_target,
                       os.path.join(save_dir, f"{prefix}_target.png"))
        plot_with_cbar(d_pred, cmap, vmin_pred, vmax_pred,
                       os.path.join(save_dir, f"{prefix}_pred.png"))
        plot_with_cbar(d_err, 'Reds', None, None,
                       os.path.join(save_dir, f"{prefix}_error.png"))

def plot_test_results(solver, save_path, interval=10, projection='3d', cmap='twilight_shifted'):
    print('plot test results')

    data = torch.load(os.path.join(save_path, "test_results.pt"),weights_only=False)
    save_dir = os.path.join(save_path, "test_plots")
    os.makedirs(save_dir, exist_ok=True)

    inputs = data["inputs"]
    predictions = data["predictions"]
    targets = data["targets"]

  

    for sample_id in range(0, inputs.shape[0], interval):
     
        input_sample = inputs[sample_id:sample_id+1]
        pred_sample = predictions[sample_id:sample_id+1]
        target_sample = targets[sample_id:sample_id+1]

        result_list = [input_sample, pred_sample, target_sample]
        plot_sample_comparison(result_list, solver, sample_id=sample_id, save_dir=save_dir, cmap=cmap)


def plot_infer_results(solver, save_path, inference_steps, interval=10, cmap='twilight_shifted'):
    import os
    import torch
    import matplotlib.pyplot as plt
    print('Plot inference results...')

    data = torch.load(os.path.join(save_path, "infer_results.pt"),weights_only=False)
    save_dir = os.path.join(save_path, "infer_plots")
    os.makedirs(save_dir, exist_ok=True)

    inputs = data["inputs"]          # [N, T, 3, H, W]
    predictions = data["predictions"]
    targets = data["targets"]

    n_samples, nfuture = predictions.shape[:2]
    assert len(inference_steps) == nfuture, f"Mismatch: len(inference_times)={len(inference_steps)} vs nfuture={nfuture}"

    for sample_id in range(0, n_samples, interval):
        for t in range(nfuture):
            input_sample = inputs[sample_id:sample_id+1, t]
            pred_sample = predictions[sample_id:sample_id+1, t]
            target_sample = targets[sample_id:sample_id+1, t]

            result_list = [input_sample, pred_sample, target_sample]
            t_real = 10*inference_steps[t]

            timestep_dir = os.path.join(save_dir, f"t{t_real:02d}")
            os.makedirs(timestep_dir, exist_ok=True)

            plot_sample_comparison(result_list, solver,
                                   sample_id=sample_id,
                                   save_dir=timestep_dir,
                                   cmap=cmap,
                                   t=t_real)

