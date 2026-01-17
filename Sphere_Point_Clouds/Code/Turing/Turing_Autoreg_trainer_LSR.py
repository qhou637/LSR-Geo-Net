


import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
#import time
import torch
import csv
import os
import pdb
import pandas as pd

import os
import time
import torch
import pandas as pd
import torch.nn as nn


class Autoreg_LR_Trainer_LSR:
    def __init__(self, solver, model,  train_loader, val_loader, test_loader,nepochs,optimizer, 
                 scheduler=None, save_path="SOS_res", loss_fn='l2',all_steps=None,idx = 2,forward_step=1,
                 device='cuda',lons=None, lats=None,
                T_init=1e5, T_min=1e2, T_decay=0.95):
        self.solver = solver
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.nepochs = nepochs
        self.scheduler = scheduler
        self.save_path = save_path
        self.loss_fn = loss_fn
        self.lowest_loss = 100
        self.device = device
        self.idx = idx
        self.all_steps = all_steps
        self.forward_step = forward_step
        self.lons = lons
        self.lats = lats


    

        # 温度退火参数
        self.T_init = T_init
        self.T_min = T_min
        self.T_decay = T_decay

        os.makedirs(self.save_path, exist_ok=True)

    def update_temperature_recursive(self, model, epoch):
        """递归更新所有 spectral 模块的温度"""
        for module in model.modules():
            if hasattr(module, 'update_temperature'):
                module.update_temperature(epoch, T_init=self.T_init,
                                          T_min=self.T_min, decay=self.T_decay)
                
    def huberloss_sphere(self, prd, tar, delta=1.0, relative=True):
        residual = prd - tar
        abs_res = torch.abs(residual)
        
=
        loss = torch.where(
            abs_res <= delta,
            0.5 * residual**2,
            delta * (abs_res - 0.5 * delta)
        )
        
   
        loss = self.solver.integrate_grid(loss, dimensionless=True).sum(dim=-1)
        
        if relative:
            denom = self.solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
            loss = loss / (denom + 1e-8)
        
        return loss.mean()
    



     
        
    def l2loss_sphere(self, prd, tar, relative=True, squared=True):
        # pdb.set_trace()
        loss = self.solver.integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
        if relative:
            loss = loss / self.solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
        if not squared:
            loss = torch.sqrt(loss)
        loss = loss.mean()
        return loss
 

    def spectral_l2loss_sphere(self, prd, tar, relative=True, squared=False):
        coeffs = torch.view_as_real(self.solver.sht(prd - tar))
        coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        loss = torch.sum(norm2, dim=(-1,-2))

        if relative:
            tar_coeffs = torch.view_as_real(self.solver.sht(tar))
            tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
            tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
            tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
            loss = loss / tar_norm2

        if not squared:
            loss = torch.sqrt(loss)
        return loss.mean()
     

    def compute_loss(self, prd, tar, model=None, relative=True):
                """
                Compute loss with optional L2 regularization on model parameters.
                
                Args:
                    prd: predicted field [batch, ...]
                    tar: target field [batch, ...]
                    model: torch.nn.Module (optional) — if provided, applies L2 regularization
                    relative: whether to use relative loss
                    alpha: regularization coefficient (e.g., 1e-4)
                """
                if self.loss_fn == 'hub l2':
                    # loss = 0.3 * self.l2loss_sphere(prd, tar) + 0.7 * self.huberloss_sphere(prd, tar, delta=1.5)
                    loss =  self.huberloss_sphere(prd, tar, delta=1)
                elif self.loss_fn == 'spectral l2':
                    loss = self.spectral_l2loss_sphere(prd, tar, relative=relative)
                elif self.loss_fn == 'l2':
                    loss = self.l2loss_sphere(prd, tar, relative=relative)

                else:
                    raise ValueError(f"Unknown loss function: {self.loss_fn}")



                return loss
    def train(self):
        train_logs = []
        train_start = time.time()

        
        for epoch in range(self.nepochs):
            epoch_start = time.time()
           =
            train_loss, grad_logs = self.train_epoch(epoch)  
            val_loss = self.val_epoch(epoch)
=

            if self.scheduler is not None:
                self.scheduler.step()

   
            is_best = val_loss < self.lowest_loss
            if is_best:
                self.lowest_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(self.save_path, 'model_best_weight.pth'))
                print('Saved new best model with lowest_loss %f' % self.lowest_loss)

            train_logs.append({
                "Epoch": epoch,
                "Training Loss": train_loss,
                "Validation Loss": val_loss
            })

            print(f'--------------------------------------------------------------------------------')
            print(f'Epoch {epoch} summary:')
            print(f'time taken: {time.time() - epoch_start:.2f}s')
            print(f'Training Loss: {train_loss:.6f}')
            print(f'Validation Loss: {val_loss:.6f}')

            df = pd.DataFrame(train_logs)
            df.to_csv(os.path.join(self.save_path, 'training_loss.csv'), index=False)

    

        train_time = time.time() - train_start
        print(f'Training completed in {train_time:.2f} seconds.')
        return self.lowest_loss, train_time
    

    def merge_all_gradient_logs(self):

        try:
            # 收集所有梯度日志文件
            all_grad_dfs = []
            for epoch in range(self.nepochs):
                log_path = os.path.join(self.gradient_log_dir, f"gradients_epoch_{epoch}.csv")
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    all_grad_dfs.append(df)
                    print(f"已合并 epoch {epoch} 的梯度日志")
                else:
                    print(f"警告：未找到 epoch {epoch} 的梯度日志文件，已跳过")

            # 合并所有数据并排序
            if all_grad_dfs:
                merged_df = pd.concat(all_grad_dfs, ignore_index=True)
                # 按 epoch 和 batch 升序排序，确保时间顺序正确
                merged_df = merged_df.sort_values(by=["epoch", "batch"], ignore_index=True)
                
                # 保存合并后的大CSV
                merged_path = os.path.join(self.save_path, "all_gradients.csv")
                merged_df.to_csv(merged_path, index=False)
        
            else:
                print("未找到任何梯度日志文件，跳过合并")
        except Exception as e:
            print(f"合并梯度日志时出错：{str(e)}")


 



    def save_params_to_csv(self, params_list, phase, epoch, save_path):
       
        import pandas as pd
        import os

        os.makedirs(save_path, exist_ok=True)
        csv_data = []

        for layer_idx in range(len(params_list)):
            
            unet_params = params_list[layer_idx]
            if not isinstance(unet_params, list) or len(unet_params) == 0:
                continue  
            params = unet_params[0]
            if not isinstance(params, dict):
                continue 

  
            amplitude = params.get('amplitude', torch.tensor([])).detach().cpu().tolist()
            shift = params.get('shift', torch.tensor([])).detach().cpu().tolist()
            beta = params.get('beta', torch.tensor([])).detach().cpu().tolist()
            h_par = params.get('hypera', torch.tensor([])).detach().cpu().tolist()

          
            row = {
                'Phase': phase,
                'Epoch': epoch,
                'Layer': layer_idx,
                'Amplitude': str(amplitude),
                'Shift': str(shift),
                'Beta': str(beta),
                'h_par': str(h_par)
            }
            csv_data.append(row)

     
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(save_path, 'params_log.csv')
        
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    def train_epoch(self, epoch):
            self.model.train()
            acc_loss = 0
            grad_logs = []  
            
            for batch_idx, batch in enumerate(self.train_loader):  
                steps = self.all_steps[:self.forward_step]
                pdb.set_trace()
                self.model.train()
                self.optimizer.zero_grad()
                batch_loss=0
                pdb.set_trace()
                inp = batch[:, 0, :, :, :].to(self.device)  
                    
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for  step_idx in steps:
                        tar = batch[:, step_idx, :, :, :].to(self.device)  
                        prd, params_list = self.model(inp)  
                        step_loss = self.compute_loss(prd, tar, relative=True)
                        batch_loss += step_loss
                        # inp = tar # Teaching Forcing
                        inp = prd

         
                batch_loss.backward()
                
    
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_logs.append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "param_name": name,
                            "grad_norm": grad_norm
                        })
                
              
                self.optimizer.step()
                acc_loss += batch_loss.item()
            
            avg_loss = acc_loss / len(self.train_loader)
            self.save_params_to_csv(params_list, phase='train', epoch=epoch, save_path= self.save_path)
            return avg_loss, grad_logs 
    

          
    def val_epoch(self,epoch):
        self.model.eval()
        val_loss = 0
        steps =  self.all_steps[:self.forward_step]
        with torch.no_grad():
            for batch in self.val_loader:
                batch_loss = 0
                inp = batch[:, 0, :, :, :].to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for step_idx in steps:
                        tar = batch[:, step_idx, :, :, :].to(self.device)
                        prd,params_list = self.model(inp) 
                        
                        mse_loss_val = self.compute_loss(prd, tar,relative=True)
                        # next_input = tar
                        batch_loss += mse_loss_val 
                        inp = prd
                        # inp = tar
                val_loss += batch_loss.item()
 


        avg_val_loss = val_loss / len(self.val_loader)
        self.save_params_to_csv(params_list, phase='val', epoch=epoch, save_path= self.save_path)
        return avg_val_loss


    def test(self, test_loader):
        os.makedirs(self.save_path, exist_ok=True)
        self.model.eval()
        test_loss = 0
        predictions, targets, inputs = [], [], []
     
        with torch.no_grad():
            for batch in test_loader:
                # pdb.set_trace()
                inp = batch[:, 0, :, :, :].to(self.device)  # t=0
                tar = batch[:, self.idx, :, :, :].to(self.device)  # t=1
      
                prd,_ = self.model(inp)
          
                loss =  self.compute_loss(prd, tar, relative=True)
                test_loss += loss.item() 


                predictions.append(prd) 
                targets.append(tar)     
                inputs.append(inp)
        pdb.set_trace()

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.6f}")

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        inputs = torch.cat(inputs, dim=0)

 
        torch.save({
            "test_loss": test_loss,
            "predictions": predictions,  # [B, C, H, W]
            "targets": targets,          # [B, C, H, W]
            "inputs": inputs,            # [B, C, H, W]
        }, os.path.join(self.save_path, "test_results.pt"))
        
        return predictions, test_loss, targets, inputs

   

    def test_inference(self, out_test_initial,test_loss=None):


        model_path = os.path.join(self.save_path, 'model_best_weight.pth')
        checkpoint = torch.load(model_path,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()

        all_inputs = []
        all_predictions = []
        all_targets = []
        mse_error_test_all = []
        # steps =  self.all_steps[self.forward_step:]
        pdb.set_trace()
        steps =  self.all_steps[1:]



        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
             
                inp = out_test_initial[i].unsqueeze(0).to(self.device)  # [1, C, H, W]

                sample_inputs = []
                sample_predictions = []
                sample_targets = []
                sample_errors = []
            
                for ind in steps:
                 
                    prd,_ = self.model(inp) 
    

           
                    target = batch[:, ind, :, :].float().to(self.device)

    
                    mse_error = self.compute_loss(prd, target, relative=True)

                    sample_inputs.append(inp.cpu())        
                    sample_predictions.append(prd.cpu())   
                    sample_targets.append(target.cpu())    
                    sample_errors.append(mse_error.item()) 

              
                    inp = prd

           
                # pdb.set_trace()
                all_inputs.append(torch.stack(sample_inputs, dim=1))
                all_predictions.append(torch.stack(sample_predictions, dim=1))
                all_targets.append(torch.stack(sample_targets, dim=1))
                mse_error_test_all.append(sample_errors)
        pdb.set_trace()
  
        inputs_tensor = torch.cat(all_inputs, dim=0)
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)


        torch.save({
            "inputs": inputs_tensor,
            "predictions": predictions_tensor,
            "targets": targets_tensor,
        }, os.path.join(self.save_path, "infer_results.pt"))

     
        csv_path = os.path.join(self.save_path, "infer_mse_error.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['sample_idx'] + [f't{t}' for t in steps] 
            writer.writerow(header)
            for idx, sample_errors in enumerate(mse_error_test_all):
                writer.writerow([idx] + sample_errors)




        csv_summary_path = os.path.join(self.save_path, "infer_mse_mean.csv")
        with open(csv_summary_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            

            header = ['sample_idx'] + [f't{self.idx}'] + [f't{t}' for t in steps if t != self.idx]
            writer.writerow(header)
            
    
            mean_errors = np.array(mse_error_test_all).mean(axis=0)
            
        
            data_row = [0, test_loss]  # sample_idx和test_loss
            for i, t in enumerate(steps):
                if t != self.idx:  #
                    data_row.append(mean_errors[i])
            
            writer.writerow(data_row)

