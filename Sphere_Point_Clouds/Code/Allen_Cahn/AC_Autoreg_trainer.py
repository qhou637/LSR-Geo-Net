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

class Autoreg_LR_Trainer:
    def __init__(self, solver, model,  train_loader, val_loader, test_loader,nepochs,optimizer, 
                 scheduler=None, save_path="SOS_res", loss_fn='l2',all_steps=None,idx = 2,forward_step=1,device='cuda',
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
        
        # 分段计算 Huber Loss
        loss = torch.where(
            abs_res <= delta,
            0.5 * residual**2,
            delta * (abs_res - 0.5 * delta)
        )
        
        # 球面积分（保持能量守恒）
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
    
    # def l2loss_grid(self, prd, tar, relative=True, squared=True, eps=1e-8):
    #     # Compute squared error over (C, H, W)
    #     sq_error = ((prd - tar) ** 2).sum(dim=(-3, -2, -1))  # [B]

    #     if relative:
    #         denom = (tar ** 2).sum(dim=(-3, -2, -1)) + eps  # [B]
    #         loss = sq_error / denom
    #     else:
    #         loss = sq_error

    #     if not squared:
    #         loss = torch.sqrt(loss)

    #     return loss.mean()  # scalar loss
    #     # return loss  # scalar loss

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



    # def train(self):
    #     train_logs = []
    #     train_start = time.time()
    #     # 更新模型中所有 spectral 模块的温度
       
    #     for epoch in range(self.nepochs):
    #         # pdb.set_trace()
    #         epoch_start = time.time()
    #         # self.update_temperature_recursive(self.model, epoch)
    #         train_loss = self.train_epoch(epoch)
    #         val_loss = self.val_epoch(epoch)

    #         if self.scheduler is not None:
    #             self.scheduler.step()

    #         # Checkpoint
    #         is_best = val_loss < self.lowest_loss
    #         if is_best:
    #             self.lowest_loss = val_loss
    #             torch.save({
    #                 'epoch': epoch + 1,
    #                 'model_state_dict': self.model.state_dict(),
    #                 'optimizer_state_dict': self.optimizer.state_dict(),
    #                 'loss': val_loss,
    #             }, os.path.join(self.save_path, 'model_best_weight.pth'))
    #             print('Saved new best model with lowest_loss %f' % self.lowest_loss)

    #         train_logs.append({
    #             "Epoch": epoch,
    #             "Training Loss": train_loss,
    #             "Validation Loss": val_loss
    #         })

    #         print(f'--------------------------------------------------------------------------------')
    #         print(f'Epoch {epoch} summary:')
    #         print(f'time taken: {time.time() - epoch_start:.2f}s')
    #         print(f'Training Loss: {train_loss:.6f}')
    #         print(f'Validation Loss: {val_loss:.6f}')

    #         # Save logs
    #         df = pd.DataFrame(train_logs)
    #         df.to_csv(os.path.join(self.save_path, 'training_loss.csv'), index=False)

    #     train_time = time.time() - train_start
    #     print(f'Training completed in {train_time:.2f} seconds.')
    #     return self.lowest_loss, train_time
    
    # def standardize_data(self,x):
    #     mean = x.mean(dim=(0, 2, 3), keepdim=True)
    #     std = x.std(dim=(0, 2, 3), keepdim=True)
    #     x_std = (x - mean) / std
    #     return x_std, mean, std
        


        
    # def train_epoch(self, epoch):
    #         self.model.train()
    #         acc_loss = 0
    #         for batch in self.train_loader:
    #             # pdb.set_trace()
    #             steps = self.all_steps[:self.forward_step]
    #             self.model.train()
    #             self.optimizer.zero_grad()
    #             batch_loss=0
    #             inp = batch[:, 0, :, :, :].to(self.device)  # t=0
    #             # pdb.set_trace()
              
    #             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    #                 for  step_idx in steps:
    #                     # pdb.set_trace()
    #                     tar = batch[:, step_idx, :, :, :].to(self.device)  

    #                     prd = self.model(inp)  

    #                     # next_input = tar
    #                     step_loss = self.compute_loss(prd, tar,relative=True)

    #                     batch_loss += step_loss
    #                     # inp = prd
    #                     inp = tar


    #             batch_loss.backward()
    #             self.optimizer.step()
    #             acc_loss += batch_loss.item()
    #             # acc_loss += batch_loss.item() * inp.size(0)

    #         # avg_loss = acc_loss / len(self.train_loader.dataset)
    #         avg_loss = acc_loss / len(self.train_loader)
    #         return avg_loss



    def train(self):
        train_logs = []
        train_start = time.time()
        # 创建梯度日志保存目录
        self.gradient_log_dir = os.path.join(self.save_path, "gradient_logs")
        os.makedirs(self.gradient_log_dir, exist_ok=True)  # 确保目录存在
        
        for epoch in range(self.nepochs):
            epoch_start = time.time()
            # 训练 epoch 并收集梯度信息
            train_loss, grad_logs = self.train_epoch(epoch)  # 获取当前epoch的梯度日志
            val_loss = self.val_epoch(epoch)

            # 保存当前epoch的梯度日志为CSV
            grad_df = pd.DataFrame(grad_logs)
            grad_df.to_csv(
                os.path.join(self.gradient_log_dir, f"gradients_epoch_{epoch}.csv"),
                index=False
            )

            if self.scheduler is not None:
                self.scheduler.step()

            # （保留你的 checkpoint 和日志记录代码）
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

        # 训练结束后，合并所有梯度日志为一个大CSV
        self.merge_all_gradient_logs()

        train_time = time.time() - train_start
        print(f'Training completed in {train_time:.2f} seconds.')
        return self.lowest_loss, train_time
    

    def merge_all_gradient_logs(self):
        """合并所有epoch的梯度日志为一个总CSV文件，按epoch和batch排序"""
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
                print(f"所有梯度日志已合并为：{merged_path}")
                print(f"总记录数：{len(merged_df)} 条")
                print(f"涉及参数数量：{merged_df['param_name'].nunique()} 个")
            else:
                print("未找到任何梯度日志文件，跳过合并")
        except Exception as e:
            print(f"合并梯度日志时出错：{str(e)}")


        


        
    def train_epoch(self, epoch):
            self.model.train()
            acc_loss = 0
            grad_logs = []  # 存储当前epoch的梯度日志
            
            for batch_idx, batch in enumerate(self.train_loader):  # 记录批次索引
                steps = self.all_steps[:self.forward_step]
                self.model.train()
                self.optimizer.zero_grad()
                batch_loss=0
                pdb.set_trace()
                inp = batch[:, 0, :, :, :].to(self.device)  # t=0
                    
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for  step_idx in steps:
                        tar = batch[:, step_idx, :, :, :].to(self.device)  
                        prd = self.model(inp) 
                        pdb.set_trace() 
                        step_loss = self.compute_loss(prd, tar, relative=True)
                        batch_loss += step_loss
                        inp = tar

                # 计算梯度
                batch_loss.backward()
                
                # 收集梯度信息并添加到日志
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_logs.append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "param_name": name,
                            "grad_norm": grad_norm
                        })
                
                # 更新参数
                self.optimizer.step()
                acc_loss += batch_loss.item()
            
            avg_loss = acc_loss / len(self.train_loader)
            return avg_loss, grad_logs  # 返回损失和梯度日志
    

    
    
          

    
    
          
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
                        prd = self.model(inp) 
                        
                        mse_loss_val = self.compute_loss(prd, tar,relative=True)
                        # next_input = tar
                        batch_loss += mse_loss_val 
                        # inp = prd
                        inp = tar
                val_loss += batch_loss.item()
                # val_loss += batch_loss.item()* inp.size(0)

        # avg_val_loss = val_loss / len(self.val_loader.dataset)
        avg_val_loss = val_loss / len(self.val_loader)
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
                # 单步预测
                prd = self.model(inp)
          
                loss =  self.compute_loss(prd, tar, relative=True)
                test_loss += loss.item() 


                predictions.append(prd)  # 保存单步预测
                targets.append(tar)      # 保存真实值
                inputs.append(inp)
        pdb.set_trace()
        # test_loss /= len(test_loader.dataset)
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.6f}")

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        inputs = torch.cat(inputs, dim=0)

        # 保存测试结果
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
                # out_test_initial[i]是第1时刻的预测数据（作为初始输入）
                inp = out_test_initial[i].unsqueeze(0).to(self.device)  # [1, C, H, W]

                sample_inputs = []
                sample_predictions = []
                sample_targets = []
                sample_errors = []
                # pdb.set_trace()
                # 从第2时刻开始预测，一直到第 nfuture-1 时刻
                # pdb.set_trace()
                for ind in steps:
                 
                    prd = self.model(inp)  # 用当前输入预测下一时刻
    

                    # 真实目标是 batch 中对应 step 时刻的数据
                    pdb.set_trace()
                    target = batch[:, ind, :, :].float().to(self.device)

                    # 计算预测误差（相对误差）
                    mse_error = self.compute_loss(prd, target, relative=True)

                    # 记录数据，方便后续保存或分析
                    sample_inputs.append(inp.cpu())        # 当前输入
                    sample_predictions.append(prd.cpu())   # 模型预测
                    sample_targets.append(target.cpu())    # 真实目标
                    sample_errors.append(mse_error.item()) # 误差标量

                    # 下一次循环输入为当前预测结果，实现递推
                    inp = prd

                # 把当前样本的多个时间步数据堆叠起来，形成形状 [1, T, C, H, W]
                # pdb.set_trace()
                all_inputs.append(torch.stack(sample_inputs, dim=1))
                all_predictions.append(torch.stack(sample_predictions, dim=1))
                all_targets.append(torch.stack(sample_targets, dim=1))
                mse_error_test_all.append(sample_errors)
        pdb.set_trace()
        # 拼接所有样本数据，形成 [B, T, C, H, W]
        inputs_tensor = torch.cat(all_inputs, dim=0)
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        # 保存预测结果到 .pt 文件，方便后续加载分析
        torch.save({
            "inputs": inputs_tensor,
            "predictions": predictions_tensor,
            "targets": targets_tensor,
        }, os.path.join(self.save_path, "infer_results.pt"))

            # 保存每个时间步的误差
        csv_path = os.path.join(self.save_path, "infer_mse_error.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['sample_idx'] + [f't{t}' for t in steps]  # 物理时间步从 t 开始
            writer.writerow(header)
            for idx, sample_errors in enumerate(mse_error_test_all):
                writer.writerow([idx] + sample_errors)

        # # 保存平均误差
        # csv_summary_path = os.path.join(self.save_path, "infer_mse_mean.csv")
        # with open(csv_summary_path, mode='w', newline='') as f:
        #     writer = csv.writer(f)
        #     header = ['sample_idx'] + [f't{t}' for t in steps] 
        #     writer.writerow(header)

        #     # mse_array = np.array(mse_error_test_all)  # shape = (N, T)
        #     # mean_per_timestep = mse_array.mean(axis=0)
        #     # overall_mean = mean_per_timestep.mean()
        #     # overall_std = mean_per_timestep.std()

        #     # writer.writerow([0] + mean_per_timestep.tolist() + [overall_mean, overall_std])


        # 保存平均误差
        csv_summary_path = os.path.join(self.save_path, "infer_mse_mean.csv")
        with open(csv_summary_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # 修改后的表头：添加单独的t{self.idx}列
            header = ['sample_idx'] + [f't{self.idx}'] + [f't{t}' for t in steps if t != self.idx]
            writer.writerow(header)
            
            # 计算各时间步平均误差（假设mse_error_test_all已计算）
            mean_errors = np.array(mse_error_test_all).mean(axis=0)
            
            # 构建数据行：test_loss在前，其他时间步在后
            data_row = [0, test_loss]  # sample_idx和test_loss
            for i, t in enumerate(steps):
                if t != self.idx:  # 跳过已处理的self.idx
                    data_row.append(mean_errors[i])
            
            writer.writerow(data_row)


    # def test_inference_all(self):
    #     """
    #     Compute test loss (single-step) and multi-step inference loss.
    #     """
    #     model_path = os.path.join(self.save_path, 'model_best_weight.pth')
    #     checkpoint = torch.load(model_path, weights_only=False)
    #     self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #     self.model.eval()

    #     all_inputs = []
    #     all_predictions = []
    #     all_targets = []
    #     mse_error_test_all = []  # per sample per step
    #     mse_test_single = []     # single-step t1 loss

    #     steps = self.all_steps  # all time steps

    #     with torch.no_grad():
    #         for i, batch in enumerate(self.test_loader):
    #             inp = batch[:, 0, :, :, :].to(self.device)  # initial input t0

    #             sample_inputs = []
    #             sample_predictions = []
    #             sample_targets = []
    #             sample_errors = []

    #             # multi-step prediction
    #             for step_idx in steps:
    #                 target = batch[:, step_idx, :, :, :].float().to(self.device)
    #                 prd = self.model(inp)

    #                 # compute relative loss
    #                 mse_error = self.compute_loss(prd, target, relative=True)
    #                 sample_errors.append(mse_error.item())

    #                 sample_inputs.append(inp.cpu())
    #                 sample_predictions.append(prd.cpu())
    #                 sample_targets.append(target.cpu())

    #                 # next step input is prediction
    #                 inp = prd

    #             all_inputs.append(torch.stack(sample_inputs, dim=1))
    #             all_predictions.append(torch.stack(sample_predictions, dim=1))
    #             all_targets.append(torch.stack(sample_targets, dim=1))
    #             mse_error_test_all.append(sample_errors)

    #             # t1 single-step loss (for comparison with previous test)
    #             mse_test_single.append(sample_errors[0])

    #     # Convert to tensors
    #     inputs_tensor = torch.cat(all_inputs, dim=0)
    #     predictions_tensor = torch.cat(all_predictions, dim=0)
    #     targets_tensor = torch.cat(all_targets, dim=0)

    #     # Save .pt results
    #     torch.save({
    #         "inputs": inputs_tensor,
    #         "predictions": predictions_tensor,
    #         "targets": targets_tensor,
    #     }, os.path.join(self.save_path, "infer_results.pt"))

    #     # Save CSV: each sample per row, steps as columns
    #     csv_path = os.path.join(self.save_path, "infer_mse_error.csv")
    #     with open(csv_path, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         header = ['sample_idx'] + [f't{t}' for t in steps]
    #         writer.writerow(header)
    #         for idx, sample_errors in enumerate(mse_error_test_all):
    #             writer.writerow([idx] + sample_errors)

    #     # Save mean per timestep
    #     mean_errors = np.array(mse_error_test_all).mean(axis=0)
    #     csv_summary_path = os.path.join(self.save_path, "infer_mse_mean.csv")
    #     with open(csv_summary_path, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         header = ['sample_idx'] + [f't{t}' for t in steps]
    #         writer.writerow(header)
    #         writer.writerow([0] + mean_errors.tolist())

    #     return mse_test_single, mse_error_test_all, inputs_tensor, predictions_tensor, targets_tensor



            