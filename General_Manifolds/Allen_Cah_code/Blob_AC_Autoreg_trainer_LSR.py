

import time
import torch
import numpy as np
import pandas as pd
import csv
import os
import pdb
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data

class Autoreg_Trainer_LSR:
    def __init__(self, model, train_loader, val_loader, test_loader, nepochs, optimizer, criterion,
                 scheduler=None, save_path="SOS_res", all_steps=None, forward_step=1,
                 device='cuda',):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.nepochs = nepochs
        self.scheduler = scheduler
        self.save_path = save_path

        self.lowest_loss = 100.0
        self.all_steps = all_steps
        self.forward_step = forward_step
        self.device = device

        os.makedirs(self.save_path, exist_ok=True)


    def compute_loss(self, prd, tar, sqrt=False):
        """
        计算损失（支持相对损失和平方根损失）。
        :param prd: 预测值，形状 [num_points, 1]
        :param tar: 目标值，形状 [num_points, 1]
        :param sqrt: 是否对损失取平方根（仅在相对损失时有效）
        :return: 标量损失（PyTorch 张量）
        """
        # 计算原始损失（假设 self.criterion 是标量损失函数，如 MSE）
        loss = self.criterion(prd, tar)
        
        # 计算相对损失的分母（避免 tar 全为 0 导致分母过小）
        tar_norm = self.criterion(tar, torch.zeros_like(tar)) + 1e-12  # 用 1e-12 增强稳定性
        
        # 计算相对损失
        relative_loss = loss / tar_norm
        
        # 如果需要，对相对损失取平方根（保持张量类型，不转换为 NumPy）
        if sqrt:
            relative_loss = torch.sqrt(relative_loss)
        
        return relative_loss

    def train(self):
        train_logs = []
        train_start = time.time()

        for epoch in range(self.nepochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            is_best = val_loss < self.lowest_loss
            if is_best:
                self.lowest_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(self.save_path, 'model_best_weight.pth'))
                print(f'==> Saved new best model with lowest loss {self.lowest_loss:.6f}')

            train_logs.append({
                "Epoch": epoch,
                "Training Loss": train_loss,
                "Validation Loss": val_loss
            })

            print(f'Epoch {epoch} summary:')
            print(f'  Time taken: {time.time() - epoch_start:.2f}s')
            print(f'  Training Loss: {train_loss:.6f}')
            print(f'  Validation Loss: {val_loss:.6f}')

            df = pd.DataFrame(train_logs)
            df.to_csv(os.path.join(self.save_path, 'training_loss.csv'), index=False)

        train_time = time.time() - train_start
        print(f'Training completed in {train_time:.2f} seconds.')
        return self.lowest_loss, train_time

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0


        if self.all_steps is None:
            raise ValueError("all_steps must be provided to the trainer for multi-step training.")

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Training Epoch {epoch}')):
            self.optimizer.zero_grad()
            batch_loss = 0
     
            pdb.set_trace()
            batch = batch.to(self.device)

        
            current_input = batch.x[:, 0].unsqueeze(1) # : [num_points_in_batch, 1]


            steps_to_train = self.all_steps[:self.forward_step]
            for step_idx in steps_to_train:
                
                pdb.set_trace()
                target = batch.x[:, step_idx].unsqueeze(1) # : [num_points_in_batch, 1]

                
                prd = self.model(current_input, batch.pos, batch.batch)

           
                loss = self.compute_loss(prd, target)
                batch_loss += loss

               
                current_input = target
             
                # current_input = prd

      
            batch_loss.backward()
            self.optimizer.step()

    
            total_loss += batch_loss.item()

       
        return total_loss / len(self.train_loader)

    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0

        if self.all_steps is None:
            raise ValueError("all_steps must be provided to the trainer for multi-step validation.")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validation Epoch {epoch}'):
                batch_loss = 0
                batch = batch.to(self.device)

                current_input = batch.x[:, 0].unsqueeze(1)

                steps_to_val = self.all_steps[:self.forward_step]
                for step_idx in steps_to_val:
                    target = batch.x[:, step_idx].unsqueeze(1)
                    prd = self.model(current_input, batch.pos, batch.batch)
                    loss = self.compute_loss(prd, target)
                    batch_loss += loss
                    current_input = target #
                    # current_input = prd #

                total_loss += batch_loss.item()

        return total_loss / len(self.val_loader)




    def test(self):
        """在测试集上进行单步预测并评估"""
        os.makedirs(self.save_path, exist_ok=True)
        self.model.eval()
        total_loss = 0
        all_predictions, all_targets, all_inputs, all_positions = [], [], [], []

        print("\nStarting single-step test evaluation...")
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc='Testing'):
                data = data.to(self.device)
                pdb.set_trace()

          
                if not self.all_steps:
                    raise ValueError("all_steps must be provided.")
                target_step = self.all_steps[0]

                input_t0 = data.x[:, 0].unsqueeze(1)
                target_t1 = data.x[:, target_step].unsqueeze(1)

                prd_t1 = self.model(input_t0, data.pos, data.batch)
                loss = self.compute_loss(prd_t1, target_t1)
                total_loss += loss.item()

   
                all_inputs.append(input_t0.cpu())
                all_predictions.append(prd_t1.cpu())
                all_targets.append(target_t1.cpu())
                all_positions.append(data.pos.cpu())

        avg_test_loss = total_loss / len(self.test_loader)
        print(f"Single-step Test Loss (t=0->t={target_step}): {avg_test_loss:.6f}")

        predictions_tensor = torch.stack(all_predictions)  # [N, num_points, 1]
        targets_tensor = torch.stack(all_targets)          # [N, num_points, 1]
        inputs_tensor = torch.stack(all_inputs)            # [N, num_points, 1]
        positions_tensor = torch.stack(all_positions)


        torch.save({
            "test_loss": avg_test_loss,
            "predictions": predictions_tensor,  #redictions
            "targets": targets_tensor,          # targets
            "inputs": inputs_tensor,            # inputs
            "positions": positions_tensor,
        }, os.path.join(self.save_path, "test_results.pt"))  # 

        print(f"[*] Test results saved to '{os.path.join(self.save_path, 'test_results.pt')}'")

        return predictions_tensor, avg_test_loss, targets_tensor, inputs_tensor, positions_tensor


    def test_inference(self, initial_predictions_t1,test_loss=None):
        """
        基于单步测试结果执行多步自回归推理。
        """
        print("\n" + "="*60)
        print(f"Starting Multi-Step Inference for steps: {self.all_steps[1:]}...")
        print(f"Initial input: test set predictions at t={self.all_steps[0]}.")
        print("="*60)

        model_path = os.path.join(self.save_path, 'model_best_weight.pth')
        if not os.path.exists(model_path):
            print(f"[!] Error: Best model weights not found at '{model_path}'.")
            return None

       

        checkpoint = torch.load(model_path, weights_only=False)
        state_dict = checkpoint['model_state_dict']

   
        if '_metadata' in state_dict:
            del state_dict['_metadata']

  
        self.model.load_state_dict(state_dict, strict=True)

     

        self.model.eval()
        print(f"[*] Loaded best model from epoch {checkpoint['epoch']}.")

        num_test_samples = initial_predictions_t1.shape[0]
        if num_test_samples == 0:
            print("[!] Error: No initial predictions provided.")
            return None

     
        multi_step_predictions = []
        multi_step_targets = []
        multi_step_inputs = []  
        multi_step_errors = []


        inference_steps = self.all_steps[1:]
        if not inference_steps:
            print("[!] Warning: No inference steps specified (all_steps[1:] is empty).")
            return None

        print(f"[*] Performing inference on {num_test_samples} test samples...")

    
        test_data_iter = iter(self.test_loader)

        with torch.no_grad():
            for sample_idx in tqdm(range(num_test_samples), desc="Processing Samples"):
               
                current_u = initial_predictions_t1[sample_idx].to(self.device)  # [num_points, 1]

             
                try:
                    data = next(test_data_iter)
                except StopIteration:
                    print(f"[!] Error: test_loader has fewer samples than initial_predictions_t1.")
                    break
                data = data.to(self.device)

                predictions_for_sample = []
                targets_for_sample = []
                inputs_for_sample = [] 
                errors_for_sample = []

          
                inputs_for_sample.append(current_u.cpu())

      
                for step_idx in inference_steps:
            
                    temp_batch = torch.zeros(current_u.shape[0], dtype=torch.long).to(self.device)

       
                    next_u_pred = self.model(current_u, data.pos, temp_batch)

                
                    true_target = data.x[:, step_idx].unsqueeze(1)

                
                    error = self.compute_loss(next_u_pred, true_target).item()

               
                    predictions_for_sample.append(next_u_pred.cpu())
                    targets_for_sample.append(true_target.cpu())
                    errors_for_sample.append(error)

            
                    if step_idx != inference_steps[-1]:
                        inputs_for_sample.append(next_u_pred.cpu())

                
                    current_u = next_u_pred

      
                multi_step_predictions.append(torch.stack(predictions_for_sample))
                multi_step_targets.append(torch.stack(targets_for_sample))
                multi_step_inputs.append(torch.stack(inputs_for_sample))  # 
                multi_step_errors.append(errors_for_sample)

  
        predictions_tensor = torch.stack(multi_step_predictions)  # [S, T, N, 1]
        targets_tensor = torch.stack(multi_step_targets)          # [S, T, N, 1]
        inputs_tensor = torch.stack(multi_step_inputs)            # [S, T, N, 1]（
        errors_array = np.array(multi_step_errors)

 
        infer_results_save_path = os.path.join(self.save_path, "infer_results.pt")
        torch.save({
            "inputs": inputs_tensor,                 # [S, T, N, 1]
            "predictions": predictions_tensor,       # [S, T, N, 1]
            "targets": targets_tensor,               # [S, T, N, 1]
            "inference_steps": inference_steps,
            "step_errors": errors_array,
        }, infer_results_save_path)
        print(f"[*] Inference results saved to '{infer_results_save_path}'.")

       
        csv_path = os.path.join(self.save_path, "inference_errors.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ["sample_idx"] + [f't{step}' for step in inference_steps]
            writer.writerow(header)
            for idx, sample_errors in enumerate(multi_step_errors):
                writer.writerow([idx] + sample_errors)

     

        csv_summary_path = os.path.join(self.save_path, "inference_mse_mean.csv")
        with open(csv_summary_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            mean_errors = np.mean(errors_array, axis=0)
            
          
            header = ["time_step", "average_mse"]
            writer.writerow(header)
            
         
            writer.writerow([self.all_steps[0], test_loss])
            

            for step, err in zip(inference_steps, mean_errors):
                writer.writerow([step, err])

        print(f"[*] Error summary saved to '{csv_path}' and '{csv_summary_path}'.")

        print("\n" + "="*60)
        print("Multi-Step Inference Completed.")
        print("="*60)

        return {
            "predictions": predictions_tensor,
            "targets": targets_tensor,
            "inputs": inputs_tensor,
            "step_errors": errors_array
        }