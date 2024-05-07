#!/usr/bin/env python
# coding: utf-8

"""
Command line script for training the DQFMNet on Mouse Mandible Dataset.
Usage:
    python train_dqfmnet.py --config config.yaml
"""

import os
import torch
import time
import numpy as np
import yaml
import json
from tqdm import tqdm
import argparse
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from datasets import MouseMandibleDataset
from model import DQFMNet
from utils import DQFMLoss, shape_to_device, augment_batch_sym
from Tools import fMap2pMap


# Helper function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train DQFMNet on Mouse Mandible Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


# Helper function to load configurations
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = euclidean_dist(x, y)
    output = F.softmax(-alpha * distance, dim=-1)
    # _, idx = distance.topk(k=k, dim=-1)
    return output


def convert_C(Phi1, Phi2, A1, A2, alpha):
    Phi1, Phi2 = Phi1[:, :, :100], Phi2[:, :, :100]
    D1 = torch.bmm(Phi1, A1)
    D2 = torch.bmm(Phi2, A2)
    T12 = knnsearch(D1, D2, alpha)
    T21 = knnsearch(D2, D1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))
    C21_new = torch.bmm(torch.pinverse(Phi1), torch.bmm(T12, Phi2))

    return C12_new, C21_new


def run_inference(dqfm_net, data_loader, criterion, device, alpha_i,
                  jsonl_file_path='batch_records_points_model_30A_on_test_data.jsonl'):
    """
    Run model inference on a dataset, compute losses, and write outputs to a JSON Lines file.

    Parameters:
    - dqfm_net (torch.nn.Module): The trained model to use for inference.
    - data_loader (DataLoader): DataLoader providing the dataset for inference.
    - criterion (function): Loss function used to evaluate model performance.
    - device (torch.device): Device to run the model computation on.
    - alpha_i (float): Alpha value used in the conversion function.
    - jsonl_file_path (str): Path to the JSON Lines file to write output data.
    """
    # Set the model to evaluation mode
    dqfm_net.eval()

    # Clear GPU cache if necessary
    torch.cuda.empty_cache()

    # Open the JSON Lines file for writing
    with open(jsonl_file_path, 'w') as file:
        # Disable gradient computation to save memory and computations
        with torch.no_grad():
            for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing batches"):
                data = shape_to_device(data, device)  # Move data to the correct device

                # Get model outputs without computing gradients
                outputs = dqfm_net(data)
                C12_gt, C21_gt = data["C12_gt"], data["C21_gt"]
                C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2 = outputs
                A1 = torch.bmm(evecs_trans1, feat1)
                A2 = torch.bmm(evecs_trans2, feat2)
                C12_pred_new, C21_pred_new = convert_C(evecs1, evecs2, A1, A2, alpha_i)

                loss, loss_gt_old, loss_gt, loss_ortho, loss_bij, loss_res, loss_rank = criterion(C12_gt, C21_gt,
                                                                                                  C12_pred.to(device),
                                                                                                  C21_pred.to(device),
                                                                                                  C12_pred_new.to(
                                                                                                      device),
                                                                                                  C21_pred_new.to(
                                                                                                      device),
                                                                                                  Q_pred.to(device),
                                                                                                  feat1, feat2,
                                                                                                  evecs_trans1,
                                                                                                  evecs_trans2)

                # Print batch progress and loss
                print(
                    f"Loss: {loss.item():.4f}, loss_ortho: {loss_ortho.item():.4f}, loss_bij: {loss_bij.item():.4f}, loss_res: {loss_res.item():.4f}, loss_rank: {loss_rank.item():.4f}")

                for j in range(len(data['shape1']['name'])):
                    # Compute point-to-point maps
                    T12_new = fMap2pMap(evecs1[j].detach().cpu().numpy(),
                                        evecs2[j].detach().cpu().numpy(),
                                        C12_pred_new[j].detach().cpu().numpy())

                    # Prepare the dictionary with the necessary information
                    batch_dict = {
                        'T12_new': T12_new.tolist(),
                        'shape1_name': data['shape1']['name'][j],
                        'shape2_name': data['shape2']['name'][j]
                    }

                    # Write the current batch's dictionary as a JSON line
                    file.write(json.dumps(batch_dict) + '\n')


# Main training function
def train_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = MouseMandibleDataset(config['train_dataset_path'], k_eig=config['k_eig'], wks_eig=config['wks_eig'],
                                    n_fmap=config['n_fmap'], n_cfmap=config['n_cfmap'])
    validation_ds = MouseMandibleDataset(config['validation_dataset_path'], k_eig=config['k_eig'],
                                         wks_eig=config['wks_eig'], n_fmap=config['n_fmap'], n_cfmap=config['n_cfmap'])

    train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=config['training_batch_size'])
    validation_dl = DataLoader(dataset=validation_ds, shuffle=True, batch_size=config['validation_batch_size'])

    dqfm_net = DQFMNet(C_in=config['C_in'], n_feat=config['n_feat'], lambda_=config['lambda'],
                       C_width=config['C_width'], N_block=config['N_block'], mlp_hidden_dims=config['mlp_hidden_dims'],
                       resolvant_gamma=config['resolvant_gamma'], n_fmap=config['n_fmap'], n_cfmap=config['n_cfmap'],
                       robust=config['robust']).to(device)

    optimizer = torch.optim.Adam(dqfm_net.parameters(), lr=config['learning_rate'], betas=(0.9, 0.99))
    criterion = DQFMLoss(w_gt=config['w_gt'], w_ortho=config['w_ortho'], w_Qortho=config['w_Qortho'],
                         w_bij=config['w_bij'], w_res=config['w_res'], w_rank=config['w_rank']).to(device)
    scheduler = StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    current_epoch = 0
    total_epochs = config['total_epochs']
    alpha_list = np.linspace(config['start_alpha'], config['final_alpha'], total_epochs)
    save_interval = config['save_interval']

    writer = SummaryWriter()  # Initialize TensorBoard writer
    best_loss = float('inf')
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'runs/mandible_shape_matching_{date_time}'
    save_dir = f'runs/mandible_shape_matching_{date_time}/npz_saves'
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(current_epoch, total_epochs):
        dqfm_net.train()  # Ensure the network is in training mode
        total_batches = len(train_dl)
        epoch_start_time = time.time()

        # Adjust alpha based on the epoch
        alpha_i = alpha_list[min(epoch, len(alpha_list) - 1)]

        epoch_loss = 0.0
        for i, data in enumerate(train_dl):
            start_time = time.time()  # Start timing the batch processing

            # Preprocess and augment data
            data = shape_to_device(data, device)  # Assuming `shape_to_device` moves the data to the correct device
            data = augment_batch_sym(data, rand=True)

            # Forward pass
            C12_gt, C21_gt = data["C12_gt"], data["C21_gt"]
            C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2 = dqfm_net(data)

            # Process predictions
            A1 = torch.bmm(evecs_trans1, feat1)
            A2 = torch.bmm(evecs_trans2, feat2)
            C12_pred_new, C21_pred_new = convert_C(evecs1, evecs2, A1, A2, alpha_i)

            # Compute losses
            loss, loss_gt_old, loss_gt, loss_ortho, loss_bij, loss_res, loss_rank = criterion(C12_gt, C21_gt,
                                                                                              C12_pred.to(device),
                                                                                              C21_pred.to(device),
                                                                                              C12_pred_new.to(device),
                                                                                              C21_pred_new.to(device),
                                                                                              Q_pred.to(device), feat1,
                                                                                              feat2,
                                                                                              evecs_trans1,
                                                                                              evecs_trans2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time  # Calculate elapsed time for the batch
            epoch_loss += loss.item()

            # Log batch statistics to TensorBoard
            writer.add_scalar('Batch/Loss/total_loss', loss.item(), epoch * total_batches + i)
            writer.add_scalar('Batch/Time', elapsed_time, epoch * total_batches + i)

            if i % save_interval == 0:
                # Save .npz file every 100 iterations
                file_name = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.npz")
                np.savez(file_name, Q_pred=Q_pred[0].detach().cpu().numpy())

        # Epoch statistics
        epoch_duration = time.time() - epoch_start_time
        average_loss = epoch_loss / total_batches

        # Log epoch statistics to TensorBoard
        writer.add_scalar('Epoch/Average Loss', average_loss, epoch)
        writer.add_scalar('Epoch/Duration', epoch_duration, epoch)
        writer.add_scalar('Epoch/Learning Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Epoch/Alpha', alpha_i, epoch)

        # Update learning rate
        scheduler.step()

        # Check and save best model
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(dqfm_net.state_dict(), os.path.join(folder_name, f"best_model_epoch_{epoch + 1}.pt"))

        torch.cuda.empty_cache()

    writer.close()  # Close the TensorBoard writer

    # After training, run validation
    run_inference(dqfm_net, validation_dl, criterion, device, config['validation']['alpha_i'],
                  config['validation']['jsonl_file_path'])


# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_model(config)
