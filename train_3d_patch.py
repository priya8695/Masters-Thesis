# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:41:56 2023

@author: PriyaPrabhakar
"""

# Import necessary libraries
import os
import numpy as np
import torch
import torch.optim as optim
from unet3d import*  
from loss import*
import torchio as tio
import gc
import torch.optim.lr_scheduler as lr_scheduler
import time
# from monai.losses import FocalLoss, DiceLoss

# Define the training function
def train(train_set, validation_set, num_epochs, loss_type, model, device, save_folder, threshold, learning_rate, use_gradscalar=False):
    """
    Trains a neural network model on the given training set and validates it on the validation set.

    Parameters:
    - train_set (torch.utils.data.Dataset): The training dataset.
    - validation_set (torch.utils.data.Dataset): The validation dataset.
    - num_epochs (int): The number of training epochs.
    - loss_type (torch.nn.Module): The loss function to be used for training.
    - model (torch.nn.Module): The neural network model to be trained.
    - device (str): The device to be used for training ('cpu' or 'gpu').
    - save_folder (str): The folder path to save the trained model and metrics.
    - threshold (float): Threshold value for prediction binarization.
    - learning_rate (float): The learning rate for optimization.
    - use_gradscalar (bool): Whether to use gradient scaling during training.

    Returns:
    - None

    This function trains a neural network model using the specified training set, loss function, and optimization algorithm.
    It then evaluates the trained model on the validation set and saves the model and training/validation metrics.
    """
    model.to(device)
    model.train()
    criterion = loss_type
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    # Lists to store training and validation metrics
    train_epoch_loss = []
    val_epoch_acc = []
    val_epoch_loss = []
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        train_loss = []
        dice_score = []
        val_loss = []
        print('epoch:' + str(epoch))
        t0 = time.time()
        # Training loop
        for patches_batch in train_set:
            # print('\patch loading Time: {:.4f}s, '.format(time.time()-t0))
            optimizer.zero_grad()
            mask_patch = patches_batch['mask'][tio.DATA].to(device)
            mask_patch = mask_patch.to(device)
            output = model(patches_batch['image'][tio.DATA].to(device))
            loss = criterion(output, mask_patch.float())

            if use_gradscalar:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                loss.backward()
                optimizer.step()

            train_loss.append(loss.mean().item())
            del output
            del mask_patch
            del loss
            torch.cuda.empty_cache()
            gc.collect()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_folder, "model.pth"))

        
        print('\nTraining loss: {:.4f}, '.format(np.mean(train_loss)))
        # Validation loop
        model.eval()
        with torch.no_grad():
            for val_patches in validation_set:
                val_image = val_patches['image'][tio.DATA].to(device)
                val_mask = val_patches['mask'][tio.DATA].to(device)
                mask_pred = model(val_image)
                criterion = loss_type
                val_mask = val_mask.to(device)
                val_loss.append(criterion(mask_pred, val_mask.float()).mean().item())
                
                dice_score.append((dice_coef(val_mask.double(), mask_pred.double())).item())

        print('\nValidation set: dice score: {:.4f}, '.format(np.mean(dice_score)))
        print('\nValidation set: loss: {:.4f}, '.format(np.mean(val_loss)))
        print('\nTime: {:.4f}s, '.format(time.time()-t0))
        train_epoch_loss.append(np.mean(train_loss))
        val_epoch_acc.append(np.mean(dice_score))
        val_epoch_loss.append(np.mean(val_loss))

        np.save(os.path.join(save_folder, "train_loss.npy"), train_epoch_loss)
        np.save(os.path.join(save_folder, "validation_accuracy.npy"), val_epoch_acc)
        np.save(os.path.join(save_folder, "val_loss.npy"), val_epoch_loss)
