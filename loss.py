# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:37:33 2023

@author: PriyaPrabhakar
"""

"dice loss"

import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
# import tensorflow.keras.backend as K 
import torchio as tio
from unet3d import*
from scipy.ndimage import zoom
import nibabel as nib
from monai.losses import FocalLoss, DiceLoss


def dice_coef(y_true, y_pred):
    threshold = 0.5
    y_pred = nn.functional.sigmoid(y_pred)
    pred = (y_pred >threshold)
    smooth = 1
    intersection = torch.sum(y_true * pred)
    union = torch.sum(y_true) + torch.sum(pred)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

# def dice_coef(y_true, y_pred):
#     y_truef=K.flatten(y_true)
#     y_predf=K.flatten(y_pred)
#     And=K.sum(y_truef* y_predf)
#     return((2* And + 100) / (K.sum(y_truef) + K.sum(y_predf) + 100))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return jac    

# class DiceLoss_func(nn.Module):

#     def __init__(self):
#         super(DiceLoss_func, self).__init__()
#         self.smooth = 1.0

#     def forward(self, y_pred, y_true):
#         loss_val = DiceLoss(reduction='none', sigmoid=True)
#         loss = loss_val (y_pred, y_true)
#         return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print(inputs)
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs_s = nn.functional.sigmoid(inputs)   
  
        # print(inputs_s)
        # print(np.unique(inputs_s.cpu().detach().numpy()))
        #flatten label and prediction tensors
        inputs_f = inputs_s.view(-1)
        targets_f = targets.view(-1)
        # print(targets)
        intersection = (inputs_f * targets_f).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_f.sum() + targets_f.sum() + smooth)  
        # print(dice_loss)
        # BCE = nn.functional.binary_cross_entropy(inputs_f.float(), targets_f.float(), reduction='none')
        # print(inputs.shape)
        # print(targets.shape)
        BCE = nn.BCEWithLogitsLoss()(inputs, targets)

        # BCE = nn.BCELoss(reduction='none')(inputs.float(), targets.float()).mean()
        # print(BCE)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
        # return BCE
    
class iou_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(iou_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
                
        return 1 - iou
    

class FocalLoss_f(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss_f, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky