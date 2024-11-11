# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:11:37 2023

@author: PriyaPrabhakar
"""

import os
# from dataloader import*
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from unet3d import*
from train_3d_patch import*
import torchio as tio
from pathlib import Path
import statistics
import numpy as np
from unet_r import *
from monai.losses import FocalLoss, DiceLoss, DiceCELoss, TverskyLoss, DiceFocalLoss

train_images_dir = Path(r"D:\laptop_stuff\c_drive\data\images")
train_labels_dir = Path(r"D:\laptop_stuff\c_drive\data\mask")

#%%
image_paths = sorted(train_images_dir.glob('*.nii.gz'))
label_paths = sorted(train_labels_dir.glob('*.nii.gz'))
assert len(image_paths) == len(label_paths)
validation_split = 0.8
subjects = []
im_spacings = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        image=tio.ScalarImage(image_path),
        mask=tio.LabelMap(label_path),
        )
    im = tio.ScalarImage(image_path)
    im_spacings.append(im.spacing)
    
    subjects.append(subject)
im_spacings_ar = np.asarray(im_spacings)
median_spacing = (get_median(im_spacings_ar[:,0]), get_median(im_spacings_ar[:,1]), get_median(im_spacings_ar[:,2]))    
# subjects = crop_im(subjects, median_spacing, threshold1, threshold2)
# subjects = subjects+subjects
dataset = tio.SubjectsDataset(subjects)

print('Dataset size:', len(dataset), 'subjects')

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(median_spacing),
   
   
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(median_spacing), 
    
    
])

num_subjects = len(dataset)
num_training_subjects = int(validation_split * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)



print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')

training_batch_size = 2
validation_batch_size = 2

patch_size = 128
samples_per_volume = 4
max_queue_length = 20
n_workers = 0
# sampler = tio.data.UniformSampler(patch_size)
probabilities = {0: 0.35, 1: 0.65}
sampler = tio.data.LabelSampler(
    patch_size=patch_size,
    label_name='mask',
    label_probabilities=probabilities,
)


patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=0,
    shuffle_subjects=True,
    shuffle_patches=True,
    
)

# patches_training_set = sampler(training_set)
#
patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=0,
    shuffle_subjects=False,
    shuffle_patches=False,
)

train_data_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size, pin_memory=False)

validation_data_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size, pin_memory=False)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(device)
# model = UneTR(img_dim=(128, 128, 128),
#             in_channels=1,
#             base_filter=16,
#             class_num=1,
#             patch_size=16,
#             embedding_dim=768,
#             block_num=12,
#             head_num=12,
#             mlp_dim=3072,
#             z_idx_list=[3, 6, 9, 12])
model = model.float()

save_folder = r'C:\Users\PriyaPrabhakar\Downloads\Aorta_segmentation-new\New folder'
train(train_data_loader, validation_data_loader, num_epochs=500, loss_type=nn.BCEWithLogitsLoss(), model=model, device=device, save_folder=save_folder, threshold=0.5, learning_rate=0.0001) 
