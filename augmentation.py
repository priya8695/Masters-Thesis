# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:14:45 2023

@author: PriyaPrabhakar
"""
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchio as tio
from pathlib import Path
import statistics
import numpy as np
# import albumentations as A
# import albumentations.augmentations.functional as F
from dataloader_augmentation import*
import nibabel as nib
import os
# import cv2
from utilities import*
# train_images_dir = Path(r"/data/s3292983/new_folder/Aorta_segmentation-main/src/data/images")
# train_labels_dir = Path(r"/data/s3292983/new_folder/Aorta_segmentation-main/src/data/mask")
# train_dir = Path(r'/data/s3292983/new_folder/Aorta_segmentation-main/src/data/')
# test_dir = Path(r'/data/s3292983/new_folder/Aorta_segmentation-main/src/data/')
train_images_dir = Path(r"/media/scratch/nnu2_priya/full_data/dataset/images")
train_labels_dir = Path(r"/media/scratch/nnu2_priya/full_data/dataset/mask")
train_dir = Path(r'/media/scratch/nnu2_priya/full_data/dataset/')
aug_dir_img = Path(r'/media/scratch/nnu2_priya/full_data/augmented_data/images')
aug_dir_mask = Path(r'/media/scratch/nnu2_priya/full_data/augmented_data/mask')                  
image_filenames = os.listdir(train_images_dir)

image_filenames = 7*image_filenames
image_paths = sorted(train_images_dir.glob('*.nii.gz'))
label_paths = sorted(train_labels_dir.glob('*.nii.gz'))
assert len(image_paths) == len(label_paths)

subjects = []
im_spacings = []

for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        image=tio.ScalarImage(image_path),
        mask=tio.LabelMap(label_path),
        )
    if subject.image.shape==subject.mask.shape:
        im = tio.ScalarImage(image_path)
        im_spacings.append(im.spacing)
        subjects.append(subject)
im_spacings_ar = np.asarray(im_spacings)
median_spacing = (get_median(im_spacings_ar[:,0]), get_median(im_spacings_ar[:,1]), get_median(im_spacings_ar[:,2]))  
# median_spacing = (0.8, 0.8, 1)  

# train_transform = A.Compose(
#     [
     
#           A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
#           A.VerticalFlip(p=0.3),
#          A.Blur(p=0.2),
#          A.GaussNoise(p=0.2),
#          A.ElasticTransform (alpha=1, sigma=20, alpha_affine=15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.2),
#          A.Affine (scale=None, translate_percent=0.01, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, rotate_method='largest_box', always_apply=False, p=0.2)
#     ]
# )
max_value = 800
min_value = -1000
max_displacement = 15, 10, 0  # in x, y and z directions

# preprocessing = tio.Compose([tio.ToCanonical(),
#                          tio.Clamp(out_min=min_value, out_max=max_value),
#                          tio.Resample(median_spacing),
#                          tio.ZNormalization()])
preprocessing = tio.Compose([tio.ToCanonical(),
                         tio.Clamp(out_min=min_value, out_max=max_value),
                                                 tio.ZNormalization()])

transform = [
    tio.RandomFlip(axes=('AP', 'IS'), p=0.5),
    tio.RandomElasticDeformation(max_displacement=max_displacement, p=0.3),
    tio.RandomAnisotropy(p=0.2),
    tio.RandomAffine(scales=(0.9, 1.2), degrees=10, p=0.6),
    tio.RandomBlur( 0.05, p=0.2),
    tio.RandomNoise(mean=0, std=0.05, p=0.2),
    tio.RandomGamma(log_gamma=(-0.3, 0.1), p=0.2)
]



train_transform = tio.OneOf(transform)
train_dataset = DataLoaderSegmentation(train_dir, image_filenames, preprocessing, train_transform)
train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

i=0
for images, mask, img_name in train_loader:
    
    i = i+1
    t = str(i)
    if len(t)==1:
        t="00"+t
    if len(t)==2:
        t="0"+t
   
    output_filename = "lung_"+t+".nii.gz"
    output_path_img = os.path.join(aug_dir_img, output_filename)
    output_path_mask = os.path.join(aug_dir_mask, output_filename)
    images = images['data']
    images = images.numpy().astype(np.float32)
    images = images.squeeze()
    images = images.squeeze()
    mask = mask['data']
    mask = mask.numpy().astype(np.uint16)
    mask = mask.squeeze()
    mask = mask.squeeze()
    img_raw = nib.load(img_name[0])
    images_n = nib.Nifti1Image(images, affine=img_raw.affine)
    mask_n = nib.Nifti1Image(mask, affine=img_raw.affine)
    nib.save(images_n, output_path_img)
    nib.save(mask_n, output_path_mask)
