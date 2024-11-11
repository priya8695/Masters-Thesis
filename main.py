# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:16:36 2024

@author: PriyaPrabhakar
"""

from inference import*
from ga_main import*


# Load configuration from the YAML file
yaml_file_path = os.path.join(os.getcwd(), 'config.yaml')
config = load_config(yaml_file_path)

# Check the selected model type from the configuration
if config['model'] == 'nnunet':
    # Perform inference using nnUNet
    nnunet_inference(config)
else: 
    # Perform inference using 3D U-Net
    unet_inference(config)
    
# Perform geometric analysis on the output masks
geometric_analysis(config['output_folder_segmentation'], config['output_folder_ga'])