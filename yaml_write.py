# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:55:55 2024

@author: PriyaPrabhakar
"""


import yaml

# Data to be written to the YAML file
data = {
    'nnUNet_raw' : r'E:\pc\nnunet_final\nnUNet_raw',
    'nnUNet_preprocessed' : r'E:\pc\nnunet_final\nnUNet_preprocessed',
    'nnUNet_results' : r'E:\pc\nnunet_final\nnUNet_results',
    'test_folder' : r'C:\Users\PriyaPrabhakar\Desktop\test_folder',
    'output_folder_segmentation' : r'C:\Users\PriyaPrabhakar\Desktop\output',
    'output_folder_ga' : r'C:\Users\PriyaPrabhakar\Desktop\output',    
    'model' : 'nnunet',
    'nnunet_dataset_id' : '06',
    'nnunet_configuration' : '3d_fullres',
    'nnunet_fold' : '2'
}

# Path to the YAML file
yaml_file_path = r'C:\Users\PriyaPrabhakar\Desktop\final_code\config.yaml'

# Writing data to the YAML file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"YAML file '{yaml_file_path}' has been created successfully.")


