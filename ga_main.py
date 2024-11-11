# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:14:33 2024

@author: PriyaPrabhakar
"""
import numpy as np
import itk
import napari
import nibabel as nib
from vmtk_libraries import*
import math
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform)
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
import napari
import nibabel as nib
from scipy import ndimage
import sknw
import networkx as nx
import meshparty as msh
import trimesh
import vtk
from vmtk_libraries import*
import scipy
from ga_libraries import*
from skeleton_f import*

# test_folder = r'C:\Users\PriyaPrabhakar\Desktop\test_folder'
# output_folder = r'C:\Users\PriyaPrabhakar\Desktop\output'


def geometric_analysis(test_folder, output_folder_ga):
    """
    Performs geometric analysis on segmented aorta images and saves the computed parameters in text files.

    This function takes a folder containing segmented aorta images and computes various geometric parameters for each image.
    The computed parameters are saved in text files, with one file generated for each input image. The output text files 
    are saved in the specified output folder.

    Parameters
    ----------
    test_folder : str
        The folder location containing segmented aorta images for which geometric parameters will be computed.
    output_folder : str
        The folder location where the text files with computed geometric parameters will be saved.

    """

    # List all files in the test folder
    test_files = os.listdir(test_folder)    
    for test_file in test_files:
        if '.nii.gz' in test_file:
            # Generate the skeleton for the current file
            nodes_final = skeleton(os.path.join(test_folder, test_file))
            print(nodes_final)
            
            # Load the image data
            img = nib.load(os.path.join(test_folder, test_file))
            
            # Sort the nodes array based on z-coordinate
            nodes_array = np.asarray(nodes_final)
            nodes_array_sort = nodes_array[nodes_array[:, 2].argsort()]
            
            # Define surface name and output file path
            sur_name = test_file.replace('.nii.gz','.stl')
            file_name = os.path.join(output_folder, sur_name) 
            test_path = os.path.join(test_folder, test_file)  # Path to the image
            output_file = os.path.join(output_folder, test_file.replace('.nii.gz','.txt'))
            output_file_mapped = os.path.join(output_folder, test_file.replace('.nii.gz','.vtk'))
            # Access the image data
            data = img.get_fdata()
            
            # Convert image data to VTK format
            itk_image_view = itk.image_view_from_array(data)
            vtk_image = itk.vtk_image_from_image(itk_image_view)
            
            # Generate surface mesh using Marching Cubes algorithm
            surface_mc = vmtkmarchingcubes(vtk_image, level=1)
            d = vmtkmeshtosurface(surface_mc, cleanoutput=10)
            
            # Smooth the surface mesh
            surface = vmtksurfacesmoothing(d, iterations=500, method='laplace')
            
            # Ensure surface connectivity
            surface_f = surface_connectivity(surface)
            
            # Split points into zones based on main centerline
            zone_points, main_centerline = split_points_new(surface_f, nodes_final)
            
            # Prepare unique and sorted zone points
            tuple_list = [tuple(item) for item in zone_points]
            unique_list = [list(item) for item in set(tuple_list)]
            sorted_list = sorted(unique_list, key=lambda x: x[2] if isinstance(x, list) else x[2][2])
            
            # sort zone points
            new_list = sorted_list.copy()
            new_list[0] = sorted_list[1]
            new_list[1:len(new_list)-1] = sorted_list[2:len(new_list)]
            new_list[-1] = sorted_list[0]
            zone_points = new_list
            
            # Compute geometric parameters for arc
            point_a = zone_points[0]
            point_b = end_point_f(main_centerline)
            arc_dic = arc_ga(point_a, point_b, surface_f, file_name, nodes_array_sort)
            
            # Write arc geometric parameters to output file
            with open(output_file, 'w') as file:
                file.write('Geometric parameters of Arc\n')
                for index, (key, value) in enumerate(arc_dic.items(), start=1):
                    line = f"{index}: {key}: {value}\n"
                    file.write(line)
            
            # Edit zone points for further analysis with 20mm proximal and distal points
            zone_points_edit = zone_points.copy()
            zone_points_edit[0] = proximal(main_centerline, zone_points[1])
            zone_points_edit[-1] = distal(main_centerline, zone_points[-2], 20)
            
            # Compute geometric parameters for each zone
            zone_dic = zone_ga(zone_points_edit, surface_f, main_centerline)
            
            #PLZs mapping
            ga_zone_visualization(nodes_final, surface_f, main_centerline, output_file_mapped)
            
            # Write zone geometric parameters to output file
            with open(output_file, 'a') as file:
                for i in range(len(zone_dic)):
                    file.write('\n Geometric parameters of Zone'+str(i)+'\n')  
                    for index, (key, value) in enumerate(zone_dic[i].items(), start=1):
                        line = f"{index}: {key}: {value}\n"
                        file.write(line)
        
if __name__ == "__main__":
    geometric_analysis(test_folder, output_folder_ga)
