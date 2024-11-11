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


def skeleton( f):
    """
    Extracts the skeleton from a 3D image and analyzes it to identify end points.
    
    This function loads a 3D image file, preprocesses it, and then extracts its skeleton.
    It then analyzes the skeleton to identify nodes, which are points where branches end.
    The identified nodes are returned as a list of coordinates.

    Parameters:
    - f (str): The file path of the 3D image file.

    Returns:
    - list: A list of coordinates representing the end points of the skeleton.

    
    """
    img = nib.load(f)
    
    # Access the image data
    data = img.get_fdata()
    
    data = ndimage.gaussian_filter(data, sigma=0.1)
    kernel2 = np.ones((3, 3, 3))
    
    # # data =  ndimage.morphology.binary_fill_holes(data)
    data = morphology.binary_opening(data, kernel2)
    # data =  ndimage.morphology.binary_fill_holes(data, kernel2)
    # data = morphology.area_opening(data,8)
    
    skeleton = morphology.skeletonize(data)
    kernel = np.ones((3, 3, 3))
    # Find connected components in the skeleton
    
    viewer = napari.view_image(data, blending='additive', colormap='green', name='nuclei')
    viewer.add_image(skeleton, blending='additive', colormap='magenta', name='edges')
    
    neighborhood_sum = convolve(skeleton, kernel, mode='constant', cval=0)
    b=np.multiply(skeleton, neighborhood_sum)
    # print(np.unique(b))
    a=np.argwhere(b==2)
    # print(a)
    # print(skeleton[219,196,183])
    # print(neighborhood_sum[219,196,183])
    
    graph = sknw.build_sknw(skeleton)
    
    nodes = graph.nodes()
    edges = graph.edges()
    edges_ar = np.asarray(edges)
    sz = edges_ar.shape
    dis_dic = {}
    dis_v = {}
    short_dis = []
    for i in range(0, sz[0]):
        v_a = edges_ar[i,0] 
        v_b = edges_ar[i,1]
        coord_a = nodes[v_a]['o']
        coord_b = nodes[v_b]['o']
        coord_a = coord_a.astype(np.int16)
        coord_b = coord_b.astype(np.int16)
        dis = np.linalg.norm(coord_a-coord_b)
        
        if v_a not in dis_dic:
            list_a = []
            list_a_v = []
        else:
            list_a = dis_dic[v_a]
            list_a_v = dis_v[v_a]
            
        if v_b not in dis_dic:
            list_b = []
            list_b_v = []
        else:
            list_b = dis_dic[v_b]
            list_b_v = dis_v[v_b]

        if dis < 12:
            pts_a = tuple(nodes[v_a]['o'].tolist())
            pts_b = tuple(nodes[v_b]['o'].tolist())
            # print(pts_a)
            # print(pts_b)
            if pts_a not in short_dis:
                short_dis.append(pts_a)
            if pts_b not in short_dis:
                short_dis.append(pts_b)
    
        list_a.append(dis)
        list_b.append(dis)
        list_a_v.append(v_b)
        list_b_v.append(v_a)
        dis_dic[v_a] = list_a
        dis_dic[v_b] = list_b
        dis_v[v_a] = list_a_v
        dis_v[v_b] = list_b_v
 
    nodes_conn_1 = []
    for i in range(edges_ar.shape[0]+1):
        if np.count_nonzero(edges_ar == i)==1:
            nodes_conn_1.append(tuple(nodes[i]['o'].tolist()))
    nodes_final = []    
    for i in range(len(nodes_conn_1)):
        if nodes_conn_1[i] not in short_dis:
            nodes_final.append(nodes_conn_1[i])
            
    return nodes_final
            
