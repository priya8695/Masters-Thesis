# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:15:41 2024

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
import pyvista as pv
import networkx as nx
import meshparty as msh
import trimesh
import vtk
from vmtk_libraries import*
import scipy

#%%
def fit_circle_to_points(x_coords, y_coords, weights=[]):
    """
    Fit a circle to a set of 2D points using least squares optimization.

    Parameters
    ----------
    x_coords : array
        The x-coordinates of the 2D points.
    y_coords : array
        The y-coordinates of the 2D points. Should have the same length as `x_coords`.
    weights : array, optional
        Weight array for weighted least squares. If provided, it should have the same length as `x_coords` and `y_coords`.

    Returns
    -------
    center_x : float
        The x-coordinate of the center of the fitted circle.
    center_y : float
        The y-coordinate of the center of the fitted circle.
    radius : float
        The radius of the fitted circle.

    """
    
    coefficients_matrix = array([x_coords, y_coords, ones(len(x_coords))]).T
    b_vector = x_coords**2 + y_coords**2
    if len(weights) == len(x_coords):
        weights_matrix = diag(weights)
        coefficients_matrix = dot(weights_matrix, coefficients_matrix)
        b_vector = dot(weights_matrix, b_vector)
    solution_vector = linalg.lstsq(coefficients_matrix, b_vector, rcond=None)[0]
    center_x = solution_vector[0] / 2
    center_y = solution_vector[1] / 2
    radius = sqrt(solution_vector[2] + center_x**2 + center_y**2)
    return center_x, center_y, radius


def rodrigues_rotation(points, original_direction, desired_direction):
    """
    Perform Rodrigues' rotation on a set of 3D points.
    
    Parameters
    ----------
    points : array
        Array of 3D points to be rotated. If `points` is a 1D array (coordinates of a single point), 
        it will be converted to a matrix.
    original_direction : array
        Vector representing the original direction.
    desired_direction : array
        Vector representing the desired direction.
    
    Returns
    -------
    rotated_points : ndarray
        Array of rotated 3D points.
    

    """
    
    if points.ndim == 1:
        points = points[newaxis,:]
    original_direction = original_direction / linalg.norm(original_direction)
    desired_direction = desired_direction / linalg.norm(desired_direction)
    rotation_axis = cross(original_direction, desired_direction)
    rotation_axis = rotation_axis / linalg.norm(rotation_axis)
    theta = arccos(dot(original_direction, desired_direction))
    rotated_points = zeros((len(points), 3))
    for i in range(len(points)):
        rotated_points[i] = points[i] * cos(theta) + cross(rotation_axis, points[i]) * sin(theta) + rotation_axis * dot(rotation_axis, points[i]) * (1 - cos(theta))
    return rotated_points




def angle_between_vectors(vector1, vector2, normal_vector=None):
    """
    Compute the angle between two vectors or the angle between a plane and a vector.

    Parameters
    ----------
    vector1 : array
        First vector or normal vector of the plane.
    vector2 : array
        Second vector.
    normal_vector : array, optional
        Normal vector of the plane. If provided, computes the angle between the plane 
        defined by `normal_vector` and `vector1` and the vector `vector2`.

    Returns
    -------
    angle : float
        Angle between the vectors or the angle between the plane and the vector, in radians.

    """
    if normal_vector is None:
        return arctan2(linalg.norm(cross(vector1, vector2)), dot(vector1, vector2))
    else:
        return arctan2(dot(normal_vector, cross(vector1, vector2)), dot(vector1, vector2))


#%%    
def bifurication_point(centerline, centerline2):
    centerlines_branched = vmtk_compute_branch_extractor(centerline)
    branchclipper = vmtk_branch_clipper(centerlines_branched, surface_f, clip_value=0, inside_out=False, use_radius_information=True,
                            interactive=False)
    # surface_viewer(branchclipper.Surface, array_name="GroupIds")
    bi = vmtkbifurcationsections(branchclipper.Surface, branchclipper.Centerlines, distance=0)
    pts = bi.GetCellData()
    pts_array = pts.GetArray('BifurcationSectionPoint')
    pts_bif = np.array(pts_array)
    pts_bif_cen = closest_point(pts_bif[0], centerline2)
    return pts_bif_cen


# new_centerline_vmtk, new_centerline = vmtk_compute_centerlines(end_point, inlet, method, new_outlet, pole_ids, resampling_step, surface, voronoi,
#                              flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
#                              simplify_voronoi=False) 

def tube_display(surface_f, nodes_final):
    end_point = 0
    method = 'pointlist'
    resampling_step = 0.5
    voronoi = None
    pole_ids = None
    nodes_array = np.asarray(nodes_final)
    nodes_array_sort = nodes_array[nodes_array[:, 2].argsort()]
    inlet_ar = nodes_array_sort[1]
    inlet = inlet_ar.tolist()
    outlet_last_ar = nodes_array_sort[0]
    outlet_last = outlet_last_ar.tolist()
    # outlet_last = np.array([268, 232, 11, 231, 279, 202, 278, 276, 209, 289, 262, 216])
    # outlet_last = outlet_last.tolist()
    main_centerline_vmtk, main_centerline = vmtk_compute_centerlines(end_point, inlet, method, outlet_last, pole_ids, resampling_step, surface_f, voronoi,
                                 flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                 simplify_voronoi=False)
    split_points = []
    split_points.append(list(inlet_ar))
    # r = vmtkscripts.vmtkRenderer()
    # r.Execute()
    surface_pv = pv.wrap(surface_f)
    colours =['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    for i in range(2, len(nodes_array_sort)):
        # outlet = np.asarray(list(nodes_array_sort[0]) + list(nodes_array_sort[i]))
        outlet = np.asarray(list(nodes_array_sort[i]))
        outlet = outlet.tolist()
        print(outlet)
        centerline_vmtk, centerline = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                     flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                     simplify_voronoi=False)
        im = vmtkcenterlinemodeller(centerline, size=[64, 64, 64])
        mc = vmtkmarchingcubes(im, level=0.0)
        mc_pv = pv.wrap(mc)
        p = pv.Plotter()
        p.add_mesh(surface_pv, color ='grey', style='surface', opacity=0.5)
        p.add_mesh(mc_pv, color=colours[i-2],style='surface',opacity=0.3)
        p.add_mesh(centerline_pv, color='blue',style='surface')
        p.show()
        


def split_points_new(surface_f, nodes_final):
    """
    Find PLZs points (split points) along centerline using VMTK tools.
    
    This function computes the main centerline of a surface using VMTK tools and splits it at specified points.
    It takes a surface mesh (`surface_f`) and a set of points (`nodes_final`) as input. It then computes the main 
    centerline of the surface and splits it at the specified points. The function returns the split points and the 
    main centerline.
    
    Parameters
    ----------
    surface_f : vtkPolyData
        The surface mesh on which the centerline computation is performed.
    nodes_final : array
        The set of PLZs points at which to split the centerline.
    
    Returns
    -------
    split_points : list
        A list of split points along the main centerline.
    main_centerline : vtkPolyData
        The main centerline computed using VMTK tools.
    
    Notes
    -----
    This function uses VMTK tools to compute the main centerline (not including branches) of a surface mesh (`surface_f`)  and splits it 
    at specified points (`nodes_final`). The computed main centerline and the split points are returned.
    
    The function iterates through the specified points in `nodes_final`, computing the main centerline of the surface 
    mesh from the inlet point to each outlet point. It then splits the main centerline at the specified outlet points 
    and returns the resulting split points.
    
    """
    end_point = 0
    method = 'pointlist'
    resampling_step = 0.5
    voronoi = None
    pole_ids = None
    nodes_array = np.asarray(nodes_final)
    nodes_array_sort = nodes_array[nodes_array[:, 2].argsort()]
    inlet_ar = nodes_array_sort[1]
    inlet = inlet_ar.tolist()
    outlet_last_ar = nodes_array_sort[0]
    outlet_last = outlet_last_ar.tolist()
    main_centerline_vmtk, main_centerline = vmtk_compute_centerlines(end_point, inlet, method, outlet_last, pole_ids, resampling_step, surface_f, voronoi,
                                 flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                 simplify_voronoi=False)
    split_points = []
    split_points.append(list(inlet_ar))
    main_centerline_ar = centerline_numpy(main_centerline)
  
    for i in range(2, len(nodes_array_sort)):
        outlet = np.asarray(list(nodes_array_sort[i]))
        outlet = outlet.tolist()
        print(outlet)
        centerline_vmtk, centerline = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                     flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                     simplify_voronoi=False)
        # split_point = bifurication_point(centerline, centerline_t)
        # main_centerline = centerline
        im = vmtkcenterlinemodeller(centerline, size=[64, 64, 64])
        mc = vmtkmarchingcubes(im, level=0.0)
        surface_ar = surface_numpy(mc)
        kdtree = scipy.spatial.cKDTree(surface_ar['Points'])
        distances, indices = kdtree.query(main_centerline_ar['Points'][0:(len(main_centerline_ar['Points'])-10)], k=1)
        d_min = np.argmin(distances)
        split_point = main_centerline_ar['Points'][d_min]
        split_points.append(split_point) 
    split_points.append(list(outlet_last_ar))
    return split_points, main_centerline

    
def proximal(centerline, point):
    """
    Find the proximal point at 20mm distance on a centerline to a given point. It calculates the distance travelled 
    along the centerline from the given point, and returns the point at 20mm distance on the centerline.

    Parameters
    ----------
    centerline : vtkPolyData
        The centerline represented as a vtkPolyData object.
    point : array
        The coordinates of the given point.

    Returns
    -------
    proximal_point : array
        The coordinates of the proximal point at 20mm distance on the centerline to the given point.

    """
    
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    point_c = locator.FindClosestPoint(point)
    cg =  vmtkcenterlinegeometry(centerline)
    cg_array = centerline_numpy(cg)
    point_c_rev = cg_array['Points'].shape[0]-point_c-1
    cg_array_rev = cg_array['Points'][::-1]
    dist_in_voxel = [np.linalg.norm(cg_array_rev[i]-cg_array_rev[i-1]) for i in range(1,point_c_rev+1)]
    dist_travelled_mm=np.cumsum(dist_in_voxel)
    dist_travelled_20 = dist_travelled_mm[point_c_rev-1] - dist_travelled_mm 
    for index, value in enumerate(dist_travelled_20):
        if value >= 21:
            final_index = index
    # print(np.linalg.norm(cg_array_rev[final_index]-point))
    return cg_array_rev[final_index]

def distal(centerline, point, distance):
    """
    This function finds the distal point on a centerline from a given point at a specified distance.
  

    Parameters
    ----------
    centerline : vtkPolyData
        The centerline represented as a vtkPolyData object.
    point : array
        The coordinates of the given point.
    distance : float
        The distance along the centerline from the given point to the distal point.

    Returns
    -------
    distal_point : array
        The coordinates of the distal point on the centerline from the given point at the specified distance.
    """
    
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    point_c = locator.FindClosestPoint(point)
    cg =  vmtkcenterlinegeometry(centerline)
    cg_array = centerline_numpy(cg)
    point_c_rev = cg_array['Points'].shape[0]-point_c-1
    cg_array_rev = cg_array['Points'][::-1]
    dist_in_voxel = [np.linalg.norm(cg_array_rev[i+1]-cg_array_rev[i]) for i in range(point_c_rev,cg_array['Points'].shape[0]-1)]
    dist_travelled_mm=np.cumsum(dist_in_voxel)
    min_idc = np.argmin(np.abs(dist_travelled_mm - distance)) + point_c_rev
    # print(np.linalg.norm(cg_array_rev[final_index]-point))
    return cg_array_rev[min_idc]

def radius_surface_curvature(filename, surface_f, cg_array, nodes_array_sort):
    """
    Compute the radius of curvature on a surface.

    This function computes the outer radius of curvature on a surface mesh. It first writes the surface 
    mesh to a file using VMTK tools, then loads the mesh using trimesh, and creates a graph with 
    edge attributes for length. It then computes the shortest path on the mesh from the start point 
    to the end point defined by the centerline points. Finally, it finds the path mirror opposite to the short
    path w.r.to centerline to find the outer surface path and calculates the radius of curvature 
    based on the computed path and returns the result.

    Parameters
    ----------
    filename : str
        The filename to save the surface mesh.
    surface_f : vtkPolyData
        The surface mesh represented as a vtkPolyData object.
    cg_array : array
        Array of centerline points.
    nodes_array_sort : array
        Array of sorted nodes.

    Returns
    -------
    radius : float
        The radius of curvature computed based on the surface.

    """

    vmtksurfacewriter(surface_f,filename)
    mesh = trimesh.load_mesh(filename)
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    # create the graph with edge attributes for length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)
    surface_ar = surface_numpy(surface_f)
    kdtree_s = scipy.spatial.cKDTree(mesh.vertices)
    distances, indices = kdtree_s.query(nodes_array_sort, k=1)
    end = indices[0]
    surface_ar = surface_numpy(surface_f)
    kdtree_s = scipy.spatial.cKDTree(mesh.vertices)
    distances, indices = kdtree_s.query(cg_array['Points'], k=1)
    # arbitrary indices of mesh.vertices to test with
    start = indices[-1]
    # run the shortest path query using length for edge weight
    path = nx.shortest_path(g, source=start, target=end, weight="length")
    path_points = mesh.vertices[path]
    kdtree = scipy.spatial.cKDTree(path_points)
    distances, indices = kdtree.query(cg_array['Points'], k=1)

    new_points = []
    for i in range(len(indices)):
        temp_val = 2*cg_array['Points'][i]-path_points[indices[i]]
        new_points.append(temp_val)
        
    radius = radius_centerline_curvature(new_points)
    
    return radius

def radius_centerline_curvature(points_ar):
    """
    Compute the radius of curvature along a centerline.

    This function computes the radius of curvature along a centerline represented by a set of points.
    It first computes the mean of the points and centers the points around the mean. It then computes 
    the singular value decomposition (SVD) of the centered points to determine the normal vector to 
    the centerline. Next, it projects the points onto a plane orthogonal to the normal vector using 
    Rodrigues' rotation. Finally, it fits a circle to the projected points and returns the radius of 
    curvature of the circle.

    Parameters
    ----------
    points_ar : array-like
        Array of points representing the centerline.

    Returns
    -------
    radius : float
        The radius of curvature along the centerline.

    """
    points_ar = np.asarray(points_ar)
    mean = points_ar.mean(axis=0)
    centered = points_ar - mean
    u_ar,s_ar,v_ar = linalg.svd(centered)
    normal = v_ar[2,:]
    dot_p_val = -dot(mean, normal)  # d = -<p,n>
    proj = rodrigues_rotation(centered, normal, [0,0,1])
    xc, yc, r = fit_circle_to_points(proj[:,0], proj[:,1])
    return r

# arr.reverse() 
def arc_ga(point_a, point_b, surface_f, filename, nodes_array_sort):
    """
    Perform geometric analysis for an arc defined by two points on a surface.

    This function performs geometric analysis for an aortic arch defined by two points (`point_a` and `point_b`) 
    on a surface mesh (`surface_f`). It computes the centerline between the two points, extracts 
    geometric features such as centerline curvature, surface curvature, tortuosity, and ratio of 
    outer curvature to centerline curvature, and returns the results as a dictionary.

    Parameters
    ----------
    point_a : array
        The coordinates of the starting point of the aortic arch.
    point_b : array
        The coordinates of the ending point of the aortic arch.
    surface_f : vtkPolyData
        The surface mesh represented as a vtkPolyData object.
    filename : str
        The filename to save the surface mesh.
    nodes_array_sort : array
        Array of sorted nodes.

    Returns
    -------
    ga_dict : dict
        A dictionary containing the geometric analysis results:
        - 'centerline_curvature': The radius of curvature along the centerline.
        - 'outer_curvature': The radius of curvature of the outer surface.
        - 'toruosity': The tortuosity of the centerline.
        - 'ratio': The ratio of outer curvature to centerline curvature.

    """
    end_point = 0
    method = 'pointlist'
    resampling_step = 0.5
    voronoi = None
    pole_ids = None
    inlet = np.asarray(point_a)
    inlet = inlet.tolist()
    outlet = np.asarray(point_b)
    outlet = outlet.tolist()
    ga_dict = {}
    centerline_vmtk, centerline = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                 flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                 simplify_voronoi=False)
 
    im = vmtkcenterlinemodeller(centerline, size=[64, 64, 64])
    mc = vmtkmarchingcubes(im, level=0.0)
    # surface_viewer(mc)
    mc_curvature = vmtksurfacecurvature(mc, curvature_type='mean', absolute_curvature=0,
                             median_filtering=0)
    mc_curvature_ar = surface_numpy(mc_curvature)
    cg =  vmtkcenterlinegeometry(centerline)
    cg_array = centerline_numpy(cg)
    tortuosity = cg_array['CellData']['Tortuosity']
    # radius_of_curvature = 1/curvature
    radius_centerline = radius_centerline_curvature(cg_array['Points'])
    radius_surface = radius_surface_curvature(filename, surface_f, cg_array, nodes_array_sort)
    ga_dict['centerline_curvature'] = radius_centerline
    ga_dict['outer_curvature'] = radius_surface
    ga_dict['toruosity'] = tortuosity
    ga_dict['ratio'] = ga_dict['outer_curvature']/ga_dict['centerline_curvature']
    return ga_dict


def tau_angle(point_a, point_b, centerline):
    """
    Compute the angle between tangent vectors at two points along a centerline.

    This function computes the angle between tangent vectors at two specified points (`point_a` and `point_b`) 
    along a centerline using dot product formula.

    Parameters
    ----------
    point_a : array
        The coordinates of the first point.
    point_b : array
        The coordinates of the second point.
    centerline : vtkPolyData
        The centerline represented as a vtkPolyData object.

    Returns
    -------
    angle_degrees : float
        The angle between tangent vectors at the specified points, in degrees.

    """
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    point_index_a = locator.FindClosestPoint(point_a)
    point_index_b = locator.FindClosestPoint(point_b)
    cg =  vmtkcenterlinegeometry(centerline)
    cg_array = centerline_numpy(cg)
    tangent_ar = cg_array['PointData'][ 'FrenetTangent']
    tangent_vector_a = tangent_ar[point_index_a]
    tangent_vector_b = tangent_ar[point_index_b]
    tangent_vector_a_n = vtk.vtkMath.Normalize(tangent_vector_a)
    tangent_vector_b_n = vtk.vtkMath.Normalize(tangent_vector_b)
    # Calculate the dot product of the vectors
    dot_product = (vtk.vtkMath.Dot(tangent_vector_a, tangent_vector_b))/(tangent_vector_a_n*tangent_vector_b_n)
    # Calculate the angle between the vectors (in radians)
    angle_radians = math.acos(dot_product)
    # Convert the angle to degrees
    angle_degrees = vtk.vtkMath.DegreesFromRadians(angle_radians)
       
    return angle_degrees
    
    
def tortuosity_angle(point_z, point_20, point_40):  
    """
   Compute the angle between two vectors defined by three points.

   This function computes the angle between two vectors defined by three points: `point_z`, `point_20`, and `point_40`. 
   It first computes the vectors between `point_z` and `point_20`, and between `point_20` and `point_40`. 
   Then, it calculates the dot product and magnitudes of the vectors and computes the cosine of the angle between 
   them. Finally, it converts the cosine of the angle to the angle in degrees and returns it.

   Parameters
   ----------
   point_z : array
       The coordinates of the first point.
   point_20 : array
       The coordinates of the point at 20mm distance
   point_40 : array
       The coordinates of the point at 40mm distance

   Returns
   -------
   angle_degrees : float
       The angle between the vectors defined by the three points, in degrees.

   """
    point1 = point_z
    point2 = point_20
    point3 = point_40
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point2) - np.array(point3)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees
    
    
def zone_ga(zone_points_edit, surface_f, main_centerline):  
    """
    Perform geometric analysis for each zones along a centerline.

    It computes various geometric features such as maximum diameter, length, tortuosity angle, and tau angle for each zone 
    between consecutive points on the centerline. The results are returned as a list of dictionaries, where each dictionary 
    contains the geometric analysis results for a specific zone.

    Parameters
    ----------
    zone_points_edit : list 
        List of points defining the zones along the centerline.
    surface_f : vtkPolyData
        The surface mesh represented as a vtkPolyData object.
    main_centerline : vtkPolyData
        The main centerline represented as a vtkPolyData object.

    Returns
    -------
    zone : list of dict
        A list of dictionaries containing the geometric analysis results for each zone:
        - 'max_diameter': The maximum diameter of the zone.
        - 'length': The length of the zone.
        - 'tortuosity_angle': The angle between two vectors defined by three points in the zone.
        - 'tau_angle': The angle between tangent vectors at proximal point and point at 40mm distance.

   
    """
    points = []
    # points = zone_points.copy()
    points = zone_points_edit.copy()
    zone = []
    
    end_point = 0
    method = 'pointlist'
    resampling_step = 0.5
    voronoi = None
    pole_ids = None
    for i in range(0, len(points)-1):
        zone_dic = {}
        inlet = np.asarray(points[i])
        inlet = inlet.tolist()
        outlet = np.asarray(points[i+1])
        outlet = outlet.tolist()
        centerline_vmtk_z, centerline= vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                     flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                     simplify_voronoi=False)
        cl_array = centerline_numpy(centerline)
        zone_dic['max_diameter'] = 2*np.max(cl_array['PointData']['MaximumInscribedSphereRadius'])
        cg =  vmtkcenterlinegeometry(centerline)
        cg_array = centerline_numpy(cg)
        zone_dic['length'] = cg_array['CellData']['Length']
        
        point_20 = distal(main_centerline, points[i], 20)
        point_40 = distal(main_centerline, points[i], 40)
        zone_dic['tortuosity_angle'] = tortuosity_angle(points[i], point_20, point_40)
        zone_dic['tau_angle'] = tau_angle(points[i], point_40, main_centerline)
        zone.append(zone_dic)
     
    return zone

#%%
def ga_zone_visualization(nodes_final, surface_f, main_centerline, output_file_mapped):
    """
    Visualize geometric analysis zones on a surface mesh.

    This function visualizes proximal landing zones(PLZs) on a surface mesh. It computes and displays zones defined by 
    the centerline and points provided as input. Each zone is distinguished by a unique color on the surface mesh plot.

    Parameters
    ----------
    nodes_final : array
        Coordinates of points defining the zones.
    surface_f : vtkPolyData
        The surface mesh represented as a vtkPolyData object.
    main_centerline : vtkPolyData
        The main centerline represented as a vtkPolyData object.

    Returns
    -------
    None

    """
    surface_pv = pv.wrap(surface_f)
    centerline_pv = pv.wrap(main_centerline)
    end_point = 0
    method = 'pointlist'
    resampling_step = 2
    voronoi = None
    pole_ids = None
    
    nodes_array = np.asarray(nodes_final)
    nodes_array_sort = nodes_array[nodes_array[:, 2].argsort()]
    colors = ['black','blue', 'red', 'green', 'yellow','violet','black']
    radius = []
    points = []
   
    centerlines={}
    side_centerline = {}
    
    inlet = nodes_array_sort[1].tolist()
    outlet = zone_points_edit[0].tolist()
    main_centerline_vmtk1, main_centerline1 = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                 flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                 simplify_voronoi=False)
    ar_temp = centerline_numpy(main_centerline1)
    points = points+list(ar_temp['Points'])
    centerlines[0] = np.asarray(points)
    radius = radius+list(ar_temp['PointData']['MaximumInscribedSphereRadius'])
    for j in range(0,(len(zone_points_edit)-1)):
        inlet = zone_points_edit[j].tolist()
        outlet = zone_points_edit[j+1].tolist()
        main_centerline_vmtk1, main_centerline1 = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                     flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                     simplify_voronoi=False)
        
        ar_temp = centerline_numpy(main_centerline1)
        points = points+list(ar_temp['Points'])
        centerlines[j+1] = ar_temp['Points']
        radius = radius+list(ar_temp['PointData']['MaximumInscribedSphereRadius'])
    
    inlet = zone_points_edit[-1].tolist()
    outlet = nodes_array_sort[0].tolist()
    main_centerline_vmtk1, main_centerline1 = vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface_f, voronoi,
                                 flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                                 simplify_voronoi=False)    
    ar_temp = centerline_numpy(main_centerline1)
    points = points+list(ar_temp['Points'])
    centerlines[j+2] = ar_temp['Points']
    radius = radius+list(ar_temp['PointData']['MaximumInscribedSphereRadius'])
    surface_ar = surface_numpy(surface_f)
    kdtree = scipy.spatial.cKDTree(points)
    distances, indices = kdtree.query(surface_ar['Points'], k=1)   
    surface_pv = pv.wrap(surface_f)
    
    n=[]
    
    
    for i in range(len(indices)):
        for k in range(0, len(centerlines)):
            
            if points[indices[i]] in centerlines[k]:
               if distances[i]-radius[indices[i]]>7 or k==(len(centerlines)-1): 
                   n.append(0)
                   break
               else:
                   n.append(k)
                   break
            # print(n)
                
    labels = n    
    surface_pv.point_data["labels"] = labels
    colours=['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    c_final = colours[0:len(centerlines)]
    surface_pv.plot(scalars="labels", show_scalar_bar=False, cmap=c_final) 
    surface_pv.save(output_file_mapped)
    # p = pv.Plotter()
    # p.add_mesh(surface_pv, style='surface', scalars="labels", show_scalar_bar=False, cmap=c_final, opacity=0.8)
    # p.add_mesh(centerline_pv, color='red',style='surface')
    # p.show()
    # pv.save_meshio('output_c.vtk', p)
    
def end_point_f(centerline):
    """
   Compute the end point of a centerline.

   This function computes the end point of a centerline provided as input on the same axial plane as the first point. 
   the centerline.

   Parameters
   ----------
   centerline : vtkPolyData
       The centerline represented as a vtkPolyData object.

   Returns
   -------
   end_point : array
       The coordinates of the end point of the centerline.

   """
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    cg =  vmtkcenterlinegeometry(centerline)
    cg_array = centerline_numpy(centerline)
    cg_array_rev = cg_array['Points'][::-1]
    # axial_data = []
    # for i in range(len(cg_array['Points'])):
    #     axial_data.append(centerline.GetPoint(i))
    
    axial_data = cg_array_rev
    axial_data = axial_data[:,2]
    axial_data_sub = axial_data - axial_data[0]
    min_idc = np.argmin(np.absolute(axial_data_sub[10:len(axial_data_sub)]))+10 
    return cg_array_rev[min_idc]    

