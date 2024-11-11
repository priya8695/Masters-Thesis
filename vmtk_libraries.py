# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:56:54 2023

@author: PriyaPrabhakar
"""

##   Copyright (c) Aslak W. Bergersen, Henrik A. Kjeldsberg. All rights reserved.
##   See LICENSE file for details.

##      This software is distributed WITHOUT ANY WARRANTY; without even
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##      PURPOSE.  See the above copyright notices for more information.

from os import path
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vmtk import vtkvmtk, vmtkscripts

# Global array names
#from morphman.common.vtk_wrapper import read_polydata, write_polydata

radiusArrayName = 'MaximumInscribedSphereRadius'
surfaceNormalsArrayName = 'SurfaceNormalArray'
parallelTransportNormalsArrayName = 'ParallelTransportNormals'
groupIDsArrayName = "GroupIds"
abscissasArrayName = 'Abscissas'
blankingArrayName = 'Blanking'
branchClippingArrayName = 'BranchClippingArray'


def vmtk_smooth_centerline(centerlines, num_iter, smooth_factor):
    """
    Wrapper for vmtkCenterlineSmoothing. Smooth centerlines with a moving average filter.

    Args:
        centerlines (vtkPolyDat): Centerline to be smoothed.
        num_iter (int): Number of smoothing iterations.
        smooth_factor (float): Smoothing factor

    Returns:
        vtkPolyData: Smoothed version of input centerline
    """
    centerline_smoothing = vmtkscripts.vmtkCenterlineSmoothing()
    centerline_smoothing.Centerlines = centerlines
    centerline_smoothing.SetNumberOfSmoothingIterations = num_iter
    centerline_smoothing.SetSmoothingFactor = smooth_factor
    centerline_smoothing.Execute()
    centerlines_smoothed = centerline_smoothing.Centerlines

    return centerlines_smoothed




def vmtk_compute_centerlines(end_point, inlet, method, outlet, pole_ids, resampling_step, surface, voronoi,
                             flip_normals=False, cap_displacement=None, delaunay_tolerance=None,
                             simplify_voronoi=False, file_name=None):
    """
    Wrapper for vmtkCenterlines.
    compute centerlines from a branching tubular surface. Seed points can be interactively selected on the surface,
    or specified as the barycenters of the open boundaries of the surface.

    Args:
        end_point (int): Toggle append open profile barycenters to centerlines
        surface (vktPolyData): Surface model
        voronoi (vtkPolyData): Voronoi diagram based on previous centerlines (Optional)
        inlet (ndarray): List of source point coordinates
        method (str): Seed point selection method
        outlet (ndarray): List of target point coordinates
        pole_ids (ndarray): Pole ID list of Voronoi diagram (Optional)
        resampling_step (float): Resampling step
        flip_normals (float): Flip normals after outward normal computation
        cap_displacement (float): Displacement of the center points of caps at open profiles along their normals
        delaunay_tolerance (float): Tolerance for evaluating coincident points during Delaunay tessellation
        simplify_voronoi (bool): Toggle simplification of Voronoi diagram

    Returns:

    """
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    centerlines.SeedSelectorName = method
    centerlines.AppendEndPoints = end_point
    centerlines.Resampling = 1
    centerlines.ResamplingStepLength = resampling_step
    centerlines.SourcePoints = inlet
    centerlines.TargetPoints = outlet
    centerlines.CenterlinesOutputFileName = file_name

    if voronoi is not None and pole_ids is not None:
        centerlines.VoronoiDiagram = voronoi
        centerlines.PoleIds = pole_ids
    if flip_normals:
        centerlines.FlipNormals = 1
    if cap_displacement is not None:
        centerlines.CapDisplacement = cap_displacement
    if delaunay_tolerance is not None:
        centerlines.DelaunayTolerance = delaunay_tolerance
    if simplify_voronoi:
        centerlines.SimplifyVoronoi = 1
    centerlines.Execute()
    centerlines_output = centerlines.Centerlines

    return centerlines, centerlines_output


def vmtksurfacewriter(polydata, path):
    """Write a vtkPolyData object (e.g. surface and centerlines) to disk.

    Args:
        polydata: vtkPolyData object.
        path: Path to the polydata file.

    Returns:
        n/a

    Note:
        Writes several polydata formats: vtp, vtk, stl (use only for
        triangulated surface meshes), ply, tec (tecplot), dat

    """
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = polydata
    writer.OutputFileName = path
    writer.Execute()

def vmtk_compute_centerline_sections(surface, centerlines):
    """
    Wrapper for vmtk centerline sections.

    Args:
        surface (vtkPolyData): Surface to measure area.
        centerlines (vtkPolyData): centerline to measure along.

    Returns:
        line (vtkPolyData): centerline with the attributes'
        centerline_sections_area (vtkPolyData): sections along the centerline
    """
    centerline_sections = vtkvmtk.vtkvmtkPolyDataCenterlineSections()
    centerline_sections.SetInputData(surface)
    centerline_sections.SetCenterlines(centerlines)
    centerline_sections.SetCenterlineSectionAreaArrayName('CenterlineSectionArea')
    centerline_sections.SetCenterlineSectionMinSizeArrayName('CenterlineSectionMinSize')
    centerline_sections.SetCenterlineSectionMaxSizeArrayName('CenterlineSectionMaxSize')
    centerline_sections.SetCenterlineSectionShapeArrayName('CenterlineSectionShape')
    centerline_sections.SetCenterlineSectionClosedArrayName('CenterlineSectionClosed')
    centerline_sections.Update()

    centerlines_sections_area = centerline_sections.GetOutput()
    line = centerline_sections.GetCenterlines()

    return line, centerlines_sections_area



def vmtk_compute_geometric_features(centerlines, smooth, output_smoothed=False, factor=1.0, iterations=100):
    """Wrapper for vmtk centerline geometry.

    Args:
        centerlines (vtkPolyData): Line to compute centerline geometry from.
        smooth (bool): Turn on and off smoothing before computing the geometric features.
        output_smoothed (bool): Turn on and off the smoothed centerline.
        factor (float): Smoothing factor.
        iterations (int): Number of iterations.

    Returns:
        line (vtkPolyData): Line with geometry.
    """
    geometry = vmtkscripts.vmtkCenterlineGeometry()
    geometry.Centerlines = centerlines

    if smooth:
        geometry.LineSmoothing = 1
        geometry.OutputSmoothedLines = output_smoothed
        geometry.SmoothingFactor = factor
        geometry.NumberOfSmoothingIterations = iterations
    geometry.FernetTangentArrayName = "FernetTangent"
    geometry.FernetNormalArrayName = "FernetNormal"
    geometry.FrenetBinormalArrayName = "FernetBiNormal"
    geometry.CurvatureArrayName = "Curvature"
    geometry.TorsionArrayName = "Torsion"
    geometry.TortuosityArrayName = "Tortuosity"
    geometry.Execute()

    return geometry.Centerlines



def vmtk_compute_centerline_attributes(centerlines):
    """ Wrapper for centerline attributes.

    Args:
        centerlines (vtkPolyData): Line to investigate.

    Returns:
        line (vtkPolyData): Line with centerline attributes.
    """
    attributes = vmtkscripts.vmtkCenterlineAttributes()
    attributes.Centerlines = centerlines
    attributes.NormalsArrayName = parallelTransportNormalsArrayName
    attributes.AbscissaArrayName = abscissasArrayName
    attributes.Execute()
    centerlines = attributes.Centerlines

    return centerlines



def vmtk_resample_centerline(centerlines, length):
    """Wrapper for vmtkcenterlineresampling

    Args:
        centerlines (vtkPolyData): line to resample.
        length (float): resampling step.

    Returns:
        line (vtkPolyData): Resampled line.
    """
    resampler = vmtkscripts.vmtkCenterlineResampling()
    resampler.Centerlines = centerlines
    resampler.Length = length
    resampler.Execute()

    resampled_centerline = resampler.Centerlines

    return resampled_centerline



def vmtk_cap_polydata(surface, boundary_ids=None, displacement=0.0, in_plane_displacement=0.0):
    """Wrapper for vmtkCapPolyData.
    Close holes in a surface model.

    Args:
        in_plane_displacement (float): Displacement of boundary barycenter, at section plane relative to the radius
        displacement (float):  Displacement of boundary barycenter along boundary normals relative to the radius.
        boundary_ids (ndarray): Set ids of the boundaries to cap.
        surface (vtkPolyData): Surface to be capped.

    Returns:
        surface (vtkPolyData): Capped surface.
    """
    surface_capper = vtkvmtk.vtkvmtkCapPolyData()
    surface_capper.SetInputData(surface)
    surface_capper.SetDisplacement(displacement)
    surface_capper.SetInPlaneDisplacement(in_plane_displacement)
    if boundary_ids is not None:
        surface_capper.SetBoundaryIds(boundary_ids)
    surface_capper.Update()

    return surface_capper.GetOutput()



def vmtk_smooth_surface(surface, method, iterations=800, passband=1.0, relaxation=0.01, normalize_coordinates=True,
                        smooth_boundary=True):
    """Wrapper for a vmtksurfacesmoothing.

    Args:
        smooth_boundary (bool): Toggle allow change of position of boundary points
        normalize_coordinates (bool): Normalization of coordinates prior to filtering,
            minimize spurious translation effects (Taubin only)
        surface (vtkPolyData): Input surface to be smoothed.
        method (str): Smoothing method.
        iterations (int): Number of iterations.
        passband (float): The passband for Taubin smoothing.
        relaxation (float): The relaxation for laplace smoothing.

    Returns:
        surface (vtkPolyData): The smoothed surface.

    """
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.NumberOfIterations = iterations

    if method == "laplace":
        smoother.RelaxationFactor = relaxation
    elif method == "taubin":
        smoother.PassBand = passband

    if not normalize_coordinates:
        smoother.NormalizeCoordinates = 0
    if not smooth_boundary:
        smoother.BoundarySmoothing = 0

    smoother.Method = method
    smoother.Execute()
    surface = smoother.Surface

    return surface



def vmtk_compute_voronoi_diagram(surface, filename, simplify_voronoi=False, cap_displacement=None, flip_normals=False,
                                 check_non_manifold=False, delaunay_tolerance=0.001, subresolution_factor=1.0):
    """
    Wrapper for vmtkDelanayVoronoi. Creates a surface model's
    corresponding voronoi diagram.

    Args:
        subresolution_factor (float): Factor for removal of subresolution tetrahedra
        flip_normals (bool): Flip normals after outward normal computation.
        cap_displacement (float): Displacement of the center points of caps at open profiles along their normals
        simplify_voronoi (bool): Use alternative algorith for compute Voronoi diagram, reducing quality, improving speed
        check_non_manifold (bool): Check the surface for non-manifold edges
        delaunay_tolerance (float): Tolerance for evaluating coincident points during Delaunay tessellation
        surface (vtkPolyData): Surface model
        filename (str): Path where voronoi diagram is stored

    Returns:
        new_voronoi (vtkPolyData): Voronoi diagram
    """
    #if path.isfile(filename):
        #return read_polydata(filename)

    voronoi = vmtkscripts.vmtkDelaunayVoronoi()
    voronoi.Surface = surface
    voronoi.RemoveSubresolutionTetrahedra = 1
    voronoi.DelaunayTolerance = delaunay_tolerance
    voronoi.SubresolutionFactor = subresolution_factor
    if simplify_voronoi:
        voronoi.SimplifyVoronoi = 1
    if cap_displacement is not None:
        voronoi.CapDisplacement = cap_displacement
    if flip_normals:
        voronoi.FlipNormals = 1
    if check_non_manifold:
        voronoi.CheckNonManifold = 1

    voronoi.Execute()
    new_voronoi = voronoi.VoronoiDiagram

    #write_polydata(new_voronoi, filename)
    return voronoi
    # return new_voronoi

def vmtk_polyball_modeller(voronoi_diagram, poly_ball_size=[64, 64, 64]):
    """
    Wrapper for vtkvmtkPolyBallModeller.
    Create an image where a polyball or polyball line are evaluated as a function.

    Args:
        voronoi_diagram (vtkPolyData): Input Voronoi diagram representing surface model
        poly_ball_size (list): Resolution of output

    Returns:
        vtkvmtkPolyBallModeller: Image where polyballs have been evaluated over a Voronoi diagram
    """
    # modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller = vmtkscripts.vmtkPolyBallModeller()
    modeller.Surface = voronoi_diagram 
    modeller.RadiusArrayName = radiusArrayName
    # modeller.UsePolyBallLineOff()
    modeller.SampleDimensions = poly_ball_size
    # modeller.Update()
    modeller.Execute()

    return modeller




def vmtk_surface_connectivity(surface, method="largest", clean_output=True, closest_point=None):
    """
    Wrapper for vmtkSurfaceConnectivity. Extract the largest connected region,
    the closest point-connected region or the scalar-connected region from a surface

    Args:
        surface (vtkPolyData): Surface model
        method (str): Connectivity method, either 'largest' or 'closest'
        clean_output (bool): Clean the unused points in the output
        closest_point (ndarray): Coordinates of the closest point

    Returns:
        vmtkSurfaceConnectivity: Filter for extracting the largest connected region
    """
    connector = vmtkscripts.vmtkSurfaceConnectivity()
    connector.Surface = surface
    connector.Method = method
    if clean_output:
        connector.CleanOutput = 1
    if closest_point is not None:
        connector.ClosestPoint = closest_point

    connector.Execute()

    return connector




def vmtk_branch_clipper(centerlines, surface, clip_value=0.0, inside_out=False, use_radius_information=True,
                        group_id=None,interactive=False):
    """
    Wrapper for vmtkBranchClipper. Divide a surface in relation to its split and grouped centerlines.

    Args:
        centerlines (vtkPolyData): Input centerlines
        surface (vtkPolyData): Input surface model
        clip_value (float):
        inside_out (bool): Get the inverse of the branch clipper output.
        use_radius_information (bool): To use MISR info for clipping branches.
        interactive (bool): Use interactive mode, requires user input.

    Returns:
        vmtkBranchClipper: Branch clipper used to divide a surface into regions.
    """
    clipper = vmtkscripts.vmtkBranchClipper()
    clipper.Surface = surface
    clipper.Centerlines = centerlines
    clipper.ClipValue = clip_value
    clipper.RadiusArrayName = radiusArrayName
    clipper.GroupIdsArrayName = groupIDsArrayName
    clipper.BlankingArrayName = blankingArrayName
    clipper.CutoffRadiusFactor = 1E16
    clipper.GroupIds = group_id
    if inside_out:
        clipper.InsideOut = 1
    if not use_radius_information:
        clipper.UseRadiusInformation = 0
    if interactive:
        clipper.Interactive = 1

    clipper.Execute()

    return clipper



def vmtk_endpoint_extractor(centerlines, number_of_end_point_spheres, number_of_gap_spheres=1):
    """
    Wrapper for vmtkEndpointExtractor.
    Find the endpoints of a split and grouped centerline

    Args:
        centerlines (vtkPolyData): Input centerlines.
        number_of_end_point_spheres (float): Number of spheres to skip at endpoint
        number_of_gap_spheres (float): Number of spheres to skip per gap.

    Returns:
        vmtkEndpointExtractor: Endpoint extractor based on centerline
    """
    extractor = vmtkscripts.vmtkEndpointExtractor()
    extractor.Centerlines = centerlines
    extractor.RadiusArrayName = radiusArrayName
    extractor.GroupIdsArrayName = groupIDsArrayName
    extractor.BlankingArrayName = branchClippingArrayName
    extractor.NumberOfEndPointSpheres = number_of_end_point_spheres
    extractor.NumberOfGapSpheres = number_of_gap_spheres
    extractor.Execute()

    return extractor



def vmtk_compute_surface_normals(surface, auto_orient_normals=True, orient_normals=True,
                                 compute_cell_normals=False, flip_normals=False):
    """
    Wrapper for vmtkSurfaceNormals.
    Computes the normals of the input surface.

    Args:
        surface (vtkPolyData): Input surface model
        auto_orient_normals (bool): Try to auto orient normals outwards
        orient_normals (bool): Try to orient normals so that neighboring points have similar orientations
        compute_cell_normals (bool): Compute cell normals instead of point normals
        flip_normals (bool): Flip normals after computing them

    Returns:
        vtkPolyData: Surface model with computed normals
    """
    surface_normals = vmtkscripts.vmtkSurfaceNormals()
    surface_normals.Surface = surface
    surface_normals.NormalsArrayName = surfaceNormalsArrayName
    if not auto_orient_normals:
        surface_normals.AutoOrientNormals = 0
    if not orient_normals:
        surface_normals.Consistency = 0
    if compute_cell_normals:
        surface_normals.ComputeCellNormals = 1
    if flip_normals:
        surface_normals.FlipNormals = 1

    surface_normals.Execute()
    surface_with_normals = surface_normals.Surface

    return surface_with_normals




def vmtk_compute_branch_extractor(centerlines):
    """
    Wrapper for vmtkBranchExtractor.
    Split and group centerlines along branches:

    Args:
        centerlines (vtkPolyData): Line to split into branches.

    Returns:
        vtkPolyData: Split centerline.
    """

    brancher = vmtkscripts.vmtkBranchExtractor()
    brancher.Centerlines = centerlines
    brancher.RadiusArrayName = radiusArrayName
    brancher.Execute()
    centerlines_branched = brancher.Centerlines

    return centerlines_branched



def vmtk_surface_curvature(surface, curvature_type="mean", absolute=False,
                           median_filtering=False, curvature_on_boundaries=False,
                           bounded_reciporcal=False, epsilon=1.0, offset=0.0):
    """Wrapper for vmtksurfacecurvature

    Args:
        surface (vtkPolyData): The input surface
        curvature_type (str): The type of surface curvature to compute (mean | gaussian | maximum | minimum)
        absolute (bool): Output the absolute value of the curvature
        median_filtering (bool): Output curvature after median filtering to suppress numerical noise speckles
        curvature_on_boundaries (bool): Turn on/off curvature on boundaries
        bounded_reciporcal (bool): Output bounded reciprocal of the curvature
        epsilon (float): Bounded reciprocal epsilon at the denominator
        offset (float): Offset curvature by the specified value

    Returns:
        surface (vtkPolydata): Input surface with an point data array with curvature values
    """
    curvature = vmtkscripts.vmtkSurfaceCurvature()
    curvature.Surface = surface
    curvature.CurvatureType = curvature_type
    if absolute:
        curvature.AbsoluteCurvature = 1
    else:
        curvature.AbsoluteCurvature = 0
    if median_filtering:
        curvature.MedianFiltering = 1
    else:
        curvature.MedianFiltering = 0
    if curvature_on_boundaries:
        curvature.CurvatureOnBoundaries = 1
    else:
        curvature.CurvatureOnBoundaries = 0
    if bounded_reciporcal:
        curvature.BoundedReciporcal = 1
    else:
        curvature.BoundedReciporcal = 0
    curvature.Epsilon = epsilon
    curvature.Offset = offset

    curvature.Execute()

    return curvature.Surface



def vmtk_surface_distance(surface1, surface2, distance_array_name="Distance",
                          distance_vectors_array_name="",
                          signed_distance_array_name="", flip_normals=False):
    """
    Compute the point-wise minimum distance of the input surface from a reference surface

    Args:
        surface1 (vtkPolyData): Input surface
        surface2 (vtkPolyData): Reference surface
        distance_array_name (str): Name of distance array
        distance_vectors_array_name (str): Name of distance array (of vectors)
        signed_distance_array_name (str): Name of distance arrays signed as positive or negative
        flip_normals (bool): Flip normals relative to reference surface

    Returns:
        surface (vtkPoyData): Output surface with distance info
    """
    distance = vmtkscripts.vmtkSurfaceDistance()
    distance.Surface = surface1
    distance.ReferenceSurface = surface2
    distance.DistanceArrayName = distance_array_name
    distance.DistanceVectorsArrayname = distance_vectors_array_name
    distance.SignedDistanceArrayName = signed_distance_array_name
    if flip_normals:
        distance.FlipNormals = 1
    else:
        distance.FlipNormals = 0

    distance.Execute()

    return distance.Surface


def vmtkcenterlineattributes(centerlines):
    """Compute centerline attributes.

    Args:
        centerlines: Centerlines.

    Returns:
        Centerlines with attributes.
        Pointdata:
            MaximumInscribedSphereRadius: If the point on the centerline is the
                center of a sphere, this is the radius of the largest possible
                sphere that does not intersect the surface.
            Abscissas: Position along the centerlines. By default, the abscissa
                is measured from the start of the centerlines.
            ParallelTransportNormals: 'Normal' of the centerlines (perpendicular
                to centerline direction).

    """
    clattributes = vmtkscripts.vmtkCenterlineAttributes()
    clattributes.Centerlines = centerlines
    clattributes.Execute()
    return clattributes.Centerlines
def vmtkbifurcationreferencesystems(centerlines):
    """Compute reference system for each bifurcation of a vessel tree.

    Args:
        centerlines: Centerlines split into branches.

    Returns:
        Reference system for each bifurcation.
        Pointdata (selection):
            Normal: Normal of the bifurcation plane.
            UpNormal: Normal pointing toward the bifurcation apex.

    """
    bifrefsystem = vmtkscripts.vmtkBifurcationReferenceSystems()
    bifrefsystem.Centerlines = centerlines
    bifrefsystem.RadiusArrayName = 'MaximumInscribedSphereRadius'
    bifrefsystem.BlankingArrayName = 'Blanking'
    bifrefsystem.GroupIdsArrayName = 'GroupIds'
    bifrefsystem.Execute()
    return bifrefsystem.ReferenceSystems

def vmtkcenterlinemodeller(centerlines, size=[64, 64, 64]):
    """Convert a centerline to an image containing the tube function.

    Args:
        centerlines: Centerlines.
        size: Image dimensions.

    Returns:
        Signed distance transform image, with the zero level set being (tapered)
        tubes running from one centerline point to the next with a radius at
        each end corresponding to the local MaximumInscribedSphereRadius.

    """
    modeller = vmtkscripts.vmtkCenterlineModeller()
    modeller.Centerlines = centerlines
    modeller.RadiusArrayName = 'MaximumInscribedSphereRadius'
    modeller.SampleDimensions = size
    modeller.Execute()
    return modeller.Image

def vmtkmarchingcubes(image, level=0.0):
    """Generate an isosurface of given level from a 3D image.

    Args:
        image: vtkImageData object.
        level: Graylevel at which to generate the isosurface.

    Returns:
        Surface mesh of the isosurface.

    """
    marcher = vmtkscripts.vmtkMarchingCubes()
    marcher.Image = image
    marcher.Level = level
    marcher.Execute()
    return marcher.Surface



def vmtkbifurcationsections(surface, centerlines, distance=1):
    """Compute sections located a fixed number of maximally inscribed
    sphere radii away from each bifurcation.

    Args:
        surface: Surface split into branches.
        centerlines: Centerlines split into branches.
        distance: Distance from bifurcation in number of maximally inscribed
            spheres, where each sphere touches the center of the previous one.

    Returns:
        Polydata with one cross section per branch of each bifurcation.
        Celldata (selection):
            BifurcationSectionArea: Section area.
            BifurcationSectionMinSize: Minimum diameter of section.
            BifurcationSectionMaxSize: Maximum diameter of section.

    """
    bifsections = vmtkscripts.vmtkBifurcationSections()
    bifsections.Surface = surface
    bifsections.Centerlines = centerlines
    bifsections.NumberOfDistanceSpheres = distance
    bifsections.RadiusArrayName = 'MaximumInscribedSphereRadius'
    bifsections.GroupIdsArrayName = 'GroupIds'
    bifsections.CenterlineIdsArrayName = 'CenterlineIds'
    bifsections.TractIdsArrayName = 'TractIds'
    bifsections.BlankingArrayName = 'Blanking'
    bifsections.Execute()
    return bifsections.BifurcationSections

def vmtk_surface_clip(centerline, surface, inlet, outlet):
    """
    Wrapper for vmtkCenterlines.
    compute centerlines from a branching tubular surface. Seed points can be interactively selected on the surface,
    or specified as the barycenters of the open boundaries of the surface.

    Args:
        end_point (int): Toggle append open profile barycenters to centerlines
        surface (vktPolyData): Surface model
        voronoi (vtkPolyData): Voronoi diagram based on previous centerlines (Optional)
        inlet (ndarray): List of source point coordinates
        method (str): Seed point selection method
        outlet (ndarray): List of target point coordinates
        pole_ids (ndarray): Pole ID list of Voronoi diagram (Optional)
        resampling_step (float): Resampling step
        flip_normals (float): Flip normals after outward normal computation
        cap_displacement (float): Displacement of the center points of caps at open profiles along their normals
        delaunay_tolerance (float): Tolerance for evaluating coincident points during Delaunay tessellation
        simplify_voronoi (bool): Toggle simplification of Voronoi diagram

    Returns:

    """
    surface_clip = vmtkscripts.vmtkSurfaceEndClipper()
    surface_clip.Surface = surface
    surface_clip.Centerlines = centerline
    surface_clip.SourcePoints = inlet
    surface_clip.TargetPoints = outlet
    surface_clip.Execute()
    return surface_clip.Surface

def vmtkcenterlinegeometry(centerlines, smoothing=0, iterations=100):
    """Compute the local geometry of centerlines.

    Args:
        centerlines: Centerlines.
        smoothing (bool): Laplacian smooth centerlines before computing
            geometric variables.
        iterations: Number of smoothing iterations.

    Returns:
        Centerlines with geometric variables defined at each point.
        Pointdata (selection):
            Curvature: Local curvature.
            Torsion: Local torsion.
        Celldata (selection):
            Tortuosity: Tortuosity of each centerline.
            Length: Length of each centerline.

    Note:
        Since the computation of the geometric variables depends on first,
        second and third derivatives of the line coordinates, and since such
        derivatives are approximated using a simple finite difference scheme
        along the line, it is very likely that such derivatives will be affected
        by noise that is not appreciable when looking at the line itself. For
        this reason, it might be necessary to run the Laplacian smoothing filter
        before computing the derivatives and the related quantities.

    """
    clgeometry = vmtkscripts.vmtkCenterlineGeometry()
    clgeometry.Centerlines = centerlines
    clgeometry.LineSmoothing = smoothing
    clgeometry.NumberOfSmoothingIterations = iterations
    clgeometry.Execute()
    return clgeometry.Centerlines


def vmtksurfacecurvature(surface, curvature_type='mean', absolute_curvature=0,
                         median_filtering=0):
    """Compute curvature of an input surface.

    Args:
        surface: Surface mesh.
        curvature_type ('mean', 'gaussian', 'maximum', 'minimum'): Type of
            curvature to compute.
        absolute_curvature (bool): Output the absolute value of curvature.
        median_filtering (bool): Output curvature after median filtering to
            suppress numerical noise speckles.

    Returns:
        Surface with curvature variable.
        Pointdata:
            Curvature: Local surface curvature.

    """
    curvaturefilter = vmtkscripts.vmtkSurfaceCurvature()
    curvaturefilter.Surface = surface
    curvaturefilter.CurvatureType = curvature_type
    curvaturefilter.AbsoluteCurvature = absolute_curvature
    curvaturefilter.MedianFiltering = median_filtering
    curvaturefilter.Execute()
    return curvaturefilter.Surface

def openSurfaceAtPoint(self, polyData, seed):
       '''
       Returns a new surface with an opening at the given seed.
       '''

       someradius = 1.0

       pointLocator = vtk.vtkPointLocator()
       pointLocator.SetDataSet(polyData)
       pointLocator.BuildLocator()

       # find the closest point next to the seed on the surface
       # id = pointLocator.FindClosestPoint(int(seed[0]),int(seed[1]),int(seed[2]))
       id = pointLocator.FindClosestPoint(seed)

       # the seed is now guaranteed on the surface
       seed = polyData.GetPoint(id)

       sphere = vtk.vtkSphere()
       sphere.SetCenter(seed[0], seed[1], seed[2])
       sphere.SetRadius(someradius)

       clip = vtk.vtkClipPolyData()
       clip.SetInputData(polyData)
       clip.SetClipFunction(sphere)
       clip.Update()

       outPolyData = vtk.vtkPolyData()
       outPolyData.DeepCopy(clip.GetOutput())

       return outPolyData
   
    
def closest_point(point, centerline):
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    point_c = locator.FindClosestPoint(point)
    # Initialize variables to store the closest point and its associated cell_id and sub_id
    return centerline.GetPoint(point_c)

def centerline_viewer(centerline, array_name=None, renderer=None):
    viewer = vmtkscripts.vmtkCenterlineViewer()
    viewer.Centerlines = centerline
    viewer.ColorMap = 'cooltowarm'
    if array_name:
        viewer.CellDataArrayName = array_name
    if renderer:
        viewer.Display = 0
        viewer.vmtkRenderer = renderer.vmtkRenderer

    viewer.Execute()

def surface_viewer(surface, array_name=None, renderer=None, opacity=1):
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.ColorMap = 'rainbow'
    if renderer:
        viewer.vmtkRenderer = renderer.vmtkRenderer
        viewer.Display = 0
    viewer.Surface = surface
    viewer.Opacity = opacity
    if array_name:
        viewer.ArrayName = array_name
    viewer.Execute() 

def centerline_numpy(centerline):
    cn = vmtkscripts.vmtkCenterlinesToNumpy()    
    cn.Centerlines = centerline
    cn.Execute()
    return cn.ArrayDict

def surface_numpy(surface):
    s = vmtkscripts.vmtkSurfaceToNumpy()    
    s.Surface = surface
    s.Execute()
    return s.ArrayDict

def reader(filename_vtp):
    # Create a VTK XML PolyDataReader
    reader = vtk.vtkXMLPolyDataReader() 
    reader.SetFileName(filename_vtp)
    reader.Update()
    # Get the PolyData from the reader's output
    polydata = reader.GetOutput()
    return polydata

def vmtkpointsplitextractor(centerlines, splitpoint, gap=1.0):
    """Split centerlines at specified location.

    Args:
        centerlines: Centerlines.
        splitpoint: Location where to split the centerlines.
        gap: Length of 'Blanking=1' part of the centerlines.

    Returns
        Centerlines split at splitpoint, with the center of the gap at the
        splitpoint. The output is similar to the output of
        vmtkbranchextractor.
        Celldata (selection):
            CenterlineId: Cellid of centerline from which the tract was split.
            TractId: Id identifying each tract along one centerline.
            GroupId: Id of the group to which the tract belongs.
            Blanking: Boolean indicating whether tract belongs to bifurcation.

    """
    extractor = vmtkscripts.vmtkPointSplitExtractor()
    extractor.Centerlines = centerlines
    extractor.RadiusArrayName = 'MaximumInscribedSphereRadius'
    extractor.GroupIdsArrayName = 'GroupIds'
    extractor.SplitPoint = splitpoint
    extractor.Execute()
    return extractor.Centerlines

def vmtkmarchingcubes(image, level=0.0):
    """Generate an isosurface of given level from a 3D image.

    Args:
        image: vtkImageData object.
        level: Graylevel at which to generate the isosurface.

    Returns:
        Surface mesh of the isosurface.

    """
    marcher = vmtkscripts.vmtkMarchingCubes()
    marcher.Image = image
    marcher.Level = level
    marcher.Execute()
    return marcher.Surface

def vmtkmeshtosurface(mesh, cleanoutput=1):
    """Convert a mesh to a surface by throwing out volume elements and (optionally) the relative points

    Args:
        mesh: Volumetric mesh.
        cleanoutput (bool): Remove unused points.

    Returns:
        vtkPolyData object.

    """
    extractor = vmtkscripts.vmtkMeshToSurface()
    extractor.Mesh = mesh
    extractor.CleanOutput = cleanoutput
    extractor.Execute()
    return extractor.Surface

def vmtksurfacesmoothing(surface, iterations=100, method='taubin'):
    """Smooth a surface.

    Args:
        surface: Surface mesh.
        iterations: Number of smoothing iterations.
        method ('taubin', 'laplace'): Taubin's volume-preserving or a Laplacian
            smoothing filter.

    Returns:
        Smoothed surface.

    """
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method = method
    smoother.NumberOfIterations = iterations
    smoother.PassBand = 0.1
    smoother.Execute()
    return smoother.Surface

def vmtksurfacenormals(surface):
    """Compute normals to a surface.

    Args:
        surface: Surface mesh.

    Returns:
        Surface mesh with 'Normals' vector pointdata.

    """
    normaller = vmtkscripts.vmtkSurfaceNormals()
    normaller.Surface = surface
    normaller.Execute()
    return normaller.Surface


def vmtksurfaceprojection(surface, referencesurface):
    """Interpolates the pointdata of a reference surface onto a surface based
    on minimum distance criterion.

    Args:
        surface: Surface mesh.
        referencesurface: Reference surface mesh.

    Returns:
        'surface' with projected pointdata from 'referencesurface'.

    """
    projector = vmtkscripts.vmtkSurfaceProjection()
    projector.Surface = surface
    projector.ReferenceSurface = referencesurface
    projector.Execute()
    return projector.Surface

def vmtksurfacecurvature(surface, curvature_type='mean', absolute_curvature=0,
                         median_filtering=0):
    """Compute curvature of an input surface.

    Args:
        surface: Surface mesh.
        curvature_type ('mean', 'gaussian', 'maximum', 'minimum'): Type of
            curvature to compute.
        absolute_curvature (bool): Output the absolute value of curvature.
        median_filtering (bool): Output curvature after median filtering to
            suppress numerical noise speckles.

    Returns:
        Surface with curvature variable.
        Pointdata:
            Curvature: Local surface curvature.

    """
    curvaturefilter = vmtkscripts.vmtkSurfaceCurvature()
    curvaturefilter.Surface = surface
    curvaturefilter.CurvatureType = curvature_type
    curvaturefilter.AbsoluteCurvature = absolute_curvature
    curvaturefilter.MedianFiltering = median_filtering
    curvaturefilter.Execute()
    return curvaturefilter.Surface

def vmtksurfacecapper(surface):
    """Caps the holes of a surface.

    Args:
        surface: Surface mesh of vascular geometry with holes at inlets and
            outlets.

    Returns:
        Surface mesh with capped holes. Each cap has an ID assigned for easy
        specification of boundary conditions.
        Celldata:
            CellEntityIds: ID assigned to caps.

    """
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Method = 'centerpoint'
    capper.Interactive = 0
    capper.Execute()
    return capper.Surface

def surface_connectivity(surface):
    conn = vmtkscripts.vmtkSurfaceConnectivity()
    conn.Surface = surface
    conn.Execute()
    return conn.Surface

def vmtkmeshaddexternallayer(mesh):
    ad = vmtkscripts.vmtkMeshAddExternalLayer()
    ad.Mesh = mesh
    ad.Execute()    
    return ad.Mesh

def dijkstradistance(surface, seedpoints):
    dd = vmtkscripts.vmtkDijkstraDistanceToPoints()
    dd.Surface = surface
    dd.seedpoints = seedpoints
    dd.Execute()
    return dd
    
