import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from itertools import combinations
import trimesh
from align_meshes import align_tgt_to_ref_meshes, load_mesh, print_verbose
import copy
from scipy.spatial import cKDTree
import pandas as pd
import os
import fixture_plane_fitting
import time

def create_plane_mesh(plane_model, inlier_cloud, plane_size=20.0, color=None):
    # Create a square plane oriented by the plane normal
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Get centroid of inlier points
    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)

    # Build local coordinate system (normal + two orthogonal in-plane axes)
    z = normal
    if np.allclose(z, [0, 0, 1]):
        x = np.array([1, 0, 0])
    else:
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
    y = np.cross(z, x)

    # Build the 4 corners of the plane
    hw = plane_size / 2.0
    corners = [
        centroid + hw * x + hw * y,
        centroid - hw * x + hw * y,
        centroid - hw * x - hw * y,
        centroid + hw * x - hw * y,
    ]
    corners = np.array(corners)

    # Create mesh from corners
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color if color else np.random.rand(3))
    return mesh

def angular_change_between_normals(n_orig, n_opt):
    n_orig = n_orig / np.linalg.norm(n_orig)
    n_opt = n_opt / np.linalg.norm(n_opt)
    dot_product = np.clip(np.dot(n_orig, n_opt), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def identify_planes_along_x(planes):
    """
    planes: list of plane coefficients
    
    Returns indices of planes in order from min X to max X.
    So returns 4 indices: outer_min, inner_min, inner_max, outer_max
    """
    
    def project_origin_to_plane(plane):
        normal = plane[0:3]
        normal_length = np.linalg.norm(normal)
        distance = plane[3] / normal_length
        unit_normal = normal / normal_length
        projected_point = -distance * unit_normal
        return projected_point, -distance
    
    x_coords = []
    for plane in planes:
        projected_point, x_coord = project_origin_to_plane(plane)
        x_coords.append(x_coord)
    
    # Sort indices by X coordinate
    sorted_indices = sorted(range(len(x_coords)), key=lambda i: x_coords[i])
    
    # Return all four sorted indices
    return tuple(sorted_indices)  # will be 4 indices

def normalize(v):
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)

def angle_between_plane_normals(plane1, plane2):
    """
    plane1, plane2: array-like of 4 values each [a, b, c, d]
    
    Returns angle in degrees between the two plane normals.
    """
    n1 = np.array(plane1[:3])
    n2 = np.array(plane2[:3])
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    angle_rad = np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0))
    return np.degrees(angle_rad)

def optimize_constrained_planes(planes, clouds, bound_factor=2.0, constraint_tolerance=1e-6, verbose=False):
    """
    Optimize plane orientations to satisfy geometric constraints while minimizing fitting error.
    
    Parameters:
    -----------
    planes : list of array-like
        List of 3 planes [a, b, c, d] where planes[0] should be orthogonal to planes[1,2],
        and planes[1,2] should be parallel
    clouds : list of array-like  
        List of 3 point clouds, each Nx3 array corresponding to each plane
    constraint_factor : float
        Multiplier for constraint tolerance based on initial misalignment
    max_rotation_deg : float
        Maximum allowed rotation in degrees for any axis
        
    Returns:
    --------
    optimized_planes : list of arrays
        The optimized plane parameters [a, b, c, d]
    result : dict
        Optimization results and diagnostics
    """
    
    # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    # Get original planes and point clouds
    base_plane = planes[0]
    outer_plane1 = planes[x_plane_indices[0]]
    outer_plane2 = planes[x_plane_indices[3]]
    
    base_cloud = clouds[0]
    outer_cloud1 = clouds[x_plane_indices[0]]
    outer_cloud2 = clouds[x_plane_indices[3]]
    
    optimize_planes = [base_plane, outer_plane1, outer_plane2]
    optimize_clouds = [base_cloud, outer_cloud1, outer_cloud2]
    
    # Convert inputs to numpy arrays and normalize normals
    optimize_planes = [np.array(p) for p in optimize_planes]
    optimize_clouds = [np.array(c) for c in optimize_clouds]
    
    # Extract and normalize initial normals
    initial_normals = []
    initial_d = []
    for plane in optimize_planes:
        normal = plane[:3]
        normal = normal / np.linalg.norm(normal)
        initial_normals.append(normal)
        initial_d.append(plane[3])
    
    initial_normals = np.array(initial_normals)
    initial_d = np.array(initial_d)
    
    # Calculate initial constraint violations to set tolerance
    angle_01 = angle_between_plane_normals(optimize_planes[0], optimize_planes[1])
    angle_02 = angle_between_plane_normals(optimize_planes[0], optimize_planes[2]) 
    angle_12 = angle_between_plane_normals(optimize_planes[1], optimize_planes[2])
    
    # Find maximum deviation from intended constraints (90°, 90°, 0°)
    max_deviation = max(abs(angle_01 - 90), abs(angle_02 - 90), abs(angle_12 - 0))
    max_deviation_rad = np.radians(max_deviation)
    
    # constraint_tolerance = constraint_factor * max_deviation

    # Set up optimization bounds (limit rotations to reasonable range)
    bounds = [(-bound_factor*max_deviation_rad, bound_factor*max_deviation_rad)] * 6  # 2 rotation params × 3 planes
    
    print_verbose(f"\n\nInitial angles: {angle_01:.2f}°, {angle_02:.2f}°, {angle_12:.2f}°", verbose)
    print_verbose(f"Max deviation: {max_deviation:.2f}°, Constraint tolerance: {constraint_tolerance:.2f}°", verbose)
    
    def rodrigues_rotation(axis, angle):
        """Apply Rodrigues' rotation formula"""
        if np.abs(angle) < 1e-10:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    
    def apply_rotation_perturbation(normal, rot_params):
        """Apply small rotation to normal vector using two rotation angles"""
        # Create two perpendicular axes in plane orthogonal to normal
        if abs(normal[2]) < 0.9:
            axis1 = np.cross(normal, [0, 0, 1])
        else:
            axis1 = np.cross(normal, [1, 0, 0])
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = np.cross(normal, axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        
        # Apply rotations around both axes
        R1 = rodrigues_rotation(axis1, rot_params[0])
        R2 = rodrigues_rotation(axis2, rot_params[1])
        
        return (R2 @ R1 @ normal.reshape(-1, 1)).flatten()
    
    def get_perturbed_normals(rotation_params):
        """Get normals after applying rotation perturbations"""
        normals = []
        for i in range(3):
            rot_params = rotation_params[2*i:2*i+2]
            perturbed = apply_rotation_perturbation(initial_normals[i], rot_params)
            normals.append(perturbed / np.linalg.norm(perturbed))
        return np.array(normals)
    
    def compute_optimal_d(normals):
        """Compute optimal d parameters for given normals (inner optimization)"""
        optimal_d = []
        for i, (normal, cloud) in enumerate(zip(normals, optimize_clouds)):
            if cloud.size > 0 and cloud.ndim >= 2:
                # Optimal d is mean signed distance
                d_opt = np.mean(np.dot(cloud, normal))
                optimal_d.append(d_opt)
            else:
                optimal_d.append(initial_d[i])
        return np.array(optimal_d)
    
    def compute_fitting_error(normals, d_values):
        """Compute total fitting error across all point clouds"""
        total_error = 0.0
        for normal, d, cloud in zip(normals, d_values, optimize_clouds):
            if cloud.size > 0 and cloud.ndim >= 2:
                # Point-to-plane distance
                distances = np.abs(np.dot(cloud, normal) - d)
                total_error += np.sum(distances**2)
        return total_error
    
    def objective_function(rotation_params):
        """Outer optimization objective: minimize fitting error with constraints"""
        
        # Get perturbed normals
        normals = get_perturbed_normals(rotation_params)
        
        # Compute optimal d values (inner optimization - closed form)
        d_values = compute_optimal_d(normals)
        
        # Compute fitting error
        fitting_error = compute_fitting_error(normals, d_values)
        
        # Compute actual angles between planes
        angle_01_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[0], normals[1])), 0.0, 1.0)))
        angle_02_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[0], normals[2])), 0.0, 1.0)))
        angle_12_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[1], normals[2])), 0.0, 1.0)))
        
        # Compute constraint violations in degrees
        violation_01 = max(0, abs(angle_01_actual - 90.0) - constraint_tolerance)
        violation_02 = max(0, abs(angle_02_actual - 90.0) - constraint_tolerance)
        violation_12 = max(0, abs(angle_12_actual - 0.0) - constraint_tolerance)
        
        # Convert to penalty (large penalty factor to enforce constraints)
        constraint_penalty = 1e6 * (violation_01**2 + violation_02**2 + violation_12**2)
        
        return fitting_error + constraint_penalty
    
    
    # Initial guess (no rotation)
    x0 = np.zeros(6)
    
    # Run optimization
    print_verbose("Running constrained optimization...", verbose)
    result = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 50}
    )
    
    if not result.success:
        print_verbose(f"Warning: Optimization did not converge. Message: {result.message}", verbose)
    
    # Extract final results
    final_normals = get_perturbed_normals(result.x)
    final_d = compute_optimal_d(final_normals)
    
    # Create optimized planes
    optimized_planes = []
    for normal, d in zip(final_normals, final_d):
        optimized_planes.append(np.array([normal[0], normal[1], normal[2], d]))
    
    # Compute final angles for verification
    final_angle_01 = angle_between_plane_normals(optimized_planes[0], optimized_planes[1])
    final_angle_02 = angle_between_plane_normals(optimized_planes[0], optimized_planes[2])
    final_angle_12 = angle_between_plane_normals(optimized_planes[1], optimized_planes[2])
    
    # Calculate rotation magnitudes
    rotation_magnitudes = []
    for i in range(3):
        rot_params = result.x[2*i:2*i+2]
        mag = np.degrees(np.linalg.norm(rot_params))
        rotation_magnitudes.append(mag)
    
    # Prepare results dictionary
    optimization_results = {
        'success': result.success,
        'message': result.message,
        'final_objective': result.fun,
        'iterations': result.nit,
        'initial_angles': [angle_01, angle_02, angle_12],
        'final_angles': [final_angle_01, final_angle_02, final_angle_12],
        'rotation_magnitudes_deg': rotation_magnitudes,
        'constraint_tolerance': constraint_tolerance,
        'rotation_params': result.x
    }
    
    print_verbose(f"Final angles: {final_angle_01:.2f}°, {final_angle_02:.2f}°, {final_angle_12:.2f}°", verbose)
    print_verbose(f"Applied rotations: {rotation_magnitudes[0]:.3f}°, {rotation_magnitudes[1]:.3f}°, {rotation_magnitudes[2]:.3f}°", verbose)
    print_verbose(f"Final objective value: {result.fun:.6f}", verbose)
    
    # Insert the elements of keep_planes that were not included in the optimization, in their original positions
    optimized_planes.append(planes[x_plane_indices[1]])
    optimized_planes.append(planes[x_plane_indices[2]])
    optimized_planes.append(planes[5])
    optimized_planes.append(planes[6])
    
    print_verbose("\n[INFO] Orientation changes:", verbose)
    for i, (original, optimized) in enumerate(zip(planes, optimized_planes)):
        orig_normal = original[:3]
        opt_normal = optimized[:3]
        
        cos_angle = np.clip(np.dot(orig_normal, opt_normal), -1, 1)
        angle_change = np.degrees(np.arccos(abs(cos_angle)))
        
        print_verbose(f"Plane {i}: {angle_change:.3f}° change", verbose)
    
    # # Verify anvil angles
    # print_verbose("\n[INFO] Anvil plane angle verification:")
    # current_base_normal = optimized_planes[0][:3]
    
    # Create plane meshes
    optimized_plane_meshes = []
    for plane, cloud in zip(optimized_planes, clouds):
        mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
        optimized_plane_meshes.append(mesh)
    
    # return optimized_planes, optimization_results
    return optimized_planes, optimization_results, x_plane_indices, optimized_plane_meshes

def optimize_anvil_planes(planes, clouds, bound_factor=2.0, constraint_tolerance=1e-6, verbose=False):
    # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    base_plane = planes[0]
    inner_plane1 = planes[x_plane_indices[1]]
    inner_plane2 = planes[x_plane_indices[2]]
    
    base_cloud = clouds[0]
    inner_cloud1 = clouds[x_plane_indices[1]]
    inner_cloud2 = clouds[x_plane_indices[2]]
    
    optimize_planes = [inner_plane1, inner_plane2]
    optimize_clouds = [inner_cloud1, inner_cloud2]
    
    # Convert inputs to numpy arrays and normalize normals
    optimize_planes = [np.array(p) for p in optimize_planes]
    optimize_clouds = [np.array(c) for c in optimize_clouds]
    
    # Extract and normalize initial normals
    initial_normals = []
    initial_d = []
    for plane in optimize_planes:
        normal = plane[:3]
        normal = normal / np.linalg.norm(normal)
        initial_normals.append(normal)
        initial_d.append(plane[3])
    
    initial_normals = np.array(initial_normals)
    initial_d = np.array(initial_d)
    
    base_normal = base_plane[:3]
    base_normal = base_normal / np.linalg.norm(base_normal)
    
    # Calculate initial constraint violations to set tolerance
    angle_01 = angle_between_plane_normals(base_plane, optimize_planes[0])
    angle_02 = angle_between_plane_normals(base_plane, optimize_planes[1]) 
    angle_12 = angle_between_plane_normals(optimize_planes[0], optimize_planes[1])
    
    # Find maximum deviation from intended constraints (90°, 90°, 0°)
    max_deviation = max(abs(angle_01 - 90), abs(angle_02 - 90), abs(angle_12 - 0))
    max_deviation_rad = np.radians(max_deviation)

    # Set up optimization bounds (limit rotations to reasonable range)
    bounds = [(-bound_factor*max_deviation_rad, bound_factor*max_deviation_rad)] * 4  # 2 rotation params × 2 planes
    
    print_verbose(f"\n\nInitial angles: {angle_01:.2f}°, {angle_02:.2f}°, {angle_12:.2f}°", verbose)
    print_verbose(f"Max deviation: {max_deviation:.2f}°, Constraint tolerance: {constraint_tolerance:.2f}°", verbose)
    
    def rodrigues_rotation(axis, angle):
        """Apply Rodrigues' rotation formula"""
        if np.abs(angle) < 1e-10:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    
    def apply_rotation_perturbation(normal, rot_params):
        """Apply small rotation to normal vector using two rotation angles"""
        # Create two perpendicular axes in plane orthogonal to normal
        if abs(normal[2]) < 0.9:
            axis1 = np.cross(normal, [0, 0, 1])
        else:
            axis1 = np.cross(normal, [1, 0, 0])
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = np.cross(normal, axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        
        # Apply rotations around both axes
        R1 = rodrigues_rotation(axis1, rot_params[0])
        R2 = rodrigues_rotation(axis2, rot_params[1])
        
        return (R2 @ R1 @ normal.reshape(-1, 1)).flatten()
    
    def get_perturbed_normals(rotation_params):
        """Get normals after applying rotation perturbations"""
        normals = []
        for i in range(2):
            rot_params = rotation_params[2*i:2*i+2]
            perturbed = apply_rotation_perturbation(initial_normals[i], rot_params)
            normals.append(perturbed / np.linalg.norm(perturbed))
        return np.array(normals)
    
    def compute_optimal_d(normals):
        """Compute optimal d parameters for given normals (inner optimization)"""
        optimal_d = []
        for i, (normal, cloud) in enumerate(zip(normals, optimize_clouds)):
            if cloud.size > 0 and cloud.ndim >= 2:
                # Optimal d is mean signed distance
                d_opt = np.mean(np.dot(cloud, normal))
                optimal_d.append(d_opt)
            else:
                optimal_d.append(initial_d[i])
        return np.array(optimal_d)
    
    def compute_fitting_error(normals, d_values):
        """Compute total fitting error across all point clouds"""
        total_error = 0.0
        for normal, d, cloud in zip(normals, d_values, optimize_clouds):
            if cloud.size > 0 and cloud.ndim >= 2:
                # Point-to-plane distance
                distances = np.abs(np.dot(cloud, normal) - d)
                total_error += np.sum(distances**2)
        return total_error
    
    def objective_function(rotation_params):
        """Outer optimization objective: minimize fitting error with constraints"""
        
        # Get perturbed normals
        normals = get_perturbed_normals(rotation_params)
        
        # Compute optimal d values (inner optimization - closed form)
        d_values = compute_optimal_d(normals)
        
        # Compute fitting error
        fitting_error = compute_fitting_error(normals, d_values)
        
        # Compute actual angles between planes
        angle_01_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(base_normal, normals[0])), 0.0, 1.0)))
        angle_02_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(base_normal, normals[1])), 0.0, 1.0)))
        angle_12_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[0], normals[1])), 0.0, 1.0)))
        
        # angle_01_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[0], normals[1])), 0.0, 1.0)))
        # angle_02_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[0], normals[2])), 0.0, 1.0)))
        # angle_12_actual = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[1], normals[2])), 0.0, 1.0)))
        
        # Compute constraint violations in degrees
        violation_01 = max(0, abs(angle_01_actual - 90.0) - constraint_tolerance)
        violation_02 = max(0, abs(angle_02_actual - 90.0) - constraint_tolerance)
        violation_12 = max(0, abs(angle_12_actual - 0.0) - constraint_tolerance)
        
        # Convert to penalty (large penalty factor to enforce constraints)
        constraint_penalty = 1e6 * (violation_01**2 + violation_02**2 + violation_12**2)
        
        return fitting_error + constraint_penalty
    
    
    # Initial guess (no rotation)
    x0 = np.zeros(4)
    
    # Run optimization
    print_verbose("Running constrained optimization...", verbose)
    result = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 50}
    )
    
    if not result.success:
        print_verbose(f"Warning: Optimization did not converge. Message: {result.message}", verbose)
        
    # Extract final results
    final_normals = get_perturbed_normals(result.x)
    final_d = compute_optimal_d(final_normals)
    
    # Create optimized planes
    optimized_planes = []
    for normal, d in zip(final_normals, final_d):
        optimized_planes.append(np.array([normal[0], normal[1], normal[2], d]))
    
    # Compute final angles for verification
    final_angle_01 = angle_between_plane_normals(base_plane, optimized_planes[0])
    final_angle_02 = angle_between_plane_normals(base_plane, optimized_planes[1])
    final_angle_12 = angle_between_plane_normals(optimized_planes[0], optimized_planes[1])
    
    # Calculate rotation magnitudes
    rotation_magnitudes = []
    for i in range(3):
        rot_params = result.x[2*i:2*i+2]
        mag = np.degrees(np.linalg.norm(rot_params))
        rotation_magnitudes.append(mag)
    
    # Prepare results dictionary
    optimization_results = {
        'success': result.success,
        'message': result.message,
        'final_objective': result.fun,
        'iterations': result.nit,
        'initial_angles': [angle_01, angle_02, angle_12],
        'final_angles': [final_angle_01, final_angle_02, final_angle_12],
        'rotation_magnitudes_deg': rotation_magnitudes,
        'constraint_tolerance': constraint_tolerance,
        'rotation_params': result.x
    }
        
    print_verbose(f"Final angles: {final_angle_01:.2f}°, {final_angle_02:.2f}°, {final_angle_12:.2f}°", verbose)
    print_verbose(f"Applied rotations: {rotation_magnitudes[0]:.3f}°, {rotation_magnitudes[1]:.3f}°, {rotation_magnitudes[2]:.3f}°", verbose)
    print_verbose(f"Final objective value: {result.fun:.6f}", verbose)
    
    # Extract final results
    final_normals = get_perturbed_normals(result.x)
    final_d = compute_optimal_d(final_normals)
    
    # Create optimized planes
    optimized_planes = [planes[0],
                        planes[x_plane_indices[0]],
                        planes[x_plane_indices[3]]]
    for normal, d in zip(final_normals, final_d):
        optimized_planes.append(np.array([normal[0], normal[1], normal[2], d]))
    optimized_planes.append(planes[5])
    optimized_planes.append(planes[6])
        
    return optimized_planes, optimization_results

def separation_between_parallel_planes(plane1, plane2):
    """
    Computes the perpendicular distance between two parallel planes.
    Planes must be in the form [a, b, c, d] for ax + by + cz + d = 0.
    """
    n1 = plane1[:3]
    d1 = plane1[3]
    d2 = plane2[3]

    # # Normalize the normal vector
    # n1_norm = n1 / np.linalg.norm(n1)

    # Compute signed distance from origin to each plane
    # Signed distance = -d / ||n||
    dist1 = -d1 / np.linalg.norm(n1)
    dist2 = -d2 / np.linalg.norm(n1)

    # Return the absolute separation between the planes
    return abs(dist1 - dist2)

def create_supports_relative_to_planes(
    plane1, plane2, plane3,
    support_offset, base_offset,
    diameter, height
):
    """
    Create two cylinders positioned relative to three planes.
    The cylinders will be positioned such that:
    - Both have axes parallel to the Y axis
    - Both have the same Z coordinate, which is base_offset above plane3
    - One cylinder is offset along +X from plane1 by support_offset
    - The other cylinder is offset along -X from plane2 by support_offset
    
    Parameters:
    - plane1, plane2: 4-element numpy arrays [a, b, c, d] for ax + by + cz + d = 0
                      These should be parallel to each other and roughly aligned to X
    - plane3: 4-element numpy array [a, b, c, d] for ax + by + cz + d = 0
              This should be aligned with the Z axis
    - support_offset: Distance to offset cylinders from plane1 (+X) and plane2 (-X)
    - base_offset: Distance above plane3 to position the cylinders
    - diameter: Cylinder diameter
    - height: Cylinder height
    
    Returns:
    - tuple: (cylinder1, cylinder2) - trimesh.Trimesh cylinder meshes
    """
    import numpy as np
    import trimesh
    
    # Extract and normalize normals
    n1 = plane1[:3] / np.linalg.norm(plane1[:3])
    n2 = plane2[:3] / np.linalg.norm(plane2[:3])
    n3 = plane3[:3] / np.linalg.norm(plane3[:3])
    
    # Find the Z coordinate for both cylinders (base_offset above plane3)
    # For plane3 (Z-aligned): ax + by + cz + d = 0, so z = -(ax + by + d)/c
    # We want the point that's base_offset above the plane
    # Distance from point to plane is |ax + by + cz + d|/sqrt(a²+b²+c²)
    # For a point at (x, y, z), distance to plane3 is |n3·[x,y,z] + plane3[3]|
    
    # We want: n3·[x,y,z] + plane3[3] = base_offset (assuming n3 points "up")
    # So: z = (base_offset - plane3[3] - n3[0]*x - n3[1]*y) / n3[2]
    
    # For Z-aligned plane3, n3 should be approximately [0, 0, ±1]
    # We'll determine the sign and calculate accordingly
    
    # Find reference Z coordinate at x=0, y=0
    if abs(n3[2]) > 0.001:  # Plane3 is roughly Z-aligned
        # Distance from origin to plane3
        plane3_z_at_origin = -plane3[3] / n3[2]
        # Adjust for the desired offset in the direction of the normal
        cylinder_z = plane3_z_at_origin + base_offset * np.sign(n3[2])
    else:
        # Fallback if plane3 is not Z-aligned
        cylinder_z = base_offset
    
    # Find X coordinates for the cylinders
    # Cylinder 1: offset along +X from plane1 by support_offset
    # Cylinder 2: offset along -X from plane2 by support_offset
    
    # For plane1, find where it intersects at y=0, z=cylinder_z
    # plane1: n1[0]*x + n1[1]*y + n1[2]*z + plane1[3] = 0
    # At y=0, z=cylinder_z: n1[0]*x + n1[2]*cylinder_z + plane1[3] = 0
    if abs(n1[0]) > 0.001:
        plane1_x_at_yz = -(n1[2] * cylinder_z + plane1[3]) / n1[0]
        cylinder1_x = plane1_x_at_yz + support_offset  # Offset in +X direction
    else:
        # If plane1 is not X-aligned, use a different approach
        cylinder1_x = support_offset
    
    # For plane2, find where it intersects at y=0, z=cylinder_z
    if abs(n2[0]) > 0.001:
        plane2_x_at_yz = -(n2[2] * cylinder_z + plane2[3]) / n2[0]
        cylinder2_x = plane2_x_at_yz - support_offset  # Offset in -X direction
    else:
        # If plane2 is not X-aligned, use a different approach
        cylinder2_x = -support_offset
    
    # Both cylinders have Y=0 as their center (you can adjust this if needed)
    cylinder1_center = np.array([cylinder1_x, 0, cylinder_z])
    cylinder2_center = np.array([cylinder2_x, 0, cylinder_z])
    
    # Create two cylinders aligned with Y axis
    # Default cylinder is aligned with +Z, so we need to rotate 90 degrees around X
    cyl1 = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)
    cyl2 = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)
    
    # Rotation matrix to align Z axis with Y axis (90 degree rotation around X axis)
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    
    # Apply rotation and translation to first cylinder
    T1 = np.eye(4)
    T1[:3, :3] = R
    T1[:3, 3] = cylinder1_center
    cyl1.apply_transform(T1)
    
    # Apply rotation and translation to second cylinder
    T2 = np.eye(4)
    T2[:3, :3] = R
    T2[:3, 3] = cylinder2_center
    cyl2.apply_transform(T2)
    
    return cyl1, cyl2

def create_anvil_relative_to_planes(
    plane1, plane2, plane3,
    diameter, height, anvil_offset
):
    """
    Create a cylinder positioned between two parallel planes.
    The cylinder's geometric center will be positioned such that:
    - It's centered between plane1 and plane2 (which should be parallel)
    - Its axis is parallel to both plane1 and plane2
    - Its axis is parallel to the XY plane
    - Its center is at Y=0
    - Its Z coordinate is anvil_offset above plane3
    
    Parameters:
    - plane1, plane2: 4-element numpy arrays [a, b, c, d] for ax + by + cz + d = 0
                      These should be parallel to each other and roughly aligned to X
    - plane3: 4-element numpy array [a, b, c, d] for ax + by + cz + d = 0
              This should be aligned with the Z axis
    - diameter: Cylinder diameter
    - height: Cylinder height
    - anvil_offset: Height in Z above plane3 where cylinder center should be placed
    
    Returns:
    - trimesh.Trimesh cylinder mesh
    """
    import numpy as np
    import trimesh
    
    # Extract and normalize normals
    n1 = plane1[:3] / np.linalg.norm(plane1[:3])
    n2 = plane2[:3] / np.linalg.norm(plane2[:3])
    n3 = plane3[:3] / np.linalg.norm(plane3[:3])
    
    # Verify that plane1 and plane2 are roughly parallel
    # (dot product of normals should be close to ±1)
    dot_product = np.dot(n1, n2)
    if abs(abs(dot_product) - 1.0) > 0.1:  # Allow some tolerance
        print(f"Warning: plane1 and plane2 may not be parallel (dot product: {dot_product})")
    
    # Find the Z coordinate for the cylinder center (anvil_offset above plane3)
    # For plane3 (Z-aligned): ax + by + cz + d = 0
    # Find the Z coordinate of plane3 at x=0, y=0
    if abs(n3[2]) > 0.001:  # Plane3 is roughly Z-aligned
        # Z coordinate where plane3 intersects at x=0, y=0
        plane3_z_at_origin = -plane3[3] / n3[2]
        # Cylinder Z coordinate is anvil_offset above this
        cylinder_z = plane3_z_at_origin + anvil_offset
    else:
        # Fallback if plane3 is not Z-aligned
        cylinder_z = anvil_offset
    
    # Find the midpoint between the two parallel planes at the desired Z level
    # We need to find where each plane intersects the line at y=0, z=cylinder_z
    
    # Since planes are parallel, use the average normal (accounting for direction)
    if dot_product < 0:
        # Normals point in opposite directions, flip one
        n2 = -n2
    #     plane2_corrected = np.array([-plane2[0], -plane2[1], -plane2[2], -plane2[3]])
    # else:
    #     plane2_corrected = plane2
    
    avg_normal = (n1 + n2) / 2
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    # Find X coordinates where each plane intersects at y=0, z=cylinder_z
    # plane1: n1[0]*x + n1[1]*y + n1[2]*z + plane1[3] = 0
    # At y=0, z=cylinder_z: n1[0]*x + n1[2]*cylinder_z + plane1[3] = 0
    if abs(n1[0]) > 0.001:
        plane1_x_at_yz = -(n1[2] * cylinder_z + plane1[3]) / n1[0]
    else:
        plane1_x_at_yz = 0
    
    # plane2: n2[0]*x + n2[1]*y + n2[2]*z + plane2[3] = 0
    # At y=0, z=cylinder_z: n2[0]*x + n2[2]*cylinder_z + plane2[3] = 0
    if abs(n2[0]) > 0.001:
        plane2_x_at_yz = -(n2[2] * cylinder_z + plane2[3]) / n2[0]
    else:
        plane2_x_at_yz = 0
    
    # Center X coordinate is midpoint between the two planes
    center_x = (plane1_x_at_yz + plane2_x_at_yz) / 2
    
    # The cylinder axis should be perpendicular to the average normal
    # and parallel to the XY plane (so Z component should be 0)
    
    # Since plane1 and plane2 are roughly X-aligned, their normals are roughly X-aligned
    # The cylinder axis should be perpendicular to this, so roughly Y-aligned
    # But we want it parallel to XY plane, so we'll use the Y direction
    
    # Find a direction perpendicular to avg_normal and parallel to XY plane
    # Project avg_normal onto XY plane and rotate 90 degrees
    avg_normal_xy = np.array([avg_normal[0], avg_normal[1], 0])
    if np.linalg.norm(avg_normal_xy) > 0.1:
        avg_normal_xy = avg_normal_xy / np.linalg.norm(avg_normal_xy)
        # Rotate 90 degrees in XY plane: [x, y, 0] -> [-y, x, 0]
        cylinder_axis = np.array([-avg_normal_xy[1], avg_normal_xy[0], 0])
    else:
        # If avg_normal is purely Z-aligned, use Y as cylinder axis
        cylinder_axis = np.array([0, 1, 0])
    
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    
    # Set the cylinder center
    cylinder_center = np.array([center_x, 0, cylinder_z])
    
    # Create default cylinder aligned with +Z
    cyl = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)
    
    # Rotate to align Z axis with cylinder axis direction
    z_axis = np.array([0, 0, 1])
    if np.allclose(cylinder_axis, z_axis):
        R = np.eye(3)
    elif np.allclose(cylinder_axis, -z_axis):
        R = -np.eye(3)
    else:
        rotation_axis = np.cross(z_axis, cylinder_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm > 1e-6:
            rotation_axis /= rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, cylinder_axis), -1.0, 1.0))
            R = trimesh.transformations.rotation_matrix(angle, rotation_axis)[:3, :3]
        else:
            R = np.eye(3)
    
    # Apply rotation and translation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cylinder_center
    cyl.apply_transform(T)
    
    return cyl

def trimesh_to_open3d(tri_mesh):
    """
    Convert a trimesh.Trimesh object to open3d.geometry.TriangleMesh.
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def align_planes_to_axes_minimal_v2(aligned_pcd, keep_inlier_clouds, optimized_planes, planeX, planeZ, verbose=False):
    """
    Align planes to coordinate axes using truly minimal rotation.
    This version uses a more robust approach to avoid large rotations.
    
    Args:
        aligned_pcd: Open3D point cloud to rotate
        optimized_planes: List of plane equations [a, b, c, d] to rotate
        planeX: Plane equation whose normal should align with X-axis
        planeZ: Plane equation whose normal should align with Z-axis
    
    Returns:
        rotated_pcd: Rotated point cloud
        rotated_planes: List of rotated plane equations
        rotation_matrix: 3x3 rotation matrix that was applied
        rotation_info: Dictionary with rotation details
    """
    
    # Extract and normalize plane normals
    normalX = planeX[:3] / np.linalg.norm(planeX[:3])
    normalZ = planeZ[:3] / np.linalg.norm(planeZ[:3])
    
    print_verbose(f"Original normalX: {normalX}", verbose)
    print_verbose(f"Original normalZ: {normalZ}", verbose)
    
    # Target axes - we'll determine the best direction more carefully
    target_X_pos = np.array([1, 0, 0])
    target_X_neg = np.array([-1, 0, 0])
    target_Z_pos = np.array([0, 0, 1])
    target_Z_neg = np.array([0, 0, -1])
    
    # Calculate all possible alignments and their required rotation angles
    def rotation_angle_between_vectors(v1, v2):
        """Calculate the rotation angle needed to align v1 with v2"""
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(abs(dot_product))
    
    # Test all four combinations
    options = [
        (target_X_pos, target_Z_pos, "X+, Z+"),
        (target_X_pos, target_Z_neg, "X+, Z-"),
        (target_X_neg, target_Z_pos, "X-, Z+"),
        (target_X_neg, target_Z_neg, "X-, Z-")
    ]
    
    best_option = None
    min_total_angle = float('inf')
    
    print_verbose("\nEvaluating alignment options:", verbose)
    for target_X, target_Z, label in options:
        angle_X = rotation_angle_between_vectors(normalX, target_X)
        angle_Z = rotation_angle_between_vectors(normalZ, target_Z)
        total_angle = angle_X + angle_Z
        
        print_verbose(f"{label}: X angle = {np.degrees(angle_X):.3f}°, Z angle = {np.degrees(angle_Z):.3f}°, Total = {np.degrees(total_angle):.3f}°", verbose)
        
        if total_angle < min_total_angle:
            min_total_angle = total_angle
            best_option = (target_X, target_Z, label, angle_X, angle_Z)
    
    final_X, final_Z, best_label, best_angle_X, best_angle_Z = best_option
    print_verbose(f"\nSelected option: {best_label}", verbose)
    print_verbose(f"Target X direction: {final_X}", verbose)
    print_verbose(f"Target Z direction: {final_Z}", verbose)
    
    # Check orthogonality of target directions
    orthogonality_check = abs(np.dot(final_X, final_Z))
    if orthogonality_check > 1e-10:
        print_verbose(f"Warning: Target directions not orthogonal! Dot product: {orthogonality_check}", verbose)
    
    # Now compute the actual minimal rotation
    # We'll use a different approach: find the rotation that simultaneously
    # minimizes the distance to both target orientations
    
    # Method: Use the fact that we want to solve:
    # R @ normalX ≈ final_X
    # R @ normalZ ≈ final_Z
    # 
    # This is an orthogonal Procrustes problem
    R = compute_optimal_rotation_procrustes(
        np.column_stack([normalX, normalZ]), 
        np.column_stack([final_X, final_Z])
    )
    
    # Alternative method if Procrustes doesn't work well:
    # Use two sequential minimal rotations
    if np.linalg.det(R) < 0.9 or np.linalg.norm(R @ R.T - np.eye(3)) > 1e-6:
        print_verbose("Procrustes method failed, using sequential rotations", verbose)
        R = compute_sequential_minimal_rotation(normalX, normalZ, final_X, final_Z)
    
    # Verify the rotation
    final_normalX = R @ normalX
    final_normalZ = R @ normalZ
    
    print_verbose("\nAfter rotation:", verbose)
    print_verbose(f"Final normalX: {final_normalX}", verbose)
    print_verbose(f"Final normalZ: {final_normalZ}", verbose)
    
    # Check alignment quality
    alignment_X = abs(np.dot(final_normalX, final_X))
    alignment_Z = abs(np.dot(final_normalZ, final_Z))
    
    angle_error_X = np.degrees(np.arccos(np.clip(alignment_X, 0, 1)))
    angle_error_Z = np.degrees(np.arccos(np.clip(alignment_Z, 0, 1)))
    
    print_verbose(f"X alignment: {alignment_X:.6f} (error: {angle_error_X:.6f}°)", verbose)
    print_verbose(f"Z alignment: {alignment_Z:.6f} (error: {angle_error_Z:.6f}°)", verbose)
    
    # Verify rotation matrix properties
    det_R = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    
    if abs(det_R - 1.0) > 1e-10:
        print_verbose(f"Warning: Rotation matrix determinant = {det_R}, should be 1.0", verbose)
    if orthogonality_error > 1e-10:
        print_verbose(f"Warning: Rotation matrix orthogonality error = {orthogonality_error}", verbose)
    
    # Create transformation matrix and apply
    T = np.eye(4)
    T[:3, :3] = R
    rotated_pcd = aligned_pcd.transform(T)
    
    # Rotate keep_inlier_clouds
    rotated_inlier_clouds = []
    for cloud in keep_inlier_clouds:
        rotated_inlier_clouds.append(cloud.transform(T))
    
    # Apply rotation to all plane equations
    rotated_planes = []
    for plane in optimized_planes:
        old_normal = plane[:3]
        new_normal = R @ old_normal
        new_plane = np.array([new_normal[0], new_normal[1], new_normal[2], plane[3]])
        rotated_planes.append(new_plane)
    
    # Prepare rotation info
    rotation_scipy = Rotation.from_matrix(R)
    euler_angles = rotation_scipy.as_euler('xyz', degrees=True)
    rotvec = rotation_scipy.as_rotvec()
    rotation_angle = np.linalg.norm(rotvec) * 180 / np.pi
    
    rotation_info = {
        'original_normalX': normalX,
        'original_normalZ': normalZ,
        'target_X': final_X,
        'target_Z': final_Z,
        'final_normalX': final_normalX,
        'final_normalZ': final_normalZ,
        'euler_angles_deg': euler_angles,
        'rotation_axis_angle': rotvec,
        'rotation_angle_deg': rotation_angle,
        'alignment_errors_deg': (angle_error_X, angle_error_Z),
        'best_option': best_label
    }
    
    print_verbose("\nRotation summary:", verbose)
    print_verbose(f"Euler angles (XYZ): {euler_angles}", verbose)
    print_verbose(f"Total rotation angle: {rotation_angle:.3f}°", verbose)
    
    return rotated_pcd, rotated_inlier_clouds, rotated_planes, R, rotation_info


def compute_optimal_rotation_procrustes(source_vectors, target_vectors):
    """
    Compute optimal rotation using orthogonal Procrustes analysis.
    This finds the rotation R that minimizes ||R @ source_vectors - target_vectors||_F
    """
    # Compute SVD of target_vectors @ source_vectors.T
    H = target_vectors @ source_vectors.T
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = U @ Vt
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    return R


def compute_sequential_minimal_rotation(normalX, normalZ, final_X, final_Z):
    """
    Compute rotation using sequential minimal rotations.
    """
    # Step 1: Rotate normalX to final_X
    R1 = compute_rotation_between_vectors(normalX, final_X)
    
    # Step 2: See where normalZ goes after R1
    normalZ_intermediate = R1 @ normalZ
    
    # Step 3: Rotate around final_X to align normalZ_intermediate with final_Z
    R2 = compute_rotation_around_axis(final_X, normalZ_intermediate, final_Z)
    
    # Combined rotation
    R = R2 @ R1
    
    return R


def compute_rotation_between_vectors(v1, v2):
    """
    Compute rotation matrix that rotates vector v1 to vector v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Check if vectors are already aligned
    dot_product = np.dot(v1, v2)
    if abs(dot_product - 1.0) < 1e-10:
        return np.eye(3)
    
    # Check if vectors are opposite
    if abs(dot_product + 1.0) < 1e-10:
        # Find an orthogonal vector for 180° rotation
        if abs(v1[0]) < 0.9:
            orthogonal = np.array([1, 0, 0])
        else:
            orthogonal = np.array([0, 1, 0])
        
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        return rodrigues_rotation(axis, np.pi)
    
    # General case using Rodrigues' formula
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) < 1e-10:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return rodrigues_rotation(axis, angle)


def compute_rotation_around_axis(axis, v_from, v_to):
    """
    Compute rotation around given axis that rotates v_from to v_to.
    """
    axis = axis / np.linalg.norm(axis)
    
    # Project vectors onto plane perpendicular to axis
    v_from_proj = v_from - np.dot(v_from, axis) * axis
    v_to_proj = v_to - np.dot(v_to, axis) * axis
    
    # Check if vectors are already aligned in the plane
    if np.linalg.norm(v_from_proj) < 1e-10 or np.linalg.norm(v_to_proj) < 1e-10:
        return np.eye(3)
    
    # Normalize projected vectors
    v_from_proj = v_from_proj / np.linalg.norm(v_from_proj)
    v_to_proj = v_to_proj / np.linalg.norm(v_to_proj)
    
    # Compute angle between projected vectors
    dot_product = np.clip(np.dot(v_from_proj, v_to_proj), -1.0, 1.0)
    cross_product = np.cross(v_from_proj, v_to_proj)
    
    # Determine angle sign
    angle = np.arccos(dot_product)
    if np.dot(cross_product, axis) < 0:
        angle = -angle
    
    return rodrigues_rotation(axis, angle)


def rodrigues_rotation(axis, angle):
    """
    Compute rotation matrix using Rodrigues' rotation formula.
    """
    if abs(angle) < 1e-10:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Skew-symmetric matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # Rodrigues' formula
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    
    return R

def validate_minimal_rotation(original_planes, rotated_planes, planeX_idx, planeZ_idx, rotation_matrix, verbose=False):
    """
    Validate that the minimal rotation alignment worked correctly.
    """
    print_verbose("\n=== MINIMAL ROTATION VALIDATION ===", verbose)
    
    # Check the target planes
    # planeX_original = original_planes[planeX_idx]
    # planeZ_original = original_planes[planeZ_idx]
    planeX_rotated = rotated_planes[planeX_idx]
    planeZ_rotated = rotated_planes[planeZ_idx]
    
    # # Original normals
    # normalX_orig = planeX_original[:3] / np.linalg.norm(planeX_original[:3])
    # normalZ_orig = planeZ_original[:3] / np.linalg.norm(planeZ_original[:3])
    
    # Rotated normals
    normalX_rot = planeX_rotated[:3] / np.linalg.norm(planeX_rotated[:3])
    normalZ_rot = planeZ_rotated[:3] / np.linalg.norm(planeZ_rotated[:3])
    
    # Target axes
    target_X = np.array([1, 0, 0])
    target_Z = np.array([0, 0, 1])
    
    # Check alignment
    alignment_X_pos = abs(np.dot(normalX_rot, target_X))
    alignment_X_neg = abs(np.dot(normalX_rot, -target_X))
    alignment_Z_pos = abs(np.dot(normalZ_rot, target_Z))
    alignment_Z_neg = abs(np.dot(normalZ_rot, -target_Z))
    
    best_X_alignment = max(alignment_X_pos, alignment_X_neg)
    best_Z_alignment = max(alignment_Z_pos, alignment_Z_neg)
    
    angle_error_X = np.arccos(np.clip(best_X_alignment, 0, 1)) * 180 / np.pi
    angle_error_Z = np.arccos(np.clip(best_Z_alignment, 0, 1)) * 180 / np.pi
    
    print_verbose(f"PlaneX alignment with X-axis: {best_X_alignment:.6f} (error: {angle_error_X:.6f}°)", verbose)
    print_verbose(f"PlaneZ alignment with Z-axis: {best_Z_alignment:.6f} (error: {angle_error_Z:.6f}°)", verbose)
    
    # Check total rotation angle
    rotation_scipy = Rotation.from_matrix(rotation_matrix)
    total_angle = np.linalg.norm(rotation_scipy.as_rotvec()) * 180 / np.pi
    print_verbose(f"Total rotation angle: {total_angle:.3f}°", verbose)
    
    success = angle_error_X < 0.1 and angle_error_Z < 0.1
    print_verbose(f"Alignment successful: {success}\n\n", verbose)
    
    return success, angle_error_X, angle_error_Z, total_angle

def o3d_to_trimesh(o3d_mesh):
    """
    Convert an Open3D TriangleMesh to a Trimesh mesh.
    
    Parameters:
        o3d_mesh (open3d.geometry.TriangleMesh): The Open3D mesh.
    
    Returns:
        trimesh.Trimesh: The equivalent Trimesh object.
    """
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # Optionally include vertex normals and colors
    vertex_normals = np.asarray(o3d_mesh.vertex_normals) if o3d_mesh.has_vertex_normals() else None
    vertex_colors = np.asarray(o3d_mesh.vertex_colors) if o3d_mesh.has_vertex_colors() else None

    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals, vertex_colors=vertex_colors, process=False)

def identify_mesh_type(obj):
    """
    Identify whether the object is an Open3D TriangleMesh or a Trimesh mesh.
    
    Parameters:
        obj: The object to check.

    Returns:
        str: One of 'open3d', 'trimesh', or 'unknown'.
    """
    if isinstance(obj, o3d.geometry.TriangleMesh):
        return 'open3d'
    elif isinstance(obj, trimesh.Trimesh):
        return 'trimesh'
    else:
        return 'unknown'

def ensure_normals_outward(mesh: trimesh.Trimesh, mesh_name="", verbose=False):
    mesh_type = identify_mesh_type(mesh)
    if mesh_type=='open3d':
        mesh = o3d_to_trimesh(mesh)
    
    if not mesh.is_watertight:
        print_verbose(f"Warning: Mesh '{mesh_name}' is not watertight. Normal orientation may be unreliable.", verbose)
    
    # Attempt to fix normals
    mesh.fix_normals()

    # Optional check
    if not mesh.is_winding_consistent:
        print_verbose(f"Warning: Mesh '{mesh_name}' has inconsistent winding after fixing normals.", verbose)

    return mesh

def check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4):
    """
    Check if flex_mesh intersects with any other mesh in all_meshes using vertex proximity.
    
    Parameters:
        flex_mesh (trimesh.Trimesh): The mesh to check for intersections.
        all_meshes (list of trimesh.Trimesh): List of meshes, including flex_mesh.
        threshold (float): Distance threshold below which vertices are considered intersecting.
    
    Returns:
        list of int: Indices of meshes in all_meshes that intersect with flex_mesh.
    """
    intersections = []
    flex_idx = all_meshes.index(flex_mesh)
    flex_vertices = flex_mesh.vertices

    for idx, other_mesh in enumerate(all_meshes):
        if idx == flex_idx:
            continue

        other_vertices = other_mesh.vertices
        tree = cKDTree(other_vertices)
        distances, _ = tree.query(flex_vertices, k=1)

        if np.any(distances < threshold):
            intersections.append(idx)

    return intersections

def create_model(fixture_scan_path, specimen_scan_path, output_path, visualization=False, verbose=False):    
    expected_planes = {
        0: (np.array([0, 0, 1]), 1),
        1: (np.array([1, 0, 0]), 4),
        2: (np.array([np.sqrt(3)/2, 0, -0.5]), 1),
        3: (np.array([-np.sqrt(3)/2, 0, -0.5]), 1)
    }
    
    max_retries = 5
    retry_delay_seconds = 2
    attempts = 0
    
    while attempts < max_retries:
        try:
            keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid = fixture_plane_fitting.create_model(fixture_scan_path, expected_planes, visualization=visualization, verbose=verbose)
            break
        except:
            attempts += 1
            print_verbose(f"Attempt {attempts} failed.", verbose)
            time.sleep(retry_delay_seconds)
    
    optimized_planes, optimization_results, x_plane_indices, optimized_plane_meshes = optimize_constrained_planes(keep_planes, keep_inlier_clouds, verbose=verbose)
    
    for i, (original, optimized) in enumerate(zip(keep_planes, optimized_planes)):
        print_verbose(f'\nPlane {i}:', verbose)
        print_verbose(f'Angle Change: {angle_between_plane_normals(original, optimized)} deg', verbose)
    
    # Verification of constraints
    print_verbose("", verbose)
    for combo in combinations(range(len(optimized_planes)), 2):
        n1 = optimized_planes[combo[0]][:3]
        n2 = optimized_planes[combo[1]][:3]
        angle_diff = angular_change_between_normals(n1, n2)
        
        # Check if the current combo contains X-oriented planes
        set_combo = set(combo)
        set_x_plane_indices = set(x_plane_indices)
        
        common_elements = list(set_combo.intersection(set_x_plane_indices))
        
        labels = []
        for ele in common_elements:
            x_plane_pos = x_plane_indices.index(ele)
            if x_plane_pos == 0:
                labels.append("Left Support")
            elif x_plane_pos == 1:
                labels.append("Left Side Anvil")
            elif x_plane_pos == 2:
                labels.append("Right Side Anvil")
            elif x_plane_pos == 3:
                labels.append("Right Support")
            else:
                raise ValueError("Index out of range")
            
        if len(common_elements) == 2:
            print_verbose(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} ({labels[1]}) are separated by {angle_diff:.4f} degrees', verbose)
        elif len(common_elements) == 1:
            if combo[0] in common_elements:
                print_verbose(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} are separated by {angle_diff:.4f} degrees', verbose)
            else:
                print_verbose(f'Planes {combo[0]} and {combo[1]} ({labels[0]}) are separated by {angle_diff:.4f} degrees', verbose)                
        else:
            print_verbose(f'Planes {combo[0]} and {combo[1]} are separated by {angle_diff:.4f} degrees', verbose)
        
        if np.abs(np.around(angle_diff,2)) <= 0.01:
            separation_dist = separation_between_parallel_planes(optimized_planes[combo[0]], optimized_planes[combo[1]])
            print_verbose(f'\tPlanes {combo[0]} and {combo[1]} are separated by a distance of {separation_dist}', verbose)
    
    # Re-align everything so the support planes and base plane define the world orientation
    planeX_idx = x_plane_indices[0]
    planeZ_idx = 0
    planeX = optimized_planes[planeX_idx]
    planeZ = optimized_planes[planeZ_idx]
    
    # Perform alignment
    rotated_pcd, rotated_inlier_clouds, rotated_planes, R_planes, info = align_planes_to_axes_minimal_v2(
        aligned_pcd, keep_inlier_clouds, optimized_planes, planeX, planeZ, verbose=verbose
    )
    
    # Validate results
    success, error_X, error_Z, total_angle = validate_minimal_rotation(
        optimized_planes, rotated_planes, planeX_idx, planeZ_idx, R_planes, verbose=verbose
    )    
    
    print_verbose(f'\nPlane adjustment R matrix:\n{R_planes}', verbose)
    
    if R_flip is not None:
        R_total = R_planes @ R_90X @ R_flip @ R_pca
    else:
        R_total = R_planes @ R_90X @ R_pca
    
    print_verbose(f'\nR Total:\n{R_total}', verbose)
    
    # === STEP 1: Load the original mesh ===
    original_mesh = load_mesh(fixture_scan_path)
    # original_vertices = np.asarray(original_mesh.vertices)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
    
    # === STEP 2: Translate to origin ===
    original_mesh.translate(-centroid)
    
    # === STEP 3: PCA alignment ===
    original_mesh.rotate(R_pca, center=(0, 0, 0))
    
    if R_flip is not None:
        original_mesh.rotate(R_flip, center=(0, 0, 0))
    
    # === STEP 4: Rotate 90° about +X ===
    original_mesh.rotate(R_90X, center=(0, 0, 0))
    pre_minimal_rotation_mesh = copy.deepcopy(original_mesh)
    
    # === STEP 5: Apply final minimal rotation matrix R ===
    original_mesh.rotate(R_planes, center=(0, 0, 0))
    
    # === RESULT ===
    transformed_reference_mesh = copy.deepcopy(original_mesh)
    
    if visualization:
        o3d.visualization.draw_geometries([
            aligned_pcd.paint_uniform_color([0.5, 0.5, 0.5]),
            transformed_reference_mesh.paint_uniform_color([1, 0, 0]),
            pre_minimal_rotation_mesh.paint_uniform_color([0, 0, 1]),
            axis], window_name="Transformed Reference Mesh (pre-minimal rotation is blue)")

    # Load and align matched specimen
    # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    matched_specimen_mesh = load_mesh(specimen_scan_path, verbose=verbose)
    
    aligned_specimen_mesh = align_tgt_to_ref_meshes(transformed_reference_mesh, matched_specimen_mesh, visualize=visualization, verbose=verbose)
    
    
    # Perform anvil alignment adjustment here (require specific spacing and orthogonal to base)
    # plane_indices = [x_plane_indices[1],
    #                  x_plane_indices[2],
    #                  0]
    # planes_list = [rotated_planes[idx] for idx in plane_indices]
    # inlier_clouds_list = [rotated_inlier_clouds[idx] for idx in plane_indices]
    
    optimized_anvil_planes, anvil_optimization_results = optimize_anvil_planes(rotated_planes, rotated_inlier_clouds)
    
    #### Create mesh models of support and anvil cylinders
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(optimized_anvil_planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    diameter = 10
    height = 40
    support_offset = 25 # Distance from outer planes to support cylinders
    anvil_offset = 70
    
    base_plane = rotated_planes[0]
    base_support_offset = 52 # Distance from base plane to center of support cylinders
    
    # support_offset = 25.4   # 1 inch
    
    anvil_plane1 = optimized_anvil_planes[x_plane_indices[1]]
    anvil_plane2 = optimized_anvil_planes[x_plane_indices[2]]
    # anvil_plane1 = rotated_planes[5]
    # anvil_plane2 = rotated_planes[6]
    anvil = create_anvil_relative_to_planes(anvil_plane1,
                                            anvil_plane2,
                                            base_plane,
                                            diameter,
                                            height,
                                            anvil_offset)
    anvil_mesh = trimesh_to_open3d(anvil)
    
    l_support_plane = optimized_anvil_planes[x_plane_indices[0]]
    r_support_plane = optimized_anvil_planes[x_plane_indices[3]]
    # l_support_plane = rotated_planes[x_plane_indices[0]]
    # r_support_plane = rotated_planes[x_plane_indices[3]]
    l_support, r_support = create_supports_relative_to_planes(
                            l_support_plane, r_support_plane, base_plane,
                            support_offset, base_support_offset,
                            diameter, height)
    
    l_support_mesh = trimesh_to_open3d(l_support)
    r_support_mesh = trimesh_to_open3d(r_support)
    
    # Create coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])

    # Visualize optimized planes and the inlier clouds used for fitting
    cylinder_meshes = [anvil_mesh, l_support_mesh, r_support_mesh]
    if visualization:
        o3d.visualization.draw_geometries([rotated_pcd] + [axis] + cylinder_meshes + [aligned_specimen_mesh], window_name="Aligned Cylinders")
    
    anvil = ensure_normals_outward(anvil, mesh_name="anvil", verbose=verbose)
    l_support = ensure_normals_outward(l_support, mesh_name="left_support", verbose=verbose)
    r_support = ensure_normals_outward(r_support, mesh_name="right_support", verbose=verbose)
    flex_mesh = ensure_normals_outward(aligned_specimen_mesh, mesh_name="flex_mesh", verbose=verbose)
    
    # Combine meshes
    all_meshes = [flex_mesh, anvil, l_support, r_support]
    
    intersecting_indices = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    
    increment = 0.01  # Adjust the movement step as needed
    max_iterations = 100  # Prevent infinite loops
    
    if intersecting_indices:
        for _ in range(max_iterations):
            still_intersecting = False
    
            # Check and resolve intersection with anvil
            if all_meshes.index(anvil) in intersecting_indices:
                # Move anvil up in Z
                anvil.apply_translation([0, 0, increment])
                print_verbose(f"Applying {increment} adjustment to Z position of anvil", verbose)
                # Recheck
                updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
                if all_meshes.index(anvil) in updated:
                    still_intersecting = True
    
            # Check and resolve intersection with supports
            l_idx = all_meshes.index(l_support)
            r_idx = all_meshes.index(r_support)
            if l_idx in intersecting_indices or r_idx in intersecting_indices:
                # Move both supports down in Z
                l_support.apply_translation([0, 0, -increment])
                r_support.apply_translation([0, 0, -increment])
                print_verbose(f"Applying -{increment} adjustment to Z position of supports", verbose)
                # Recheck
                updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
                if l_idx in updated or r_idx in updated:
                    still_intersecting = True
    
            if not still_intersecting:
                break
        else:
            print_verbose("Warning: Maximum adjustment iterations reached, intersection may still exist.", verbose)
    
    # Adjust for good measure
    n_increments = 2
    anvil.apply_translation([0, 0, n_increments*increment])
    l_support.apply_translation([0, 0, -n_increments*increment])
    r_support.apply_translation([0, 0, -n_increments*increment])
    
    merged_mesh = trimesh.util.concatenate(all_meshes)
    
    # output_filepath = "E:/Fixture Scans/prepared_test.stl"
    merged_mesh.export(output_path)

def create_models(test_data_filepath, scanned_fixtures_folder, scanned_specimens_folder, prepared_meshes_folder, visualization=False, verbose=False):
    df_test_data = pd.read_excel(test_data_filepath)
    os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # Iterate through df_test_data and create the necessary multibody STL files for simulation
    for index, row in df_test_data.iterrows():
        fixture_scan_filename = row["Fixture Scan File"]
        fixture_scan_path = os.path.join(scanned_fixtures_folder, fixture_scan_filename)
        
        specimen_scan_filename = row["Specimen Scan File"]
        specimen_scan_path = os.path.join(scanned_specimens_folder, specimen_scan_filename)
        
        specimen = row["Specimen"]
        test_num = row["Test_Num"]
        output_filename = f"{specimen}_Test{int(test_num)}.stl"
        output_path = os.path.join(prepared_meshes_folder, output_filename)
        create_model(fixture_scan_path, specimen_scan_path, output_path, visualization=visualization, verbose=verbose)
        
        # Save the job name and add it to test_data.xlsx
        job_name = f"{specimen}_{test_num}"
        df_test_data.loc[index, "Job Name"] = job_name
        
        # Save the test specific mesh file name and add it to test_data.xlsx
        df_test_data.loc[index, "Test Specific Mesh File"] = output_path
        
        print(f"{output_filename} model creation complete")
        
    # Export changed dataframe
    df_test_data.to_excel(test_data_filepath, index=False)
        
    

if __name__ == "__main__":
    # fixture_scan_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Fixtures/X4_Test2_raw.stl"
    # specimen_scan_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Specimens/X4_raw.stl"
    # output_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes/X4_Test2.stl"
    
    # create_model(fixture_scan_path, specimen_scan_path, output_path)
    
    test_data_filepath = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    scanned_fixtures_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Fixtures"
    scanned_specimens_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Specimens"
    prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    create_models(test_data_filepath, scanned_fixtures_folder, scanned_specimens_folder, prepared_meshes_folder)