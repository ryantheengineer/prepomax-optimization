"""
Assume the mesh that is brought into this script has already been cleaned and
PCA-aligned to the coordinate axes.
"""

import open3d as o3d
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pyswarms as ps

def load_mesh_as_point_cloud(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

def voxel_downsample(pcd, voxel_size=0.2):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def extract_window_points(points, x_center, window):
    half_w = window / 2.0
    mask = (points[:, 0] >= (x_center - half_w)) & (points[:, 0] <= (x_center + half_w))
    selected = points[mask]
    print(f"  Window [{x_center - half_w:.2f}, {x_center + half_w:.2f}] selected {len(selected)} points")
    return selected

def create_colored_pcd(points, color, z_offset=0.1):
    offset_points = points.copy()
    offset_points[:, 2] += z_offset  # slight Z offset for visibility
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(offset_points)
    pcd.paint_uniform_color(color)
    return pcd

def point_to_cylinder_distances(points, C, A, radius, height):
    v = points - C  # vector from base center to points
    proj_len = np.dot(v, A)  # projection lengths on axis
    proj_len_clamped = np.clip(proj_len, 0, height)  # clamp to cylinder height
    
    proj_points = C + np.outer(proj_len_clamped, A)
    radial_vec = points - proj_points
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    dist_surface = np.abs(radial_dist - radius)
    return dist_surface

def point_to_cylinder_surface_distance(points, C, A, radius, height):
    """
    Compute shortest distance from each point to the finite cylinder defined by:
    - Base center `C`
    - Axis direction `A` (unit vector)
    - Radius `radius`
    - Height `height` (along direction `A`)
    """
    A = A / np.linalg.norm(A)  # ensure A is unit
    v = points - C
    t = np.dot(v, A)  # signed projection along axis
    closest_points = np.empty_like(points)

    radial_dir = v - np.outer(t, A)  # vector from axis to point

    # Case 1: between caps (side surface)
    side_mask = (t >= 0) & (t <= height)
    if np.any(side_mask):
        radial_norm = np.linalg.norm(radial_dir[side_mask], axis=1, keepdims=True)
        radial_norm[radial_norm == 0] = 1e-8  # prevent divide-by-zero
        radial_unit = radial_dir[side_mask] / radial_norm
        closest_points[side_mask] = C + np.outer(t[side_mask], A) + radius * radial_unit

    # Case 2a: before base
    below_mask = t < 0
    if np.any(below_mask):
        cap_center = C
        v_below = points[below_mask] - cap_center
        d_xy = v_below - np.outer(np.dot(v_below, A), A)
        d_norm = np.linalg.norm(d_xy, axis=1)
        inside = d_norm <= radius
        # project to cap face
        closest_points[below_mask] = cap_center + d_xy
        # outside cap circle → distance to edge
        if np.any(~inside):
            r_unit = d_xy[~inside] / d_norm[~inside][:, None]
            closest_points[np.where(below_mask)[0][~inside]] = cap_center + radius * r_unit

    # Case 2b: beyond top
    above_mask = t > height
    if np.any(above_mask):
        cap_center = C + height * A
        v_above = points[above_mask] - cap_center
        d_xy = v_above - np.outer(np.dot(v_above, A), A)
        d_norm = np.linalg.norm(d_xy, axis=1)
        inside = d_norm <= radius
        closest_points[above_mask] = cap_center + d_xy
        if np.any(~inside):
            r_unit = d_xy[~inside] / d_norm[~inside][:, None]
            closest_points[np.where(above_mask)[0][~inside]] = cap_center + radius * r_unit

    distances = np.linalg.norm(points - closest_points, axis=1)
    return distances

def objective_swarm(params, points, C0, A0, radius, height, max_angle_deg=5):
    # params shape: (n_particles, 6) for [t_x, t_y, t_z, rvec_x, rvec_y, rvec_z]
    n_particles = params.shape[0]
    fitness = np.zeros(n_particles)
    y_axis = np.array([0, 1, 0])
    
    for i in range(n_particles):
        t = params[i, 0:3]
        rvec = params[i, 3:6]
        rot = R.from_rotvec(rvec)
        A = rot.apply(A0)
        C = C0 + t
        
        # Axis orientation constraint
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(A, y_axis) / (np.linalg.norm(A) * np.linalg.norm(y_axis)), -1, 1)))
        if angle_deg > max_angle_deg:
            fitness[i] = 1e6  # large penalty
            continue
        
        dists = point_to_cylinder_surface_distance(points, C, A, radius, height)
        fitness[i] = np.median(dists)
    
    return fitness

def optimize_cylinder_fit_pso(points, expected_center, expected_axis, radius, height):
    # Define bounds: [min, max] for each of 6 params
    # Translation bounds ±10 units, rotation vector bounds ±0.5 radians
    # bounds = (
    #     [expected_center[0] - 10, expected_center[1] - 10, expected_center[2] - 10, -0.5, -0.5, -0.5],  # min bounds
    #     [expected_center[0] + 10, expected_center[1] + 10, expected_center[2] + 10, 0.5, 0.5, 0.5]   # max bounds
    # )
    
    # Initialize swarm size and options
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # cognitive, social, inertia weights (tune these)
    optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=6, options=options)
    # optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=6, options=options, bounds=bounds)
    
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(objective_swarm, 600, points=points, C0=expected_center,
                                             A0=expected_axis, radius=radius, height=height)
    
    t_opt = best_pos[0:3]
    rvec_opt = best_pos[3:6]
    rot_opt = R.from_rotvec(rvec_opt)
    A_opt = rot_opt.apply(expected_axis)
    C_opt = expected_center + t_opt
    
    return C_opt, A_opt, radius, height, best_cost

def create_cylinder_mesh(center, axis, radius, height, resolution=30, color=[0.8, 0.2, 0.2]):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.paint_uniform_color(color)
    axis = axis / np.linalg.norm(axis)
    
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, axis)
    s = np.linalg.norm(v)
    if s > 1e-6:
        c = np.dot(z_axis, axis)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_mat = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    else:
        R_mat = np.eye(3)
    
    cylinder.rotate(R_mat, center=np.zeros(3))
    cylinder.translate(center)
    return cylinder

def filter_window_pts(window_pts, cylinder_diameter):
    zvals = window_pts[:,2]
    counts, bins, bars = plt.hist(zvals, bins=50)
    bin_centers = []
    for i,bin_edge in enumerate(bins):
        if i == len(bins)-1:
            break
        bin_centers.append(np.mean([bins[i],bins[i+1]]))
        
    counts_filtered = gaussian_filter1d(counts, sigma=1)
    counts_filtered = np.concatenate(([min(counts_filtered)], counts_filtered, [min(counts_filtered)]))
    peaks, _ = find_peaks(counts_filtered, height=70)
    peaks = [peak-1 for peak in peaks]
    
    # Use peaks (indices of peaks) to define bounds
    n_peaks = len(peaks)
    if n_peaks != 3:
        raise ValueError(f"Detected {n_peaks} peaks, but expected 3")
    z_bounds = [bin_centers[peak] for peak in peaks]
    
    separations = []
    for i,bound in enumerate(z_bounds):
        if i == len(z_bounds)-1:
            break
        separations.append(np.abs(z_bounds[i] - z_bounds[i+1]))
        
    diffs = [separation - cylinder_diameter for separation in separations]
    absdiffs = [np.abs(diff) for diff in diffs]
    min_ind = absdiffs.index(min(absdiffs))
    # If the selected region doesn't completely cover the expected cylinder,
    # then grow the selected region until it does, plus a buffer
    buffer = 0.5
    if diffs[min_ind] < 0:
        low_bound = z_bounds[min_ind] + diffs[min_ind]
        up_bound = z_bounds[min_ind+1] + np.abs(diffs[min_ind])
    else:
        low_bound = z_bounds[min_ind]
        up_bound = z_bounds[min_ind+1]
        
    low_bound -= buffer
    up_bound += buffer
        
    # Select window_pts within low_bound and up_bound for z coordinates
    mask = (window_pts[:,2] >= low_bound) & (window_pts[:,2] <= up_bound)
    selected_pts = window_pts[mask]
    
    return selected_pts

def main():
    mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/cylinder_detection_test.stl"
    expected_x_positions = [-32, 0, 32]
    window = 15.0
    voxel_size = 0.2
    
    expected_radius = 4.0
    expected_diameter = 2 * expected_radius
    expected_height = 20.0
    expected_axis = np.array([0, 0, 1])
    
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(mesh_path)
    down_pcd = voxel_downsample(base_pcd, voxel_size)
    down_pts = np.asarray(down_pcd.points)
    print(f"Downsampled to {len(down_pts)} points")
    
    geoms = [down_pcd]
    
    for x in expected_x_positions:
        window_pts = extract_window_points(down_pts, x_center=x, window=window)
        
        window_pts = filter_window_pts(window_pts, expected_diameter)
        
        if len(window_pts) == 0:
            print(f"⚠️ No points found near x = {x}")
            continue
        
        color = [random.random() for _ in range(3)]
        pcd_slice = create_colored_pcd(window_pts, color)
        geoms.append(pcd_slice)
        
        mean_yz = np.mean(window_pts[:, 1:], axis=0)
        expected_center = np.array([x, mean_yz[0], mean_yz[1]])
        
        C_opt, A_opt, radius, height, best_cost = optimize_cylinder_fit_pso(
            window_pts, expected_center, expected_axis, expected_radius, expected_height)
        
        print(f"✅ Cylinder at x={x}: radius={radius:.2f}, height={height:.2f}, median fit error={best_cost:.4f}")
        
        cyl_mesh = create_cylinder_mesh(C_opt, A_opt, radius, height)
        geoms.append(cyl_mesh)
        
        # Calculate point-wise errors for final fit
        errors = point_to_cylinder_distances(window_pts, C_opt, A_opt, radius, height)
        
        # Plot histogram of errors
        plt.figure(figsize=(6,4))
        plt.hist(errors, bins=50, color=color, alpha=0.7)
        plt.title(f"Error Distribution for Cylinder at x={x}")
        plt.xlabel("Distance to Cylinder Surface (units)")
        plt.ylabel("Number of Points")
        plt.grid(True)
        plt.show()
    
    o3d.visualization.draw_geometries(geoms, window_name="Optimized Cylinder Fits")

if __name__ == "__main__":
    main()
