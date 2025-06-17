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
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN
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

def point_to_cylinder_surface_distance(points, center, axis, radius, height):
    """
    Compute distances from points to the surface of a finite cylinder.

    Parameters:
    - points: (N, 3) array of 3D points.
    - center: (3,) array, the center of the cylinder (center of the axis).
    - axis: (3,) unit vector along the cylinder axis.
    - radius: scalar, radius of the cylinder.
    - height: scalar, total height of the cylinder (along axis).

    Returns:
    - distances: (N,) array of distances from each point to the surface of the cylinder.
    """

    # Normalize axis just in case
    A = axis / np.linalg.norm(axis)
    C = center
    v = points - C  # vectors from center to points

    # Project points onto cylinder axis
    proj_len = np.dot(v, A)  # scalar projection length (signed)
    half_h = height / 2

    # Clamp projection to within the height of the cylinder
    proj_len_clamped = np.clip(proj_len, -half_h, half_h)

    # Closest point on the infinite axis line (clamped to the finite cylinder length)
    proj_points = C + np.outer(proj_len_clamped, A)

    # Radial distance from the axis
    radial_vecs = points - proj_points
    radial_dist = np.linalg.norm(radial_vecs, axis=1)

    # Distance to side surface
    side_dist = np.abs(radial_dist - radius)

    # Determine which points are within height bounds
    is_within_height = np.abs(proj_len) <= half_h

    # If within height, use side distance. If outside, compute cap distance.
    # Compute cap centers
    cap_base = C - half_h * A
    cap_top  = C + half_h * A

    # Vector from points to nearest cap
    closer_cap = np.where((proj_len > half_h)[:, np.newaxis], cap_top, cap_base)
    cap_vecs = points - closer_cap
    cap_proj = np.dot(cap_vecs, A)[:, np.newaxis] * A
    cap_radial = cap_vecs - cap_proj

    cap_dist = np.linalg.norm(cap_radial, axis=1)

    # Distance to cap surface (radial)
    dist_outside = np.sqrt(cap_dist**2 + np.maximum(np.abs(proj_len) - half_h, 0)**2)

    # Use side distance for points within the height, cap distance otherwise
    distances = np.where(is_within_height, side_dist, dist_outside)

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
            fitness[i] = 1e6 * angle_deg  # large penalty
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
    n_iter = 300
    best_cost, best_pos = optimizer.optimize(objective_swarm, n_iter, points=points, C0=expected_center,
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

def filter_window_pts_z(window_pts, cylinder_diameter):
    zvals = window_pts[:,2]
    counts, bins, bars = plt.hist(zvals, bins=50)
    plt.show()
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

def filter_window_pts_cyl_end(window_pts, end_factor=0.05):
    yvals = window_pts[:,1]
    max_y = np.max(yvals)
    min_y = np.min(yvals)
    y_length = max_y - min_y
    
    # end_factor is the percentage of the y length that should be considered for filtering the window_pts
    end_cap_a = max_y - y_length * end_factor
    # end_cap_b = min_y + y_length * end_factor
    mask = (window_pts[:,1] >= end_cap_a)
    # mask = (window_pts[:,1] >= end_cap_a) & (window_pts[:,1] <= end_cap_b)
    selected_pts = window_pts[mask]
    
    return selected_pts
    

# def refine_translation_along_axis(points, C0, A, radius, height):
#     def cap_alignment_objective(t_scalar):
#         C = C0 + t_scalar * A
#         half_h = height / 2
#         cap_bottom = C - half_h * A
#         cap_top = C + half_h * A

#         # Project each point onto the axis
#         v = points - C
#         proj_len = np.dot(v, A)

#         # Determine whether point is closer to top or bottom cap
#         closer_cap = np.where((proj_len > 0)[:, np.newaxis], cap_top[np.newaxis, :], cap_bottom[np.newaxis, :])

#         cap_distances = np.linalg.norm(points - closer_cap, axis=1)

#         # Focus on points near ends (e.g., 10% from each side)
#         height_thresh = height * 0.15
#         end_mask = np.abs(proj_len) > (height / 2 - height_thresh)
#         end_cap_errors = cap_distances[end_mask]

#         if len(end_cap_errors) == 0:
#             return np.inf
#         return np.median(end_cap_errors)

#     result = minimize_scalar(cap_alignment_objective, bounds=(-5.0, 5.0), method='bounded')
#     C_refined = C0 + result.x * A
#     return C_refined, result.fun

def refine_translation_along_axis(points, C0, A, radius, height):
    def cap_alignment_objective(t_scalar):
        # Updated cylinder center along axis
        C = C0 + t_scalar * A

        # Project each point onto axis
        v = points - C
        proj_len = np.dot(v, A)

        # Define expected cap bounds
        half_h = height / 2

        # Option A: Penalize deviation from ideal range
        # Use both outside-cap and asymmetry penalties

        # 1. Overhang penalty: points that exceed cap bounds
        overhang = np.abs(np.abs(proj_len) - half_h)
        # overhang = np.clip(np.abs(proj_len) - half_h, 0, None)
        overhang_penalty = np.mean(overhang**2)

        # # 2. Centering penalty: deviation of the point distribution from being centered
        # mean_proj = np.mean(proj_len)
        # centering_penalty = mean_proj**2
        
        print(f"t={t_scalar:.3f}, overhang_penalty={overhang_penalty:.4f}")
        # print(f"t={t_scalar:.3f}, overhang_penalty={overhang_penalty:.4f}, center={mean_proj:.3f}")

        return overhang_penalty
        # return overhang_penalty + 0.1 * centering_penalty  # Adjust weight if needed

    result = minimize_scalar(cap_alignment_objective, bounds=(-5.0, 5.0), method='bounded')
    C_refined = C0 + result.x * A
    return C_refined, result.fun

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
        
        window_pts = filter_window_pts_z(window_pts, expected_diameter)
        
        if len(window_pts) == 0:
            print(f"⚠️ No points found near x = {x}")
            continue
        
        color = [random.random() for _ in range(3)]
        # pcd_slice = create_colored_pcd(window_pts, color)
        # geoms.append(pcd_slice)
        
        mean_yz = np.mean(window_pts[:, 1:], axis=0)
        expected_center = np.array([x, mean_yz[0], mean_yz[1]])
        
        C_opt, A_opt, radius, height, best_cost = optimize_cylinder_fit_pso(
            window_pts, expected_center, expected_axis, expected_radius, expected_height)
        
        # Further refine the axial placement of the cylinder
        window_pts = filter_window_pts_cyl_end(window_pts, end_factor=1e-4)     
        print(f"  Cylinder ends selection found {len(window_pts)} points")
        C_opt_refined, refined_cost = refine_translation_along_axis(window_pts, C_opt, A_opt, expected_radius, expected_height)
        
        # color = [random.random() for _ in range(3)]
        pcd_slice = create_colored_pcd(window_pts, color)
        geoms.append(pcd_slice)
        
        print(f"✅ Cylinder at x={x}: radius={radius:.2f}, height={height:.2f}, median fit error={best_cost:.4f}")
        
        cyl_mesh = create_cylinder_mesh(C_opt_refined, A_opt, radius, height)
        geoms.append(cyl_mesh)
        
        # # Calculate point-wise errors for final fit
        # errors = point_to_cylinder_distances(window_pts, C_opt, A_opt, radius, height)
        
        # # Plot histogram of errors
        # plt.figure(figsize=(6,4))
        # plt.hist(errors, bins=50, color=color, alpha=0.7)
        # plt.title(f"Error Distribution for Cylinder at x={x}")
        # plt.xlabel("Distance to Cylinder Surface (units)")
        # plt.ylabel("Number of Points")
        # plt.grid(True)
        # plt.show()
    
    o3d.visualization.draw_geometries(geoms, window_name="Optimized Cylinder Fits")

if __name__ == "__main__":
    main()
