import open3d as o3d
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pyswarms as ps
import copy
from itertools import combinations
import time

def load_mesh_as_point_cloud(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

def align_point_cloud_with_pca(pcd):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud.")

    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Center the points at origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    axes = pca.components_

    # Sort axes by explained variance (descending)
    order = np.argsort(pca.explained_variance_)[::-1]
    ordered_axes = axes[order]

    # Align principal components with +X, +Y, +Z
    target_axes = np.eye(3)
    rotation_matrix = ordered_axes.T @ target_axes

    # Apply rotation
    rotated_points = centered_points @ rotation_matrix

    # Create new point cloud
    pcd_pca = o3d.geometry.PointCloud()
    pcd_pca.points = o3d.utility.Vector3dVector(rotated_points)

    # Copy over colors if they exist
    if pcd.has_colors():
        pcd_pca.colors = pcd.colors
    if pcd.has_normals():
        # Recompute normals since they may no longer be valid
        pcd_pca.estimate_normals()
        
    # o3d.visualization.draw_geometries([pcd_pca], window_name="PCA Point Cloud")
    
    return pcd_pca

def voxel_downsample(pcd, voxel_size=0.2):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def extract_window_points(points, x_center, window):
    half_w = window / 2.0
    mask = (points[:, 0] >= (x_center - half_w)) & (points[:, 0] <= (x_center + half_w))
    selected = points[mask]
    print(f"  Window [{x_center - half_w:.2f}, {x_center + half_w:.2f}] selected {len(selected)} points")
    return selected

def find_cylinders_x(points, expected_x_positions):
    x_coords = points[:,0]
    hist, bin_edges = np.histogram(x_coords, bins=500)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    smoothed_hist = gaussian_filter1d(hist, sigma=3)
    
    peaks, _ = find_peaks(smoothed_hist, height=20)
    peak_x_positions = bin_centers[peaks]
    
    def align_expected_to_peaks(expected_x_positions, peak_x_positions):
        expected = np.array(expected_x_positions)
        peaks = np.array(peak_x_positions)
    
        # Normalize expected to relative spacing (center around 0)
        expected_rel = expected - np.mean(expected)
    
        # Search for best scale and offset that maps expected_rel into peak space
        best_error = float('inf')
        best_aligned = None
    
        # Try all combinations of 3 peaks out of peak_x_positions
        from itertools import combinations
    
        for combo in combinations(peaks, 3):
            combo = np.array(combo)
            # Compute best affine transform from expected_rel to combo
            A = np.vstack([expected_rel, np.ones_like(expected_rel)]).T
            try:
                sol, _, _, _ = np.linalg.lstsq(A, combo, rcond=None)
            except np.linalg.LinAlgError:
                continue
            scale, offset = sol
            aligned = expected_rel * scale + offset
            error = np.linalg.norm(aligned - combo)
            if error < best_error:
                best_error = error
                best_aligned = aligned
    
        return best_aligned
    
    best_aligned = align_expected_to_peaks(expected_x_positions, peak_x_positions)
    
    return best_aligned

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

def proportion_near_zero(values, tolerance=1e-6):
    """
    Returns the proportion of values that are within a given tolerance of zero.

    Parameters:
        values (list or np.ndarray): A list or array of numeric values.
        tolerance (float): The absolute tolerance within which values are considered near zero.

    Returns:
        float: Proportion of values within the tolerance of zero.
    """
    values = np.asarray(values)
    near_zero = np.abs(values) <= tolerance
    return np.sum(near_zero) / len(values)

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
        # plt.hist(dists)
        fitness[i] = 1 - proportion_near_zero(dists, tolerance=0.1)
        # fitness[i] = np.median(dists)
    
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
        
    counts_filtered = gaussian_filter1d(counts, sigma=2)
    counts_filtered = np.concatenate(([min(counts_filtered)], counts_filtered, [min(counts_filtered)]))
    peaks, properties = find_peaks(counts_filtered, height=50)
    peaks = [peak-1 for peak in peaks]
    
    # # Use peaks (indices of peaks) to define bounds
    n_peaks = len(peaks)
    if n_peaks not in (2,3):
        raise ValueError(f"Detected {n_peaks} peaks, but expected 2 or more")
    z_bounds = [bin_centers[peak] for peak in peaks]
    
    peak_tol = 2.5
    if len(z_bounds) < 3:
        if not any(abs(z - min(zvals)) <= peak_tol for z in z_bounds):
            z_bounds.insert(0, min(zvals))
            print("Added a z_bound at the bottom")
        elif not any(abs(z - max(zvals)) <= peak_tol for z in z_bounds):
            z_bounds.append(max(zvals))
            print("Added a z_bound at the top")
            
    
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

def filter_window_pts_z_simple(window_pts, cylinder_diameter, support=True):
    z_buffer = 0.5
    if support:
        # Get the bottom points that are within cylinder_diameter of the min
        min_z = min(window_pts[:,2])
        mask = (window_pts[:,2] <= min_z + cylinder_diameter + z_buffer)
    else:
        max_z = max(window_pts[:,2])
        mask = (window_pts[:,2] >= max_z - cylinder_diameter - z_buffer)
        
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

def refine_translation_along_axis(points, C0, A, radius, height):
    def cap_alignment_objective(t_scalar):
        # Updated cylinder center along axis
        C = C0 + t_scalar * A

        # Project each point onto axis
        v = points - C
        proj_len = np.dot(v, A)

        # Define expected cap bounds
        half_h = height / 2

        # 1. Overhang penalty: points that exceed cap bounds
        overhang = np.abs(np.abs(proj_len) - half_h)
        overhang_penalty = np.mean(overhang**2)
        
        print(f"t={t_scalar:.3f}, overhang_penalty={overhang_penalty:.4f}")

        return overhang_penalty

    result = minimize_scalar(cap_alignment_objective, bounds=(-5.0, 5.0), method='bounded')
    C_refined = C0 + result.x * A
    return C_refined, result.fun

def find_cylinders():
    # mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/test_setup_2_L44_R32.stl"
    mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/test_setup_1_L44_R51_2.stl"
    # mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/cylinder_detection_test.stl"
    expected_x_positions = [-44, 0, 51]
    # expected_x_positions = [-44, 0, 32]
    window = 30.0
    voxel_size = 0.2
    
    expected_radius = 5.0
    expected_diameter = 2 * expected_radius
    expected_height = 57.0
    expected_axis = np.array([0, 0, 1])
    
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(mesh_path)
    
    # Align the point cloud with PCA
    pcd_pca = align_point_cloud_with_pca(base_pcd)
    
    best_aligned = find_cylinders_x(np.asarray(pcd_pca.points), expected_x_positions)
    
    down_pcd = voxel_downsample(pcd_pca, voxel_size)
    # down_pcd = voxel_downsample(base_pcd, voxel_size)
    down_pts = np.asarray(down_pcd.points)
    print(f"Downsampled to {len(down_pts)} points")
    
    geoms = [down_pcd]
    
    for i,x in enumerate(best_aligned):
    # for x in expected_x_positions:
        window_pts = extract_window_points(down_pts, x_center=x, window=window)
        
        window_pcd = o3d.geometry.PointCloud()
        window_pcd.points = o3d.utility.Vector3dVector(window_pts)
        o3d.visualization.draw_geometries([window_pcd], window_name=f"Selected Window {i}")
        
        if i==1:
            window_pts = filter_window_pts_z_simple(window_pts, expected_diameter, support=False)
        else:
            window_pts = filter_window_pts_z_simple(window_pts, expected_diameter, support=True)
        
        window_pcd = o3d.geometry.PointCloud()
        window_pcd.points = o3d.utility.Vector3dVector(window_pts)
        o3d.visualization.draw_geometries([window_pcd], window_name=f"Filtered Window {i}")
        # window_pts = filter_window_pts_z(window_pts, expected_diameter)
        
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

def filter_duplicate_planes_by_alignment(planes, axis=np.array([1, 0, 0]), angle_thresh_deg=5.0, dist_thresh=1.0):
    axis = axis / np.linalg.norm(axis)
    angle_thresh_rad = np.radians(angle_thresh_deg)
    cos_thresh = np.cos(angle_thresh_rad)

    planes = list(planes)  # Ensure it's indexable
    used = set()
    final_planes = []
    retained_indices = []

    for i in range(len(planes)):
        if i in used:
            continue

        a1, b1, c1, d1 = planes[i]
        n1 = np.array([a1, b1, c1])
        n1 /= np.linalg.norm(n1)
        group = [i]

        for j in range(i + 1, len(planes)):
            if j in used:
                continue

            a2, b2, c2, d2 = planes[j]
            n2 = np.array([a2, b2, c2])
            n2 /= np.linalg.norm(n2)

            # Check angle between normals
            dot_product = np.dot(n1, n2)
            if abs(dot_product) >= cos_thresh:
                # Check distance between offsets
                if abs(d1 - d2) < dist_thresh:
                    group.append(j)

        # Keep the best-aligned plane in the group
        best_idx = max(
            group,
            key=lambda k: abs(np.dot(np.array(planes[k][:3]) / np.linalg.norm(planes[k][:3]), axis))
        )
        final_planes.append(planes[best_idx])
        retained_indices.append(best_idx)
        used.update(group)

    return final_planes, retained_indices

def detect_planes(target_axis=[1, 0, 0], angle_deg=3, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3_point_bend_flat_surfaces.stl"
    # voxel_size = 0.2
    
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(mesh_path)
    
    # Align the point cloud with PCA
    pcd = align_point_cloud_with_pca(base_pcd)
    
    # Rotate the point cloud about X so the greatest variation aligns with Z
    rotation_angle = np.radians(90)
    axis_angle = np.array([rotation_angle, 0, 0])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(rotation_matrix, center=(0,0,0))    
    
    target_axis = np.array(target_axis)
    # x_axis = np.array([1, 0, 0])
    angle_rad = np.radians(angle_deg)
    cos_angle_thresh = np.cos(angle_rad)
    planes = []
    plane_meshes = []

    pcd_copy = pcd.select_by_index([], invert=True)  # clone for modification

    while np.asarray(pcd_copy.points).shape[0] >= ransac_n:
        plane_model, inliers = pcd_copy.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        if len(inliers) == 0:
            break

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        # Check if the normal is within ±angle_deg of the X axis (either direction)
        dot = np.abs(np.dot(normal, target_axis))
        if dot < cos_angle_thresh:
            # Not close enough to X axis — skip
            pcd_copy = pcd_copy.select_by_index(inliers, invert=True)
            continue
        
        inlier_cloud = pcd_copy.select_by_index(inliers)
        
        # Generate plane mesh and save it
        plane_mesh = create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0)
        plane_meshes.append(plane_mesh)
        planes.append(plane_model)

        # Remove inliers and continue
        pcd_copy = pcd_copy.select_by_index(inliers, invert=True)
    
    filtered_planes, retained_idxs = filter_duplicate_planes_by_alignment(
        planes, axis=target_axis, angle_thresh_deg=5.0, dist_thresh=1.0
    )
    if len(filtered_planes) < len(planes):
        print(f"Filtered out {len(planes) - len(filtered_planes)} planes")
    
    print("Planes:")
    for plane in filtered_planes:
        print(plane)
    time.sleep(2)
    
    # Combine all for visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis] + [plane_meshes[i] for i in retained_idxs])

    return pcd, filtered_planes, retained_idxs, plane_meshes, inlier_cloud

def detect_fixture_planes(target_axes):
    keep_planes = []
    for target_axis in target_axes:
        pcd, filtered_planes, retained_idxs, plane_meshes, inlier_cloud = detect_planes(target_axis=target_axis)
        
        # Base of fixture
        if target_axis == [0, 0, 1]:
            base_plane = max(filtered_planes, key=lambda x: x[3]) # Select the maximum d value (the more positive d, the further below the origin)
            keep_planes.append(base_plane)
        else:
            for plane in filtered_planes:
                keep_planes.append(plane)
                
    time.sleep(2)
    
    # Combine all for visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis] + [plane_meshes[i] for i in retained_idxs])
    
    return keep_planes
            

if __name__ == "__main__":
    # find_cylinders()
    # target_axis = [np.sqrt(3)/2, -0.5, 0]   # One angled face of anvil
    # target_axis = [-np.sqrt(3)/2, -0.5, 0]  # Other angled face of anvil
    # target_axis = [1, 0, 0]     # X axis
    # target_axis = [0, 0, 1]     # Z axis
    # planes, inlier_clouds = detect_planes(target_axis=target_axis)
    
    target_axes = [[0, 0, 1],
                   [1, 0, 0],
                   [np.sqrt(3)/2, 0, -0.5],
                   [-np.sqrt(3)/2, 0, -0.5]]
    
    keep_planes = detect_fixture_planes(target_axes)