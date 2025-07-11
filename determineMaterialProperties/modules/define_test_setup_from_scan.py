import open3d as o3d
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from itertools import combinations
import time
import trimesh
from align_meshes import align_tgt_to_ref_meshes, load_mesh
import copy
from scipy.spatial import cKDTree
import pandas as pd
import os
import fixture_plane_fitting
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import pygad
from tqdm import tqdm

def load_mesh_as_point_cloud(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

def align_with_pca_separated(geometry, is_point_cloud=True):
    """
    Create separate translation and rotation matrices for PCA alignment
    
    Returns:
    - T_translate: 4x4 matrix that translates to centroid
    - T_rotate: 4x4 matrix that applies PCA rotation (around origin)
    - T_combined: 4x4 matrix that does both (translate then rotate)
    - rotation_matrix_3x3: 3x3 rotation matrix for downstream processes
    - centroid: the centroid point
    """
    if is_point_cloud:
        points = np.asarray(geometry.points)
    else:
        points = np.asarray(geometry.vertices)
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Perform PCA (same as manual method)
    pca = PCA(n_components=3)
    pca.fit(points - centroid)
    axes = pca.components_
    
    # Sort axes by explained variance (descending)
    order = np.argsort(pca.explained_variance_)[::-1]
    ordered_axes = axes[order]
    
    # Enforce right-handed coordinate system
    if np.linalg.det(ordered_axes) < 0:
        ordered_axes[2, :] *= -1
    
    # The 3x3 rotation matrix for downstream processes
    rotation_matrix_3x3 = ordered_axes.T
    
    # Create separate transformation matrices
    
    # 1. Translation matrix: moves geometry so centroid is at origin
    T_translate = np.eye(4)
    T_translate[:3, 3] = -centroid
    
    # 2. Rotation matrix: applies PCA rotation around origin
    T_rotate = np.eye(4)
    T_rotate[:3, :3] = rotation_matrix_3x3.T  # For Open3D transform
    
    # 3. Combined matrix (equivalent to original working version)
    T_combined = T_rotate @ T_translate
    
    return T_translate, T_rotate, T_combined, rotation_matrix_3x3, centroid

def align_with_pca(geometry, is_point_cloud=True):
    """
    Original working function - kept for compatibility
    """
    T_translate, T_rotate, T_combined, rotation_matrix_3x3, centroid = align_with_pca_separated(geometry, is_point_cloud)
    return T_combined, rotation_matrix_3x3, centroid

def align_point_cloud_with_pca_manual(pcd):
    """
    Your working function for reference - aligns point cloud manually
    """
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
    
    # Enforce right-handed coordinate system
    if np.linalg.det(ordered_axes) < 0:
        ordered_axes[2, :] *= -1
    
    # Apply rotation (this is what your function does)
    rotation_matrix_manual = ordered_axes.T
    rotated_points = centered_points @ rotation_matrix_manual
    
    # Create new point cloud
    pcd_pca = o3d.geometry.PointCloud()
    pcd_pca.points = o3d.utility.Vector3dVector(rotated_points)
    
    # Copy over colors if they exist
    if pcd.has_colors():
        pcd_pca.colors = pcd.colors
    if pcd.has_normals():
        pcd_pca.estimate_normals()
        
    return pcd_pca, rotation_matrix_manual, centroid

# def align_point_cloud_with_pca(pcd):
#     if not isinstance(pcd, o3d.geometry.PointCloud):
#         raise ValueError("Input must be an Open3D PointCloud.")

#     # Convert to numpy array
#     points = np.asarray(pcd.points)

#     # Center the points at origin
#     centroid = np.mean(points, axis=0)
#     centered_points = points - centroid

#     # Perform PCA
#     pca = PCA(n_components=3)
#     pca.fit(centered_points)
#     axes = pca.components_

#     # Sort axes by explained variance (descending)
#     order = np.argsort(pca.explained_variance_)[::-1]
#     ordered_axes = axes[order]

#     # Align principal components with +X, +Y, +Z
#     target_axes = np.eye(3)
#     rotation_matrix = ordered_axes.T @ target_axes

#     # Apply rotation
#     rotated_points = centered_points @ rotation_matrix

#     # Create new point cloud
#     pcd_pca = o3d.geometry.PointCloud()
#     pcd_pca.points = o3d.utility.Vector3dVector(rotated_points)

#     # Copy over colors if they exist
#     if pcd.has_colors():
#         pcd_pca.colors = pcd.colors
#     if pcd.has_normals():
#         # Recompute normals since they may no longer be valid
#         pcd_pca.estimate_normals()
        
#     # o3d.visualization.draw_geometries([pcd_pca], window_name="PCA Point Cloud")
    
#     return pcd_pca, rotation_matrix, centroid

# def align_mesh_with_pca(mesh):
#     if not isinstance(mesh, o3d.geometry.TriangleMesh):
#         raise ValueError("Input must be an Open3D TriangleMesh.")

#     # Convert vertices to numpy array
#     vertices = np.asarray(mesh.vertices)

#     # Center the mesh at the origin
#     centroid = np.mean(vertices, axis=0)
#     centered_vertices = vertices - centroid

#     # Perform PCA
#     pca = PCA(n_components=3)
#     pca.fit(centered_vertices)
#     axes = pca.components_

#     # Sort axes by explained variance (descending)
#     order = np.argsort(pca.explained_variance_)[::-1]
#     ordered_axes = axes[order]

#     # Align principal components with +X, +Y, +Z
#     target_axes = np.eye(3)
#     rotation_matrix = ordered_axes.T @ target_axes

#     # Apply rotation
#     rotated_vertices = centered_vertices @ rotation_matrix

#     # Create a new mesh with rotated vertices and same faces
#     aligned_mesh = o3d.geometry.TriangleMesh()
#     aligned_mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
#     aligned_mesh.triangles = mesh.triangles

#     # Recompute normals since geometry has changed
#     aligned_mesh.compute_vertex_normals()

#     return aligned_mesh


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

def normalize_plane(plane):
    normal = plane[:3]
    d = plane[3]
    norm = np.linalg.norm(normal)
    return normal / norm, d / norm

def plane_similarity(plane1, plane2):
    # Normalize both
    n1, d1 = normalize_plane(plane1)
    n2, d2 = normalize_plane(plane2)

    # Check both directions (n2 or -n2)
    dot_same = np.dot(n1, n2)
    dot_flipped = np.dot(n1, -n2)
    if abs(dot_same) > abs(dot_flipped):
        normal_diff = np.linalg.norm(n1 - n2)
        d_diff = abs(d1 - d2)
    else:
        normal_diff = np.linalg.norm(n1 + n2)
        d_diff = abs(d1 + d2)

    return normal_diff + d_diff  # Lower = more similar

def filter_duplicate_planes(planes, target_axis, d_threshold=1.0):
    """
    Filters out planes with similar 'd' values (i.e., close positions along their shared normal direction).
    Uses DBSCAN to cluster by 'd' and picks the plane in each cluster whose normal is most aligned with target_axis.

    Args:
        planes: (N, 4) numpy array of plane definitions [a, b, c, d]
        target_axis: list or np.array of shape (3,), the preferred alignment direction
        d_threshold: float, max absolute difference in 'd' value for clustering (adjust as needed)

    Returns:
        List of filtered plane definitions (each is a 4-element numpy array)
    """
    planes = np.array(planes)
    normals = planes[:, :3]
    d_values = planes[:, 3].reshape(-1, 1)

    # Cluster based on the d value only
    clustering = DBSCAN(eps=d_threshold, min_samples=1).fit(d_values)
    labels = clustering.labels_

    keep = []
    target_axis = np.array(target_axis) / np.linalg.norm(target_axis)

    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        best_idx = max(
            cluster_indices,
            key=lambda idx: abs(np.dot(normals[idx] / np.linalg.norm(normals[idx]), target_axis))
        )
        keep.append(planes[best_idx])

    return keep

def detect_and_correct_pca(pcd):
    def find_highest_y_peak(pcd, bin_width=0.25, min_count=20):
        """
        Find the highest Y-coordinate peak in a point cloud.
        
        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.
            bin_width (float): Width of histogram bins along y.
            min_count (int): Minimum number of points to consider a bin a peak.
        
        Returns:
            peak_y_center (float): The y value at the center of the highest peak bin.
            count (int): The number of points in that bin.
        """
        y_vals = np.asarray(pcd.points)[:, 1]
        y_min, y_max = np.min(y_vals), np.max(y_vals)

        bins = np.arange(y_min, y_max + bin_width, bin_width)
        hist, edges = np.histogram(y_vals, bins=bins)
        
        filtered_hist = gaussian_filter1d(hist, sigma=3)
        
        peak_indices, _ = find_peaks(filtered_hist, height=50)
        peak_heights = y_vals[peak_indices]
        
        # max_peak__height = np.max(peak_heights)
        
        peaks_max_idx = np.argmax(peak_heights)
        original_max_idx = peak_indices[peaks_max_idx]
        peak_y = y_vals[original_max_idx]
        
        
        # # Only consider bins with a meaningful number of points
        # peak_idx = None
        # for i in reversed(range(len(hist))):  # Search from highest y downward
        #     if hist[i] >= min_count:
        #         peak_idx = i
        #         break

        # if peak_idx is None:
        #     print("No significant peak found.")
        #     return None, 0

        # peak_y_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        return peak_y
    
    def find_y_peak(pcd, bins=50):
        """
        Find the y-coordinate with the highest density of points in a point cloud.
        
        Parameters:
        -----------
        pcd : array-like or object with points attribute
            Point cloud data. Can be a numpy array of shape (N, 3) or an object
            with a 'points' attribute (like Open3D point cloud)
        bins : int, optional
            Number of bins for the histogram (default: 50)
        
        Returns:
        --------
        peak_y : float
            Y-coordinate corresponding to the histogram peak with maximum density
        """
        
        # Extract points from point cloud (handle different input formats)
        if hasattr(pcd, 'points'):
            # Handle Open3D point cloud format
            points = np.asarray(pcd.points)
        else:
            # Assume it's already a numpy array
            points = np.asarray(pcd)
        
        # Filter y coordinates into numpy array
        y_vals = points[:, 1]  # Extract y coordinates (column 1)
        
        # Create histogram of y values
        hist_counts, bin_edges = np.histogram(y_vals, bins=bins)
        
        # Calculate bin centers from bin edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply Gaussian filter with sigma=3 to smooth the histogram
        filtered_hist = gaussian_filter1d(hist_counts.astype(float), sigma=3)
        
        # Find the bin center with the maximum count (peak)
        max_idx = np.argmax(filtered_hist)
        peak_y = bin_centers[max_idx]
        
        return peak_y
    
    peak_y = find_y_peak(pcd, bins=200)
    # peak_y = find_highest_y_peak(pcd)
    
    if peak_y < 0:
        rotation_angle = np.radians(180)
        axis_angle = np.array([rotation_angle, 0, 0])
        R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        pcd.rotate(R_flip, center=(0,0,0))
    else:
        R_flip = None
        
    return pcd, R_flip
    
def prepare_scan_orientation(base_pcd, is_point_cloud=True):
    # Align the point cloud with PCA
    T_translate, R_pca, T_combined, R_3x3, centroid = align_with_pca_separated(base_pcd, is_point_cloud=True)
    
    pcd = copy.deepcopy(base_pcd)
    pcd.transform(T_translate)
    pcd.transform(R_pca)
    
    # # Visualize the initial PCA alignment
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis], window_name="Initial PCA Alignment")
    
    pcd, R_flip = detect_and_correct_pca(pcd)
    
    # if R_flip is not None:
    #     # Visualize the initial PCA alignment
    #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd, axis], window_name="Flip Correction of PCA Alignment")
    
    R_pca = R_pca[:3,:3]
    
    print(f'\nPCA Rotation Matrix:\n{R_pca}')
    print(f'PCA Centroid:\t{centroid}')
    
    # Rotate the point cloud about X so the greatest variation aligns with Z
    rotation_angle = np.radians(90)
    axis_angle = np.array([rotation_angle, 0, 0])
    R_90X = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(R_90X, center=(0,0,0))
    
    return pcd, R_pca, R_90X, R_flip, centroid

def detect_planes(base_pcd, target_axis=[1, 0, 0], angle_deg=3, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    # Align the point cloud with PCA
    T_translate, R_pca, T_combined, R_3x3, centroid = align_with_pca_separated(base_pcd, is_point_cloud=True)
    
    pcd = copy.deepcopy(base_pcd)
    pcd.transform(T_translate)
    pcd.transform(R_pca)
    
    # # Visualize the initial PCA alignment
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis], window_name="Initial PCA Alignment")
    
    pcd, R_flip = detect_and_correct_pca(pcd)
    
    # if R_flip is not None:
    #     # Visualize the initial PCA alignment
    #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd, axis], window_name="Flip Correction of PCA Alignment")
    
    R_pca = R_pca[:3,:3]
    
    print(f'\nPCA Rotation Matrix:\n{R_pca}')
    print(f'PCA Centroid:\t{centroid}')
    
    # Rotate the point cloud about X so the greatest variation aligns with Z
    rotation_angle = np.radians(90)
    axis_angle = np.array([rotation_angle, 0, 0])
    R_90X = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(R_90X, center=(0,0,0))
    
    print(f'\n90 degree X rotation matrix:\n{R_90X}')
    
    R_PCA_90X = R_90X @ R_pca
    
    print(f'\nPCA Rotation with 90 degree X matrix:\n{R_PCA_90X}')
    
    target_axis = np.array(target_axis)
    # x_axis = np.array([1, 0, 0])
    angle_rad = np.radians(angle_deg)
    cos_angle_thresh = np.cos(angle_rad)
    planes = []
    plane_meshes = []

    pcd_copy = pcd.select_by_index([], invert=True)  # clone for modification
    
    inlier_clouds = []

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
        inlier_clouds.append(inlier_cloud)
        
        # Generate plane mesh and save it
        plane_mesh = create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0)
        plane_meshes.append(plane_mesh)
        planes.append(plane_model)

        # Remove inliers and continue
        pcd_copy = pcd_copy.select_by_index(inliers, invert=True)
        
    print(f'{len(planes)} planes detected for axis {target_axis}')
    
    filtered_planes, retained_idxs = filter_duplicate_planes_by_alignment(
        planes, axis=target_axis, angle_thresh_deg=5.0, dist_thresh=1.0
    )
    
    filtered_inlier_clouds = [inlier_clouds[i] for i in retained_idxs]
    
    print(f"\nPlanes found for target_axis {target_axis}:")
    if len(filtered_planes) < len(planes):
        print(f"Kept {len(filtered_planes)}/{len(planes)} planes")
    
    for plane in filtered_planes:
        print(plane)
    time.sleep(2)
    
    filtered_plane_meshes = [plane_meshes[i] for i in retained_idxs]
    return pcd, filtered_planes, retained_idxs, filtered_plane_meshes, filtered_inlier_clouds, R_pca, R_90X, R_flip, centroid

# def detect_fixture_planes(base_pcd, target_axes):
#     keep_planes = []
#     keep_plane_meshes = []
#     keep_inlier_clouds = []
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    
#     while len(keep_planes) != 7:
#     # while True:
#         for target_axis in target_axes:
#             max_retries = 5
#             success = False
        
#             for attempt in range(max_retries):
#                 try:
#                     pcd, filtered_planes, retained_idxs, plane_meshes, filtered_inlier_clouds, R_PCA, R_90X, R_flip, centroid = detect_planes(
#                         base_pcd, target_axis=target_axis)
        
#                     prev_len = len(filtered_planes)
        
#                     tmp_planes = []
#                     tmp_meshes = []
#                     tmp_clouds = []
        
#                     if target_axis == [0, 0, 1]:
#                         # base_plane = max(filtered_planes, key=lambda x: x[3])
#                         # base_idx = filtered_planes.index(base_plane)
                        
#                         base_idx = max(range(len(filtered_planes)), key=lambda i: filtered_planes[i][3])
#                         base_plane = filtered_planes[base_idx]
                        
#                         # original_idx = retained_idxs[base_idx]
        
#                         tmp_planes.append(base_plane)
#                         tmp_meshes.append(plane_meshes[base_idx])
#                         tmp_clouds.append(filtered_inlier_clouds[base_idx])
        
#                     elif target_axis == [1, 0, 0]:
#                         if len(filtered_planes) > 4:
#                             filtered_planes = filter_duplicate_planes(filtered_planes, target_axis)
#                             print(f'Further filtered from {prev_len} to {len(filtered_planes)} planes')
        
#                         if len(filtered_planes) == 4:
#                             for i, plane in enumerate(filtered_planes):
#                                 tmp_planes.append(plane)
#                                 tmp_meshes.append(plane_meshes[retained_idxs[i]])
#                                 tmp_clouds.append(filtered_inlier_clouds[retained_idxs[i]])
#                         else:
#                             raise ValueError(f"Expected 4 planes detected, but found {len(filtered_planes)}")
        
#                     else:
#                         for i, plane in enumerate(filtered_planes):
#                             tmp_planes.append(plane)
#                             tmp_meshes.append(plane_meshes[retained_idxs[i]])
#                             tmp_clouds.append(filtered_inlier_clouds[retained_idxs[i]])
        
#                     # All good, commit results
#                     keep_planes.extend(tmp_planes)
#                     keep_plane_meshes.extend(tmp_meshes)
#                     keep_inlier_clouds.extend(tmp_clouds)
        
#                     success = True
#                     break
        
#                 except Exception as e:
#                     print(f"[Attempt {attempt+1}/{max_retries}] Failed with error: {e}")
        
#             if not success:
#                 print(f"Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")
#                 # raise Exception(f"Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")
                
                
#         # break
#     time.sleep(2)
    
#     # Combine all for visualization
#     o3d.visualization.draw_geometries([pcd, axis] + keep_plane_meshes)
#     # o3d.visualization.draw_geometries([pcd, axis] + [plane_meshes[i] for i in retained_idxs])
    
#     return keep_planes, keep_inlier_clouds, pcd, R_PCA, R_90X, R_flip, centroid

def detect_fixture_planes(base_pcd, target_axes):
    # Define expected number of planes per axis
    expected_planes = {
        str([0, 0, 1]): 1,
        str([1, 0, 0]): 4,
        str([np.sqrt(3)/2, 0, -0.5]): 1,
        str([-np.sqrt(3)/2, 0, -0.5]): 1
    }
    
    keep_planes = []
    keep_plane_meshes = []
    keep_inlier_clouds = []
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    
    max_overall_retries = 10
    overall_attempt = 0
    
    while overall_attempt < max_overall_retries:
        overall_attempt += 1
        print(f"\n=== Overall attempt {overall_attempt}/{max_overall_retries} ===")
        
        # Reset for each overall attempt
        current_planes = []
        current_meshes = []
        current_clouds = []
        all_axes_successful = True
        
        for target_axis in target_axes:
            axis_key = str(target_axis)
            expected_count = expected_planes[axis_key]
            
            max_retries = 5
            axis_success = False
            
            print(f"\nProcessing axis {target_axis}, expecting {expected_count} planes")
            
            for attempt in range(max_retries):
                try:
                    pcd, filtered_planes, retained_idxs, plane_meshes, filtered_inlier_clouds, R_PCA, R_90X, R_flip, centroid = detect_planes(
                        base_pcd, target_axis=target_axis)
                    
                    prev_len = len(filtered_planes)
                    
                    # Process based on target axis
                    if target_axis == [0, 0, 1]:
                        if len(filtered_planes) >= 1:
                            # Select the base plane (highest d value)
                            base_idx = max(range(len(filtered_planes)), key=lambda i: filtered_planes[i][3])
                            selected_planes = [filtered_planes[base_idx]]
                            selected_meshes = [plane_meshes[base_idx]]
                            selected_clouds = [filtered_inlier_clouds[base_idx]]
                        else:
                            raise ValueError(f"Expected at least 1 plane for Z-axis, found {len(filtered_planes)}")
                    
                    elif target_axis == [1, 0, 0]:
                        # Apply additional filtering if too many planes
                        if len(filtered_planes) > 4:
                            filtered_planes_extra = filter_duplicate_planes(filtered_planes, target_axis)
                            print(f'Further filtered from {prev_len} to {len(filtered_planes_extra)} planes')
                            
                            if len(filtered_planes_extra) == 4:
                                selected_planes = filtered_planes_extra
                                # Map back to original indices
                                selected_meshes = []
                                selected_clouds = []
                                for filtered_plane in filtered_planes_extra:
                                    # Find the index of this plane in the original filtered_planes
                                    for i, orig_plane in enumerate(filtered_planes):
                                        if np.allclose(filtered_plane, orig_plane, atol=1e-6):
                                            selected_meshes.append(plane_meshes[i])
                                            selected_clouds.append(filtered_inlier_clouds[i])
                                            break
                            else:
                                raise ValueError(f"Expected 4 planes for X-axis after extra filtering, found {len(filtered_planes_extra)}")
                        elif len(filtered_planes) == 4:
                            selected_planes = filtered_planes
                            selected_meshes = plane_meshes
                            selected_clouds = filtered_inlier_clouds
                        else:
                            raise ValueError(f"Expected 4 planes for X-axis, found {len(filtered_planes)}")
                    
                    else:  # Angled axes
                        if len(filtered_planes) >= 1:
                            # For angled axes, take the first plane or the one with best alignment
                            target_axis_norm = np.array(target_axis) / np.linalg.norm(target_axis)
                            best_idx = max(range(len(filtered_planes)), 
                                         key=lambda i: abs(np.dot(filtered_planes[i][:3] / np.linalg.norm(filtered_planes[i][:3]), target_axis_norm)))
                            selected_planes = [filtered_planes[best_idx]]
                            selected_meshes = [plane_meshes[best_idx]]
                            selected_clouds = [filtered_inlier_clouds[best_idx]]
                        else:
                            raise ValueError(f"Expected at least 1 plane for axis {target_axis}, found {len(filtered_planes)}")
                    
                    # Validate we got the expected number
                    if len(selected_planes) != expected_count:
                        raise ValueError(f"Expected {expected_count} planes for axis {target_axis}, got {len(selected_planes)}")
                    
                    # If we get here, this axis was successful
                    current_planes.extend(selected_planes)
                    current_meshes.extend(selected_meshes)
                    current_clouds.extend(selected_clouds)
                    
                    print(f"✓ Successfully found {len(selected_planes)} planes for axis {target_axis}")
                    axis_success = True
                    break
                    
                except Exception as e:
                    print(f"[Attempt {attempt+1}/{max_retries}] Failed for axis {target_axis}: {e}")
            
            if not axis_success:
                print(f"✗ Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")
                all_axes_successful = False
                break  # Break out of axis loop
        
        if all_axes_successful:
            # All axes were successful, commit the results
            keep_planes = current_planes
            keep_plane_meshes = current_meshes
            keep_inlier_clouds = current_clouds
            
            total_planes = len(keep_planes)
            print(f"\n✓ SUCCESS: Found all required planes. Total: {total_planes}")
            print(f"Distribution: Z={sum(1 for i, axis in enumerate(target_axes) if axis == [0, 0, 1])}, "
                  f"X={sum(1 for i, axis in enumerate(target_axes) if axis == [1, 0, 0]) * 4}, "
                  f"Others={sum(1 for axis in target_axes if axis not in [[0, 0, 1], [1, 0, 0]])}")
            break
        else:
            print(f"✗ Overall attempt {overall_attempt} failed. Retrying...")
    
    if overall_attempt >= max_overall_retries:
        raise Exception(f"Failed to detect all required planes after {max_overall_retries} overall attempts.")
    
    # Visualization
    time.sleep(2)
    o3d.visualization.draw_geometries([pcd, axis] + keep_plane_meshes)
    
    return keep_planes, keep_inlier_clouds, pcd, R_PCA, R_90X, R_flip, centroid
            










class EfficientPlaneDetector:
    def __init__(self, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        
    def detect_all_planes_simultaneously(self, pcd, target_axes, expected_counts):
        """
        Strategy 1: Detect all planes at once, then assign to axes
        Most efficient for dense point clouds with many planes
        """
        print("=== Strategy 1: Simultaneous Detection ===")
        
        # Step 1: Detect ALL planes in the point cloud
        all_planes = []
        all_inliers = []
        pcd_working = pcd.select_by_index([], invert=True)  # clone
        
        max_total_planes = sum(expected_counts.values()) + 5  # buffer for extras
        
        for i in range(max_total_planes):
            if len(pcd_working.points) < self.ransac_n:
                break
                
            plane_model, inliers = pcd_working.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if len(inliers) < 50:  # minimum points for valid plane
                break
                
            all_planes.append(plane_model)
            all_inliers.append(inliers)
            pcd_working = pcd_working.select_by_index(inliers, invert=True)
        
        print(f"Detected {len(all_planes)} total planes")
        
        # Step 2: Assign planes to target axes based on normal alignment
        axis_assignments = self._assign_planes_to_axes(all_planes, target_axes, expected_counts)
        
        return axis_assignments, all_planes, all_inliers
    
    def detect_with_adaptive_parameters(self, pcd, target_axes, expected_counts):
        """
        Strategy 2: Adaptive parameter tuning per axis
        Adjusts RANSAC parameters based on expected number of planes
        """
        print("=== Strategy 2: Adaptive Parameters ===")
        
        results = {}
        
        for axis, expected_count in expected_counts.items():
            target_axis = eval(axis)  # Convert string back to list
            print(f"\nProcessing axis {target_axis}, expecting {expected_count} planes")
            
            # Adaptive parameters based on expected count
            if expected_count == 1:
                # For single planes, be more permissive
                distance_thresh = self.distance_threshold * 1.5
                iterations = self.num_iterations // 2
            elif expected_count == 4:
                # For multiple planes, be more strict
                distance_thresh = self.distance_threshold * 0.8
                iterations = self.num_iterations * 2
            else:
                distance_thresh = self.distance_threshold
                iterations = self.num_iterations
            
            # Detect planes for this axis
            planes = self._detect_planes_for_axis(
                pcd, target_axis, expected_count, 
                distance_thresh, iterations
            )
            
            results[axis] = planes
            
        return results
    
    def detect_with_region_growing(self, pcd, target_axes, expected_counts):
        """
        Strategy 3: Region growing approach
        More robust for noisy data or partial planes
        """
        print("=== Strategy 3: Region Growing ===")
        
        # Convert to numpy for easier manipulation
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        if normals is None:
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
        
        results = {}
        used_points = set()
        
        for axis, expected_count in expected_counts.items():
            target_axis = np.array(eval(axis))
            target_axis = target_axis / np.linalg.norm(target_axis)
            
            # Find candidate points aligned with this axis
            alignments = np.abs(np.dot(normals, target_axis))
            angle_threshold = np.cos(np.radians(15))  # 15 degree tolerance
            candidate_mask = alignments > angle_threshold
            
            # Remove already used points
            available_mask = np.ones(len(points), dtype=bool)
            if used_points:
                available_mask[list(used_points)] = False
            
            candidate_indices = np.where(candidate_mask & available_mask)[0]
            
            if len(candidate_indices) < 100:  # minimum points needed
                print(f"Warning: Only {len(candidate_indices)} candidate points for axis {target_axis}")
                results[axis] = []
                continue
            
            # Cluster candidate points to find plane regions
            candidate_points = points[candidate_indices]
            planes = self._cluster_points_into_planes(
                candidate_points, candidate_indices, target_axis, expected_count
            )
            
            results[axis] = planes
            
            # Mark points as used
            for plane_info in planes:
                used_points.update(plane_info['point_indices'])
        
        return results
        
    def project_points_onto_axis(self, points, axis):
        axis = axis / np.linalg.norm(axis)
        return np.dot(points, axis)
    
    def angle_between_vectors(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)
        return min(angle, np.pi - angle)
    
    def fit_ransac_plane(self, points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        try:
            plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
            return plane_model, inliers
        except:
            return None, []
    
    def remove_close_planes(self, planes, distance_threshold=0.01, angle_threshold=5):
        filtered = []
        for i, (n1, d1) in enumerate(planes):
            is_close = False
            for j, (n2, d2) in enumerate(filtered):
                angle = self.angle_between_vectors(n1, n2)
                dist = abs(d1 - d2)
                if angle < angle_threshold and dist < distance_threshold:
                    is_close = True
                    break
            if not is_close:
                filtered.append((n1, d1))
        return filtered
    
    def detect_planes_along_axes(self, pcd, target_axes, expected_counts,
                                 projection_window=0.05,
                                 n_centers=100,
                                 angle_tolerance_deg=10,
                                 max_iterations=5):
    
        points = np.asarray(pcd.points)
        all_keep_planes = {}
    
        for axis_idx, axis in enumerate(target_axes):
            print('')
            print('#'*60)
            print(f'Working on axis {axis}')
            keep_planes = []
            projection = self.project_points_onto_axis(points, axis)
            min_proj, max_proj = projection.min(), projection.max()
            target_count = expected_counts.get(axis_idx, 0)
    
            for attempt in range(max_iterations):
                keep_planes.clear()
                # Sweep the projected range in windows
                sweep_range = np.linspace(min_proj, max_proj, n_centers)
                # sweep_range = np.arange(min_proj, max_proj, n_centers)
                for center in sweep_range:
                    low = center - projection_window / 2
                    high = center + projection_window / 2
                    indices = np.where((projection >= low) & (projection <= high))[0]
                    
                    # print(f'{len(indices)} indices found between {low} and {high}')
    
                    if len(indices) < 100:
                        continue
    
                    sub_points = points[indices]
                    model, inliers = self.fit_ransac_plane(sub_points)
                    if model is None or len(inliers) < 100:
                        continue
                    
                    # print(f'\nModel found with {len(inliers)} inliers')
                    
                    normal = np.array(model[:3])
                    d = model[3]
                    angle = self.angle_between_vectors(normal, axis)
                    if np.degrees(angle) < angle_tolerance_deg:
                        keep_planes.append((normal, d))
                        print(f'\nModel is {np.degrees(angle)} away from target axis')
                        # print(f'\tKEPT THIS PLANE MODEL')
    
                keep_planes = self.remove_close_planes(keep_planes)
                if len(keep_planes) == target_count:
                    break
    
            all_keep_planes[axis_idx] = keep_planes
    
        return all_keep_planes
    
    def detect_with_hierarchical_ransac(self, pcd, target_axes, expected_counts):
        """
        Strategy 4: Hierarchical RANSAC
        Use coarse-to-fine approach for better efficiency
        """
        print("=== Strategy 4: Hierarchical RANSAC ===")
        
        results = {}
        
        # Sort axes by expected count (process easier cases first)
        sorted_axes = sorted(expected_counts.items(), key=lambda x: x[1])
        
        pcd_remaining = pcd.select_by_index([], invert=True)  # clone
        
        for axis_str, expected_count in sorted_axes:
            target_axis = np.array(eval(axis_str))
            
            print(f"\nProcessing axis {target_axis} (expecting {expected_count})")
            print(f"Points remaining: {len(pcd_remaining.points)}")
            
            if len(pcd_remaining.points) < self.ransac_n:
                results[axis_str] = []
                continue
            
            # Phase 1: Coarse detection with relaxed parameters
            coarse_planes = self._coarse_plane_detection(
                pcd_remaining, target_axis, expected_count * 2  # detect more than needed
            )
            
            # Phase 2: Fine-tune and select best planes
            fine_planes, used_indices = self._fine_tune_planes(
                pcd_remaining, coarse_planes, target_axis, expected_count
            )
            
            results[axis_str] = fine_planes
            
            # Remove used points for next iteration
            if used_indices:
                pcd_remaining = pcd_remaining.select_by_index(used_indices, invert=True)
        
        return results
    
    def _assign_planes_to_axes(self, planes, target_axes, expected_counts):
        """Assign detected planes to target axes based on normal alignment"""
        plane_normals = np.array([plane[:3] / np.linalg.norm(plane[:3]) for plane in planes])
        target_axes_norm = np.array([np.array(axis) / np.linalg.norm(axis) for axis in target_axes])
        
        # Calculate alignment scores (absolute dot product)
        alignment_matrix = np.abs(np.dot(plane_normals, target_axes_norm.T))
        
        assignments = {}
        used_planes = set()
        
        # Greedy assignment: for each axis, pick best aligned unused planes
        for i, axis in enumerate(target_axes):
            axis_str = str(axis)
            expected_count = expected_counts[axis_str]
            
            # Get alignment scores for unused planes
            available_planes = [j for j in range(len(planes)) if j not in used_planes]
            if not available_planes:
                assignments[axis_str] = []
                continue
            
            # Sort by alignment score
            scores = [(j, alignment_matrix[j, i]) for j in available_planes]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N planes
            selected = scores[:expected_count]
            selected_indices = [idx for idx, score in selected if score > 0.8]  # minimum alignment
            
            assignments[axis_str] = [planes[idx] for idx in selected_indices]
            used_planes.update(selected_indices)
        
        return assignments
    
    def _detect_planes_for_axis(self, pcd, target_axis, expected_count, distance_thresh, iterations):
        """Detect planes for a specific axis with adaptive parameters"""
        target_axis = np.array(target_axis) / np.linalg.norm(target_axis)
        angle_threshold = np.cos(np.radians(10))  # 10 degree tolerance
        
        planes = []
        pcd_working = pcd.select_by_index([], invert=True)
        
        # Try to find more planes than expected, then filter
        max_attempts = expected_count * 3
        
        for _ in range(max_attempts):
            if len(pcd_working.points) < self.ransac_n:
                break
            
            plane_model, inliers = pcd_working.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=self.ransac_n,
                num_iterations=iterations
            )
            
            if len(inliers) < 30:  # minimum plane size
                break
            
            # Check alignment with target axis
            normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])
            alignment = np.abs(np.dot(normal, target_axis))
            
            if alignment > angle_threshold:
                planes.append(plane_model)
                pcd_working = pcd_working.select_by_index(inliers, invert=True)
            else:
                # Remove some inliers but not all (partial removal strategy)
                remove_count = len(inliers) // 3
                remove_indices = np.random.choice(inliers, remove_count, replace=False)
                pcd_working = pcd_working.select_by_index(remove_indices, invert=True)
        
        # Post-process: select best planes if we have too many
        if len(planes) > expected_count:
            planes = self._select_best_planes(planes, target_axis, expected_count)
        
        return planes
    
    def _cluster_points_into_planes(self, points, point_indices, target_axis, expected_count):
        """Cluster points into plane regions using spatial clustering"""
        if len(points) < 100:
            return []
        
        # Project points onto plane perpendicular to target axis
        projection_matrix = np.eye(3) - np.outer(target_axis, target_axis)
        projected_points = np.dot(points, projection_matrix.T)
        
        # Cluster in 2D projected space
        if expected_count == 1:
            # Single cluster
            labels = np.zeros(len(points))
        else:
            # Use DBSCAN or KMeans
            clustering = DBSCAN(eps=2.0, min_samples=50).fit(projected_points)
            labels = clustering.labels_
        
        planes = []
        for label in np.unique(labels):
            if label == -1:  # noise in DBSCAN
                continue
                
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            cluster_indices = point_indices[cluster_mask]
            
            if len(cluster_points) < 50:
                continue
            
            # Fit plane to cluster
            plane_model = self._fit_plane_to_points(cluster_points)
            
            planes.append({
                'model': plane_model,
                'points': cluster_points,
                'point_indices': cluster_indices
            })
        
        return planes[:expected_count]  # Return only expected number
    
    def _coarse_plane_detection(self, pcd, target_axis, max_planes):
        """Coarse plane detection with relaxed parameters"""
        planes = []
        pcd_working = pcd.select_by_index([], invert=True)
        
        # Relaxed parameters for coarse detection
        coarse_distance_thresh = self.distance_threshold * 2
        coarse_iterations = self.num_iterations // 4
        
        for _ in range(max_planes):
            if len(pcd_working.points) < self.ransac_n:
                break
                
            plane_model, inliers = pcd_working.segment_plane(
                distance_threshold=coarse_distance_thresh,
                ransac_n=self.ransac_n,
                num_iterations=coarse_iterations
            )
            
            if len(inliers) < 20:
                break
            
            planes.append((plane_model, inliers))
            pcd_working = pcd_working.select_by_index(inliers, invert=True)
        
        return planes
    
    def _fine_tune_planes(self, pcd, coarse_planes, target_axis, expected_count):
        """Fine-tune coarse planes and select the best ones"""
        target_axis = np.array(target_axis) / np.linalg.norm(target_axis)
        
        refined_planes = []
        
        for plane_model, inliers in coarse_planes:
            # Check alignment
            normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])
            alignment = np.abs(np.dot(normal, target_axis))
            
            if alignment > 0.8:  # Good alignment
                # Refine with stricter parameters
                inlier_cloud = pcd.select_by_index(inliers)
                refined_model, refined_inliers = inlier_cloud.segment_plane(
                    distance_threshold=self.distance_threshold,
                    ransac_n=self.ransac_n,
                    num_iterations=self.num_iterations
                )
                
                if len(refined_inliers) > 30:
                    refined_planes.append((refined_model, refined_inliers, alignment))
        
        # Sort by alignment and size, take top N
        refined_planes.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
        
        selected_planes = refined_planes[:expected_count]
        all_used_indices = []
        
        for _, inliers, _ in selected_planes:
            all_used_indices.extend(inliers)
        
        return [plane[0] for plane in selected_planes], all_used_indices
    
    def _select_best_planes(self, planes, target_axis, expected_count):
        """Select best planes based on alignment and separation"""
        if len(planes) <= expected_count:
            return planes
        
        target_axis = np.array(target_axis) / np.linalg.norm(target_axis)
        
        # Calculate alignment scores
        scores = []
        for plane in planes:
            normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
            alignment = np.abs(np.dot(normal, target_axis))
            scores.append(alignment)
        
        # Sort by alignment score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Select top planes ensuring they're not too close to each other
        selected = [sorted_indices[0]]  # Always take the best
        
        for i in range(1, len(sorted_indices)):
            candidate_idx = sorted_indices[i]
            candidate_plane = planes[candidate_idx]
            
            # Check if this plane is sufficiently different from selected ones
            is_different = True
            for selected_idx in selected:
                selected_plane = planes[selected_idx]
                if abs(candidate_plane[3] - selected_plane[3]) < 1.0:  # too close in d
                    is_different = False
                    break
            
            if is_different:
                selected.append(candidate_idx)
                if len(selected) >= expected_count:
                    break
        
        return [planes[i] for i in selected]
    
    def _fit_plane_to_points(self, points):
        """Fit plane to point cluster using SVD"""
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # SVD to find normal
        U, S, Vt = np.linalg.svd(centered_points)
        normal = Vt[-1]  # Last row is the normal
        
        # Calculate d parameter
        d = -np.dot(normal, centroid)
        
        return np.array([normal[0], normal[1], normal[2], d])


# Usage example
def efficient_plane_detection_main(pcd, target_axes, expected_counts_dict):
    """
    Main function demonstrating all strategies
    """
    detector = EfficientPlaneDetector()
    
    # Try strategies in order of efficiency
    strategies = [
        detector.detect_all_planes_simultaneously,
        detector.detect_with_hierarchical_ransac,
        detector.detect_with_adaptive_parameters,
        detector.detect_with_region_growing
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            print(f"\n{'='*50}")
            print(f"Trying Strategy {i}")
            print(f"{'='*50}")
            
            results = strategy(pcd, target_axes, expected_counts_dict)
            
            # Validate results
            total_found = sum(len(planes) for planes in results.values())
            total_expected = sum(expected_counts_dict.values())
            
            if total_found >= total_expected * 0.8:  # 80% success rate
                print(f"Strategy {i} succeeded! Found {total_found}/{total_expected} planes")
                return results
            else:
                print(f"Strategy {i} partially succeeded: {total_found}/{total_expected}")
                
        except Exception as e:
            print(f"Strategy {i} failed: {e}")
            continue
    
    raise Exception("All strategies failed to find sufficient planes")
    
def detect_fixture_planes_efficient(pcd, target_axes):
    expected_counts = {
        str([0, 0, 1]): 1,
        str([1, 0, 0]): 4,
        str([np.sqrt(3)/2, 0, -0.5]): 1,
        str([-np.sqrt(3)/2, 0, -0.5]): 1
    }
    
    # # Apply your existing PCA alignment
    # pcd = apply_pca_alignment(base_pcd)  # your existing code
    
    # Use efficient detection
    # results = efficient_plane_detection_main(pcd, target_axes, expected_counts)
    
    detector = EfficientPlaneDetector()
    projection_window = 50.0
    n_centers = 50
    all_keep_planes = detector.detect_planes_along_axes(pcd,
                                                        target_axes,
                                                        expected_counts,
                                                        projection_window=projection_window,
                                                        n_centers=n_centers)
    
    return all_keep_planes




# Example usage:
# expected_counts = {
#     str([0, 0, 1]): 1,
#     str([1, 0, 0]): 4,
#     str([np.sqrt(3)/2, 0, -0.5]): 1,
#     str([-np.sqrt(3)/2, 0, -0.5]): 1
# }
# 
# results = efficient_plane_detection_main(pcd, target_axes, expected_counts)










def normalize_plane_append(plane):
    n = plane[:3]
    d = plane[3]
    norm = np.linalg.norm(n)
    return np.append(n / norm, d / norm)

def total_residual_loss(flat_params, inlier_clouds, plane_counts):
    num_planes = len(flat_params) // 4
    planes = [flat_params[i*4:(i+1)*4] for i in range(num_planes)]

    # Map each plane to its cloud index
    cloud_assignments = []
    for cloud_idx, count in enumerate(plane_counts):
        cloud_assignments.extend([cloud_idx] * count)

    loss = 0.0
    for i, plane in enumerate(planes):
        cloud = inlier_clouds[cloud_assignments[i]]
        n = plane[:3]
        n /= np.linalg.norm(n)
        d = plane[3]
        points = np.asarray(cloud.points)
        distances = points @ n + d
        loss += np.sum(distances ** 2)
    return loss

def constraint_orthogonal(plane1, plane2):
    n1 = plane1[:3] / np.linalg.norm(plane1[:3])
    n2 = plane2[:3] / np.linalg.norm(plane2[:3])
    return np.dot(n1, n2)

def constraint_unit_norm(plane):
    return np.linalg.norm(plane[:3]) - 1

def check_plane_point_residuals(planes, inlier_clouds, verbose=True, threshold=0.1):
    """
    For each plane and associated inlier cloud, compute the residual distance of each point to the plane.

    Args:
        planes (list of [a, b, c, d]): Plane coefficients.
        inlier_clouds (list of o3d.geometry.PointCloud): Point clouds associated with each plane.
        verbose (bool): If True, print mean and max residuals.
        threshold (float): Flag any planes with mean residuals exceeding this value.

    Returns:
        residual_stats (list of dict): Mean and max residuals per plane.
    """
    residual_stats = []

    for i, (plane, cloud) in enumerate(zip(planes, inlier_clouds)):
        n = np.array(plane[:3])
        d = plane[3]
        n = n / np.linalg.norm(n)
        points = np.asarray(cloud.points)

        distances = points @ n + d  # signed distances
        mean_res = np.mean(np.abs(distances))
        max_res = np.max(np.abs(distances))

        residual_stats.append({
            "plane_index": i,
            "mean_residual": mean_res,
            "max_residual": max_res,
            "num_points": len(points)
        })

        if verbose:
            status = "✅" if mean_res <= threshold else "⚠️"
            print(f"[{status}] Plane {i}: mean = {mean_res:.4f}, max = {max_res:.4f}, points = {len(points)}")

    return residual_stats

def optimize_planes_with_fixed_angle(plane1, plane2, cloud1, cloud2, target_angle_deg=60):
    """
    Optimize two planes to fit their point clouds,
    and constrain their normals to be separated by target_angle_deg degrees.
    Returns optimized (plane1, plane2) definitions.
    """

    target_cos = np.cos(np.radians(target_angle_deg))

    def loss(params):
        n1 = params[:3]
        n1 = n1 / np.linalg.norm(n1)
        n2 = params[3:6]
        n2 = n2 / np.linalg.norm(n2)
        d1 = params[6]
        d2 = params[7]

        pts1 = np.asarray(cloud1.points)
        pts2 = np.asarray(cloud2.points)

        res1 = pts1 @ n1 + d1
        res2 = pts2 @ n2 + d2

        return np.sum(res1**2) + np.sum(res2**2)

    def unit_norm_constraint1(params):
        return np.linalg.norm(params[:3]) - 1

    def unit_norm_constraint2(params):
        return np.linalg.norm(params[3:6]) - 1

    def angle_constraint(params):
        n1 = params[:3] / np.linalg.norm(params[:3])
        n2 = params[3:6] / np.linalg.norm(params[3:6])
        return np.dot(n1, n2) - target_cos

    # Initial guesses from current planes
    n1_init = plane1[:3] / np.linalg.norm(plane1[:3])
    n2_init = plane2[:3] / np.linalg.norm(plane2[:3])
    d1_init = plane1[3]
    d2_init = plane2[3]

    x0 = np.hstack([n1_init, n2_init, d1_init, d2_init])

    constraints = [
        {'type': 'eq', 'fun': unit_norm_constraint1},
        {'type': 'eq', 'fun': unit_norm_constraint2},
        {'type': 'eq', 'fun': angle_constraint},
    ]

    result = minimize(
        loss,
        x0=x0,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 500, 'disp': True}
    )

    if not result.success:
        print("Optimization failed:", result.message)

    n1_opt = result.x[:3] / np.linalg.norm(result.x[:3])
    n2_opt = result.x[3:6] / np.linalg.norm(result.x[3:6])
    d1_opt = result.x[6]
    d2_opt = result.x[7]

    plane1_opt = np.append(n1_opt, d1_opt)
    plane2_opt = np.append(n2_opt, d2_opt)

    return plane1_opt, plane2_opt

def angular_change_between_normals(n_orig, n_opt):
    n_orig = n_orig / np.linalg.norm(n_orig)
    n_opt = n_opt / np.linalg.norm(n_opt)
    dot_product = np.clip(np.dot(n_orig, n_opt), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# def identify_planes_along_x(planes, clouds):
#     """
#     planes: list of plane coefficients
#     clouds: list of corresponding point clouds
    
#     Returns indices of planes in order from min X to max X.
#     So returns 4 indices: outer_min, inner_min, inner_max, outer_max
#     """
#     # Compute centroid X for each cloud
#     centroids = [np.mean(np.asarray(c.points), axis=0) for c in clouds]
#     x_coords = [c[0] for c in centroids]
    
#     # Sort indices by X coordinate
#     sorted_indices = sorted(range(len(x_coords)), key=lambda i: x_coords[i])
    
#     for idx in sorted_indices:
#         print(f'\nX Plane Fit Coordinate: {x_coords[idx]}')    
#         print(f'Plane Definition: {planes[idx]}')
    
#     # Return all four sorted indices
#     return tuple(sorted_indices)  # will be 4 indices

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
    
    for idx in sorted_indices:
        print(f'\nX Plane Fit Coordinate: {x_coords[idx]}')    
        print(f'Plane Definition: {planes[idx]}')
    
    # Return all four sorted indices
    return tuple(sorted_indices)  # will be 4 indices



###############################################################################
# OPTIMIZE FOR ALL CONSTRAINTS

def normalize(v):
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)


def residual_loss(flat_params, inlier_clouds):
    loss = 0.0
    for i in range(len(inlier_clouds)):
        n = normalize(flat_params[i * 4:i * 4 + 3])
        d = flat_params[i * 4 + 3]
        pts = np.asarray(inlier_clouds[i].points)
        distances = pts @ n + d
        loss += np.sum(distances ** 2)
    return loss


def define_constraints():
    cons = []

    # Plane 0 orthogonal to planes 1 to 4
    for i in range(1, 5):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.dot(
                normalize(x[0:3]), normalize(x[i * 4:i * 4 + 3])
            )
        })

    # Planes 1 & 4 parallel: normals must match or be opposite
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[1 * 4:1 * 4 + 3]), normalize(x[4 * 4:4 * 4 + 3])
        ) - 1
    })

    # Planes 2 & 3 parallel
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[2 * 4:2 * 4 + 3]), normalize(x[3 * 4:3 * 4 + 3])
        ) - 1
    })

    # Planes 5 and 6: 60 degree angle between normals
    target_angle_rad = np.radians(60)
    target_dot = np.cos(target_angle_rad)
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[5 * 4:5 * 4 + 3]), normalize(x[6 * 4:6 * 4 + 3])
        ) - target_dot
    })

    # Line of intersection of planes 5 and 6 should be parallel to inner planes 2 & 3 (in XY only)
    def intersection_dir(x):
        n5 = normalize(x[5 * 4:5 * 4 + 3])
        n6 = normalize(x[6 * 4:6 * 4 + 3])
        direction = normalize(np.cross(n5, n6))
        return direction

    def dir_xy_parallel(x):
        d = intersection_dir(x)
        n2 = normalize(x[2 * 4:2 * 4 + 3])
        return np.dot(d[:2], n2[:2])

    cons.append({
        'type': 'eq',
        'fun': dir_xy_parallel
    })

    # Intersection line should have zero Z component
    cons.append({
        'type': 'eq',
        'fun': lambda x: intersection_dir(x)[2]
    })

    # Unit norm constraint for each plane normal
    for i in range(7):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.linalg.norm(x[i * 4:i * 4 + 3]) - 1
        })

    return cons


# def optimize_all_planes(keep_planes, keep_inlier_clouds):
#     """
#     Optimize planes with the following constraints:
#     - keep_planes[0] orthogonal to keep_planes[1:5]
#     - Among keep_planes[1:5], identify outer and inner pairs along X:
#        * outer two planes parallel
#        * inner two planes parallel
#     - keep_planes[5] and keep_planes[6] have 60 degrees separation
#     - Intersection line of planes 5 & 6 is parallel to the inner pair and has zero Z component
#     """
    
#     # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
#     idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(
#         keep_planes[1:5], keep_inlier_clouds[1:5]
#     )
#     # Map these local indices to global indices in keep_planes
#     x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]

#     # Flatten all planes into parameters for optimization
#     flat_params = np.hstack(keep_planes)

#     def constraint_func(params):
#         n_planes = len(params) // 4
#         planes = [params[i*4:(i+1)*4] for i in range(n_planes)]

#         constraints = []

#         # 1) keep_planes[0] orthogonal to all planes in keep_planes[1:5]
#         base_plane = planes[0]
#         for idx in x_plane_indices:
#             n_base = base_plane[:3] / np.linalg.norm(base_plane[:3])
#             n_other = planes[idx][:3] / np.linalg.norm(planes[idx][:3])
#             # Dot product should be zero for orthogonality
#             constraints.append(np.dot(n_base, n_other))

#         # 2) Outer pair of keep_planes[1:5] parallel
#         l_support = planes[x_plane_indices[0]][:3] / np.linalg.norm(planes[x_plane_indices[0]][:3])
#         r_support = planes[x_plane_indices[3]][:3] / np.linalg.norm(planes[x_plane_indices[3]][:3])
#         constraints.append(np.dot(l_support, r_support) - 1)  # parallel means dot == ±1, use +1 here and consider orientation fix if needed

#         # 3) Inner pair of keep_planes[1:5] parallel
#         l_anvil = planes[x_plane_indices[1]][:3] / np.linalg.norm(planes[x_plane_indices[1]][:3])
#         r_anvil = planes[x_plane_indices[2]][:3] / np.linalg.norm(planes[x_plane_indices[2]][:3])
#         constraints.append(np.dot(l_anvil, r_anvil) - 1)

#         # 4) keep_planes[5] and keep_planes[6] separated by 60 degrees
#         n5 = planes[5][:3] / np.linalg.norm(planes[5][:3])
#         n6 = planes[6][:3] / np.linalg.norm(planes[6][:3])
#         cos_60 = np.cos(np.radians(60))
#         constraints.append(np.dot(n5, n6) - cos_60)
        
#         # 5) Ensure sides of anvil and angled faces of anvil are 150 degrees apart
#         cos_150 = np.cos(np.radians(150))
#         constraints.append(np.dot(n5, l_anvil) - np.abs(cos_150))
#         constraints.append(np.dot(n6, r_anvil) - np.abs(cos_150))

#         # 6) Each plane normal must be unit length
#         for plane in planes:
#             constraints.append(np.linalg.norm(plane[:3]) - 1)
            
#         # # 7) Support planes must be separated by supports_distance
#         # n5 = planes[5][:3]
#         # n6 = planes[6][:3]
#         # d5 = planes[5][3]
#         # d6 = planes[6][3]
        
#         # # Compute unit normals
#         # n5_unit = normalize(n5)
#         # n6_unit = normalize(n6)
        
#         # # Mid-normal (should be parallel to both)
#         # n_avg = normalize((n5_unit + n6_unit) / 2)
        
#         # # Signed separation: project difference of d terms onto normal
#         # if d6 >= d5:
#         #     separation = (d6 - d5) / np.dot(n_avg, n5_unit)
#         # else:
#         #     separation = (d5 - d6) / np.dot(n_avg, n5_unit)
        
#         # # Enforce positive direction by sign convention
#         # constraints.append(separation - supports_distance)
        
        
                
#         # 8) Anvil vertical planes must be separated by 25 mm
#         l_anvil_plane = planes[x_plane_indices[1]]
#         r_anvil_plane = planes[x_plane_indices[2]]
        
#         na1 = l_anvil_plane[:3]
#         na2 = r_anvil_plane[:3]
#         da1 = l_anvil_plane[3]
#         da2 = r_anvil_plane[3]
        
#         na1_unit = normalize(na1)
#         na2_unit = normalize(na2)
#         na_avg = normalize((na1_unit + na2_unit) / 2)
        
#         if da1 >= da2:
#             anvil_sep = (da1 - da2) / np.dot(na_avg, na1_unit)
#         else:
#             anvil_sep = (da2 - da1) / np.dot(na_avg, na1_unit)
#         constraints.append(anvil_sep - 25.0)


#         return np.array(constraints)

#     # Define objective function: sum of squared residuals of points to their respective planes
#     def objective_func(params):
#         n_planes = len(params) // 4
#         planes = [params[i*4:(i+1)*4] for i in range(n_planes)]
#         loss = 0.0
#         for i, plane in enumerate(planes):
#             n = plane[:3]
#             n /= np.linalg.norm(n)
#             d = plane[3]
#             pts = np.asarray(keep_inlier_clouds[i].points)
#             res = pts @ n + d
#             loss += np.sum(res**2)
#         return loss

#     n_constraints = len(constraint_func(flat_params))
#     cons = [{'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]} for i in range(n_constraints)]
    
#     print("[INFO] Optimizing fit planes")
#     result = minimize(
#         objective_func,
#         flat_params,
#         method='SLSQP',
#         constraints=cons,
#         options={'maxiter': 500,
#                  'disp': True,
#                  'ftol': 1e-10}
#     )

#     if not result.success:
#         print("Optimization failed:", result.message)

#     optimized_planes = [result.x[i*4:(i+1)*4] for i in range(len(keep_planes))]
    
#     # After optimization, create plane meshes for visualization
#     optimized_plane_meshes = []
#     for plane, cloud in zip(optimized_planes, keep_inlier_clouds):
#         mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
#         optimized_plane_meshes.append(mesh)

#     return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds

def optimize_all_planes(keep_planes, keep_inlier_clouds):
    """
    Optimize planes with the following constraints:
    - keep_planes[0] normal aligned with Z axis (preserving initial direction)
    - keep_planes[1:5] normals aligned with X axis (preserving initial directions)
    - Among keep_planes[1:5], identify outer and inner pairs along X:
       * outer two planes parallel (same or opposite direction based on initial)
       * inner two planes parallel (same or opposite direction based on initial)
    - keep_planes[5] and keep_planes[6] have 60 degrees separation
    - Intersection line of planes 5 & 6 is parallel to the inner pair and has zero Z component
    """
    
    # Determine target directions for axis alignment based on initial normals
    def get_target_direction(normal, target_axis):
        """Determine whether normal should align with +axis or -axis"""
        normal_unit = normal / np.linalg.norm(normal)
        dot_pos = np.dot(normal_unit, target_axis)
        dot_neg = np.dot(normal_unit, -target_axis)
        return target_axis if dot_pos > dot_neg else -target_axis
    
    # Define target axes
    z_axis = np.array([0, 0, 1])
    x_axis = np.array([1, 0, 0])
    
    # Determine target directions for each plane
    z_target = get_target_direction(keep_planes[0][:3], z_axis)
    x_targets = [get_target_direction(keep_planes[i][:3], x_axis) for i in range(1, 6)]
    
    # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(keep_planes[1:5])
    # idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(
    #     keep_planes[1:5], keep_inlier_clouds[1:5]
    # )
    # Map these local indices to global indices in keep_planes
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    # Determine parallel relationship for outer and inner pairs
    def should_be_same_direction(normal1, normal2):
        """Check if two normals should point in same direction (dot > 0) or opposite"""
        n1_unit = normal1 / np.linalg.norm(normal1)
        n2_unit = normal2 / np.linalg.norm(normal2)
        return np.dot(n1_unit, n2_unit) > 0
    
    # Check initial relationships
    outer_same_dir = should_be_same_direction(
        keep_planes[x_plane_indices[0]][:3], 
        keep_planes[x_plane_indices[3]][:3]
    )
    inner_same_dir = should_be_same_direction(
        keep_planes[x_plane_indices[1]][:3], 
        keep_planes[x_plane_indices[2]][:3]
    )

    # Flatten all planes into parameters for optimization
    flat_params = np.hstack(keep_planes)

    def constraint_func(params):
        n_planes = len(params) // 4
        planes = [params[i*4:(i+1)*4] for i in range(n_planes)]

        constraints = []

        # 1) keep_planes[0] normal aligned with Z axis (preserving direction)
        base_normal = planes[0][:3]
        base_normal_unit = base_normal / np.linalg.norm(base_normal)
        # Constrain to be exactly aligned with target Z direction
        constraints.extend([
            base_normal_unit[0],  # x component = 0
            base_normal_unit[1],  # y component = 0
            base_normal_unit[2] - z_target[2]  # z component = ±1
        ])

        # 2) keep_planes[1:5] normals aligned with X axis (preserving directions)
        for i, global_idx in enumerate(x_plane_indices):
            normal = planes[global_idx][:3]
            normal_unit = normal / np.linalg.norm(normal)
            target_x = x_targets[global_idx - 1]  # -1 because x_targets is indexed from keep_planes[1]
            constraints.extend([
                normal_unit[1],  # y component = 0
                normal_unit[2],  # z component = 0
                normal_unit[0] - target_x[0]  # x component = ±1
            ])

        # 3) Outer pair of keep_planes[1:5] parallel
        n_outer1 = planes[x_plane_indices[0]][:3] / np.linalg.norm(planes[x_plane_indices[0]][:3])
        n_outer2 = planes[x_plane_indices[3]][:3] / np.linalg.norm(planes[x_plane_indices[3]][:3])
        if outer_same_dir:
            constraints.append(np.dot(n_outer1, n_outer2) - 1)
        else:
            constraints.append(np.dot(n_outer1, n_outer2) + 1)

        # 4) Inner pair of keep_planes[1:5] parallel
        n_inner1 = planes[x_plane_indices[1]][:3] / np.linalg.norm(planes[x_plane_indices[1]][:3])
        n_inner2 = planes[x_plane_indices[2]][:3] / np.linalg.norm(planes[x_plane_indices[2]][:3])
        if inner_same_dir:
            constraints.append(np.dot(n_inner1, n_inner2) - 1)
        else:
            constraints.append(np.dot(n_inner1, n_inner2) + 1)

        # 5) keep_planes[5] and keep_planes[6] separated by 60 degrees
        n5 = planes[5][:3] / np.linalg.norm(planes[5][:3])
        n6 = planes[6][:3] / np.linalg.norm(planes[6][:3])
        cos_60 = np.cos(np.radians(60))
        constraints.append(np.dot(n5, n6) - cos_60)
        
        # 6) Ensure sides of anvil and angled faces of anvil are 150 degrees apart
        cos_150 = np.cos(np.radians(150))
        l_anvil = planes[x_plane_indices[1]][:3] / np.linalg.norm(planes[x_plane_indices[1]][:3])
        r_anvil = planes[x_plane_indices[2]][:3] / np.linalg.norm(planes[x_plane_indices[2]][:3])
        constraints.append(np.dot(n5, l_anvil) - np.abs(cos_150))
        constraints.append(np.dot(n6, r_anvil) - np.abs(cos_150))

        # 7) Each plane normal must be unit length
        for plane in planes:
            constraints.append(np.linalg.norm(plane[:3]) - 1)
                        
        # 8) Anvil vertical planes must be separated by 25 mm
        l_anvil_plane = planes[x_plane_indices[1]]
        r_anvil_plane = planes[x_plane_indices[2]]
        
        na1 = l_anvil_plane[:3]
        na2 = r_anvil_plane[:3]
        da1 = l_anvil_plane[3]
        da2 = r_anvil_plane[3]
        
        na1_unit = normalize(na1)
        na2_unit = normalize(na2)
        na_avg = normalize((na1_unit + na2_unit) / 2)
        
        if da1 >= da2:
            anvil_sep = (da1 - da2) / np.dot(na_avg, na1_unit)
        else:
            anvil_sep = (da2 - da1) / np.dot(na_avg, na1_unit)
        constraints.append(anvil_sep - 25.0)

        return np.array(constraints)

    # Define objective function: sum of squared residuals of points to their respective planes
    def objective_func(params):
        n_planes = len(params) // 4
        planes = [params[i*4:(i+1)*4] for i in range(n_planes)]
        loss = 0.0
        for i, plane in enumerate(planes):
            n = plane[:3]
            n /= np.linalg.norm(n)
            d = plane[3]
            pts = np.asarray(keep_inlier_clouds[i].points)
            res = pts @ n + d
            loss += np.sum(res**2)
        return loss

    n_constraints = len(constraint_func(flat_params))
    cons = [{'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]} for i in range(n_constraints)]
    
    print(f"[INFO] Target Z direction: {z_target}")
    print(f"[INFO] Target X directions: {[x_targets[i] for i in range(len(x_plane_indices))]}")
    print(f"[INFO] Outer planes same direction: {outer_same_dir}")
    print(f"[INFO] Inner planes same direction: {inner_same_dir}")
    print("[INFO] Optimizing fit planes")
    
    result = minimize(
        objective_func,
        flat_params,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 500,
                 'disp': True,
                 'ftol': 1e-10}
    )

    if not result.success:
        print("Optimization failed:", result.message)

    optimized_planes = [result.x[i*4:(i+1)*4] for i in range(len(keep_planes))]
    
    # After optimization, create plane meshes for visualization
    optimized_plane_meshes = []
    for plane, cloud in zip(optimized_planes, keep_inlier_clouds):
        mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
        optimized_plane_meshes.append(mesh)

    return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds


def optimize_all_planes_genetic_v2(keep_planes, keep_inlier_clouds):
    ### Prep keep_planes so all similar planes have their normals facing in the
    ### axis positive direction
    # Define target axes
    z_axis = [0, 0, 1]
    x_axis = [1, 0, 0]
    
    # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(keep_planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    # Determine bounds on individual genes based on keep_planes
    base_plane = keep_planes[0]
    outer_plane1 = keep_planes[x_plane_indices[0]]
    outer_plane2 = keep_planes[x_plane_indices[3]]
    inner_plane1 = keep_planes[x_plane_indices[1]]
    inner_plane2 = keep_planes[x_plane_indices[2]]
    anvil_plane1 = keep_planes[5]
    anvil_plane2 = keep_planes[6]
    
    planes = [copy.deepcopy(base_plane),
              copy.deepcopy(outer_plane1),
              copy.deepcopy(outer_plane2),
              copy.deepcopy(inner_plane1),
              copy.deepcopy(inner_plane2),
              copy.deepcopy(anvil_plane1),
              copy.deepcopy(anvil_plane2)]
    
    def flip_plane_to_face_positive_axis(plane, target_axis=[1.0, 0.0, 0.0]):
        normal = plane[:3]
        target_axis = np.asarray(target_axis)
        if np.dot(normal, target_axis) < 0:
            # Flip the normal and d
            return np.hstack([-normal, -plane[3]])
        else:
            return plane
    
    # Redefine planes so they are aligned roughly with positive axis definitions
    base_plane = flip_plane_to_face_positive_axis(base_plane, z_axis)
    outer_plane1 = flip_plane_to_face_positive_axis(outer_plane1, x_axis)
    outer_plane2 = flip_plane_to_face_positive_axis(outer_plane2, x_axis)
    inner_plane1 = flip_plane_to_face_positive_axis(inner_plane1, x_axis)
    inner_plane2 = flip_plane_to_face_positive_axis(inner_plane2, x_axis)
    anvil_plane1 = flip_plane_to_face_positive_axis(anvil_plane1, x_axis)
    anvil_plane2 = flip_plane_to_face_positive_axis(anvil_plane2, x_axis)
    
    flipped_planes = [base_plane,
              outer_plane1,
              outer_plane2,
              inner_plane1,
              inner_plane2,
              anvil_plane1,
              anvil_plane2]
    
    for i in range(len(planes)):
        if planes[i] != flipped_planes[i]:
            print(f'Flipped plane {i}')
        else:
            print(f'Plane {i} was left as-is')
    
    
    ### Determine the bounds on the optimization
    def plane_to_spherical(plane):
        """
        Convert plane [a, b, c, d] to spherical coordinates [theta, phi, distance].
        
        Args:
            plane: [a, b, c, d] where ax + by + cz + d = 0
        
        Returns:
            [theta, phi, distance] where:
            - theta: azimuthal angle (0 to 2π) around z-axis
            - phi: polar angle (0 to π) from positive z-axis
            - distance: absolute distance from origin to plane
        """
        a, b, c, d = plane
        
        # Normalize the normal vector
        normal = np.array([a, b, c])
        normal_magnitude = np.linalg.norm(normal)
        
        if normal_magnitude < 1e-12:
            raise ValueError("Invalid plane: normal vector has zero magnitude")
        
        # Ensure normal points away from origin (positive distance)
        if d > 0:
            normal = -normal
            d = -d
        
        unit_normal = normal / normal_magnitude
        
        # Convert to spherical coordinates
        x, y, z = unit_normal
        
        # Calculate spherical coordinates
        # theta: azimuthal angle (atan2(y, x))
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi  # Ensure theta is in [0, 2π]
        
        # phi: polar angle (arccos(z))
        phi = np.arccos(np.clip(z, -1, 1))
        
        # Distance from origin to plane
        distance = abs(d) / normal_magnitude
        
        return np.array([theta, phi, distance])
    
    def spherical_to_plane(spherical_coords):
        """
        Convert spherical coordinates [theta, phi, distance] to plane [a, b, c, d].
        
        Args:
            spherical_coords: [theta, phi, distance] where:
            - theta: azimuthal angle (0 to 2π) around z-axis
            - phi: polar angle (0 to π) from positive z-axis  
            - distance: absolute distance from origin to plane
        
        Returns:
            [a, b, c, d] where ax + by + cz + d = 0
        """
        theta, phi, distance = spherical_coords
        
        # Convert spherical to Cartesian (unit normal)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Create plane equation: ax + by + cz + d = 0
        # where [a, b, c] is the unit normal and d = -distance
        a, b, c = x, y, z
        d = -distance  # Negative because we want normal pointing away from origin
        
        return np.array([a, b, c, d])
    
    def compute_spherical_bounds(theta_center, phi_center, theta_tol, phi_tol, distance_center, distance_tol):
        """
        Compute bounds for spherical coordinates with angular and distance tolerances.
        
        Args:
            theta_center: Central azimuthal angle (radians)
            phi_center: Central polar angle (radians)
            theta_tol: Angular tolerance in theta direction (radians)
            phi_tol: Angular tolerance in phi direction (radians)
            distance_center: Central distance
            distance_tol: Distance tolerance
        
        Returns:
            Dictionary with bounds for each parameter
        """
        # Theta bounds (with wraparound handling)
        theta_min = theta_center - theta_tol
        theta_max = theta_center + theta_tol
        
        # Handle wraparound at 0/2π
        if theta_min < 0:
            theta_min += 2 * np.pi
        if theta_max > 2 * np.pi:
            theta_max -= 2 * np.pi
        
        # Phi bounds (clamped to [0, π])
        phi_min = max(0, phi_center - phi_tol)
        phi_max = min(np.pi, phi_center + phi_tol)
        
        # Distance bounds (must be positive)
        distance_min = max(0, distance_center - distance_tol)
        distance_max = distance_center + distance_tol
        
        bounds = {
            'theta': (theta_min, theta_max),
            'phi': (phi_min, phi_max),
            'distance': (distance_min, distance_max),
            'wraparound_theta': theta_min > theta_max  # Flag for wraparound case
        }
        
        return bounds
    
    def compute_angular_bounds_from_cone(theta_center, phi_center, max_angular_deviation):
        """
        Compute spherical bounds that approximate a cone of allowed normals.
        
        Args:
            theta_center: Central azimuthal angle (radians)
            phi_center: Central polar angle (radians)
            max_angular_deviation: Maximum angular deviation from center (radians)
        
        Returns:
            Dictionary with bounds that approximate the cone
        """
        # For small angles, approximate the cone with rectangular bounds
        # This is more accurate than using the same tolerance for both theta and phi
        
        # Phi tolerance is straightforward
        phi_tol = max_angular_deviation
        
        # Theta tolerance depends on phi (gets larger near poles)
        if abs(np.sin(phi_center)) < 1e-10:
            # Near poles, theta is essentially unconstrained
            theta_tol = np.pi
        else:
            # Away from poles, theta tolerance is scaled by sin(phi)
            theta_tol = max_angular_deviation / abs(np.sin(phi_center))
            theta_tol = min(theta_tol, np.pi)  # Clamp to reasonable value
        
        bounds = {
            'theta': (theta_center - theta_tol, theta_center + theta_tol),
            'phi': (max(0, phi_center - phi_tol), min(np.pi, phi_center + phi_tol)),
            'angular_deviation': max_angular_deviation
        }
        
        return bounds
    
    def validate_angular_constraint(spherical_coords, spherical_center, max_angular_deviation):
        """
        Check if a spherical coordinate point satisfies the angular constraint.
        
        Args:
            spherical_coords: [theta, phi, distance] to check
            spherical_center: [theta_center, phi_center, distance_center] reference
            max_angular_deviation: Maximum allowed angular deviation (radians)
        
        Returns:
            (is_valid, actual_deviation)
        """
        # Convert both to unit normals
        normal1 = spherical_to_cartesian_unit(spherical_coords[0], spherical_coords[1])
        normal2 = spherical_to_cartesian_unit(spherical_center[0], spherical_center[1])
        
        # Calculate actual angular deviation
        cos_angle = np.clip(np.dot(normal1, normal2), -1, 1)
        actual_deviation = np.arccos(cos_angle)
        
        is_valid = actual_deviation <= max_angular_deviation
        
        return is_valid, actual_deviation
    
    def spherical_to_cartesian_unit(theta, phi):
        """Convert spherical coordinates to unit Cartesian vector"""
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z])
    
    base_plane_sphere = plane_to_spherical(base_plane)
    outer_plane1_sphere = plane_to_spherical(outer_plane1)
    outer_plane2_sphere = plane_to_spherical(outer_plane2)
    inner_plane1_sphere = plane_to_spherical(inner_plane1)
    inner_plane2_sphere = plane_to_spherical(inner_plane2)
    anvil_plane1_sphere = plane_to_spherical(anvil_plane1)
    anvil_plane2_sphere = plane_to_spherical(anvil_plane2)
    
    d_tol = 10.0
    angle_tol = np.radians(5.0)
    
    base_plane_bounds = compute_angular_bounds_from_cone(base_plane_sphere[0], base_plane_sphere[1], angle_tol)
    outer_plane1_bounds = compute_angular_bounds_from_cone(outer_plane1_sphere[0], outer_plane1_sphere[1], angle_tol)
    outer_plane2_bounds = compute_angular_bounds_from_cone(outer_plane2_sphere[0], outer_plane2_sphere[1], angle_tol)
    inner_plane1_bounds = compute_angular_bounds_from_cone(inner_plane1_sphere[0], inner_plane1_sphere[1], angle_tol)
    inner_plane2_bounds = compute_angular_bounds_from_cone(inner_plane2_sphere[0], inner_plane2_sphere[1], angle_tol)
    anvil_plane1_bounds = compute_angular_bounds_from_cone(anvil_plane1_sphere[0], anvil_plane1_sphere[1], angle_tol)
    anvil_plane2_bounds = compute_angular_bounds_from_cone(anvil_plane2_sphere[0], anvil_plane2_sphere[1], angle_tol)
    
    
    # base_plane_nmin, base_plane_nmax = compute_normal_component_bounds(base_plane[:3], angle_tol)
    # outer_plane1_nmin, outer_plane1_nmax = compute_normal_component_bounds(outer_plane1[:3], angle_tol)
    # outer_plane2_nmin, outer_plane2_nmax = compute_normal_component_bounds(outer_plane2[:3], angle_tol)
    # inner_plane1_nmin, inner_plane1_nmax = compute_normal_component_bounds(inner_plane1[:3], angle_tol)
    # inner_plane2_nmin, inner_plane2_nmax = compute_normal_component_bounds(inner_plane2[:3], angle_tol)
    # anvil_plane1_nmin, anvil_plane1_nmax = compute_normal_component_bounds(anvil_plane1[:3], angle_tol)
    # anvil_plane2_nmin, anvil_plane2_nmax = compute_normal_component_bounds(anvil_plane2[:3], angle_tol)
    
    base_plane_dmin = base_plane_sphere[2] - d_tol
    base_plane_dmax = base_plane_sphere[2] + d_tol
    outer_plane1_dmin = outer_plane1_sphere[2] - d_tol
    outer_plane1_dmax = outer_plane1_sphere[2] + d_tol
    outer_plane2_dmin = outer_plane2_sphere[2] - d_tol
    outer_plane2_dmax = outer_plane2_sphere[2] + d_tol
    inner_plane1_dmin = inner_plane1_sphere[2] - d_tol
    inner_plane1_dmax = inner_plane1_sphere[2] + d_tol
    inner_plane2_dmin = inner_plane2_sphere[2] - d_tol
    inner_plane2_dmax = inner_plane2_sphere[2] + d_tol
    anvil_plane1_dmin = anvil_plane1_sphere[2] - d_tol
    anvil_plane1_dmax = anvil_plane1_sphere[2] + d_tol
    anvil_plane2_dmin = anvil_plane2_sphere[2] - d_tol
    anvil_plane2_dmax = anvil_plane2_sphere[2] + d_tol
    
    
    ### Objective
        # Build planes from params
        # Calculate fitting error
    params = [base_plane_sphere[0],
              base_plane_sphere[1],
              base_plane_sphere[2],
              outer_plane1_sphere[2],
              outer_plane2_sphere[2],
              inner_plane1_sphere[2],
              anvil_plane1_sphere[2],
              anvil_plane2_sphere[2]]
    
    gene_space = [{'low': base_plane_bounds['theta'][0], 'high': base_plane_bounds['theta'][1]},
                  {'low': base_plane_bounds['phi'][0], 'high': base_plane_bounds['phi'][1]},
                  {'low': base_plane_dmin, 'high': base_plane_dmax},
                  {'low': outer_plane1_dmin, 'high': outer_plane1_dmax},
                  {'low': outer_plane2_dmin, 'high': outer_plane2_dmax},
                  {'low': inner_plane1_dmin, 'high': inner_plane1_dmax},
                  {'low': anvil_plane1_dmin, 'high': anvil_plane1_dmax},
                  {'low': anvil_plane2_dmin, 'high': anvil_plane2_dmax},
                  ]
    
    def create_planes_from_params(params, verify=False):
        """
        Create plane definitions from parameter values.
        
        Each plane is represented as [a, b, c, d] where ax + by + cz + d = 0
        and (a, b, c) is the unit normal vector.
        
        Args:
            params: List of 9 float values as defined in the problem
            
        Returns:
            Dictionary containing all plane definitions
        """
        
        # Extract base plane from first 3 parameters
        base_plane = np.array(params[:3])
        base_plane_d = base_plane[2]
        
        
        # Extract d-values (distances) from remaining parameters
        outer_plane1_d = params[3]
        outer_plane2_d = params[4]
        inner_plane1_d = params[5]
        anvil_plane1_d = params[6]
        anvil_plane2_d = params[7]
        
        
        
        # Get base plane normal (already normalized since it's a plane equation)
        base_normal = spherical_to_cartesian_unit(base_plane[0], base_plane[1])
        base_plane = np.append(base_normal, np.array([base_plane_d]))
        
        # Helper function to find a vector orthogonal to base_normal
        def find_orthogonal_vector(normal):
            """Find a unit vector orthogonal to the given normal"""
            # Choose a vector that's not parallel to normal
            if abs(normal[0]) < 0.9:
                temp = np.array([1.0, 0.0, 0.0])
            else:
                temp = np.array([0.0, 1.0, 0.0])
            
            # Use cross product to get orthogonal vector
            orthogonal = np.cross(normal, temp)
            return orthogonal / np.linalg.norm(orthogonal)
        
        # Helper function to create another orthogonal vector
        def find_second_orthogonal_vector(normal, first_orthogonal):
            """Find a second unit vector orthogonal to both normal and first_orthogonal"""
            second_orthogonal = np.cross(normal, first_orthogonal)
            return second_orthogonal / np.linalg.norm(second_orthogonal)
        
        # Create two orthogonal vectors in the plane perpendicular to base_normal
        orth1 = find_orthogonal_vector(base_normal)
        orth2 = find_second_orthogonal_vector(base_normal, orth1)
        
        # Create outer planes (parallel to each other, orthogonal to base)
        outer_normal = orth1  # Use first orthogonal vector
        outer_plane1 = np.array([outer_normal[0], outer_normal[1], outer_normal[2], outer_plane1_d])
        outer_plane2 = np.array([outer_normal[0], outer_normal[1], outer_normal[2], outer_plane2_d])
        
        # Create inner planes (parallel to each other, orthogonal to base)
        inner_normal = orth2  # Use second orthogonal vector
        inner_plane1 = np.array([inner_normal[0], inner_normal[1], inner_normal[2], inner_plane1_d])
        
        # Calculate inner_plane2_d to ensure 25.0 separation
        # Distance between parallel planes ax + by + cz + d1 = 0 and ax + by + cz + d2 = 0
        # is |d2 - d1| / sqrt(a² + b² + c²). Since normal is unit vector, denominator is 1.
        inner_plane2_d = inner_plane1_d + 25.0  # Add 25.0 to create separation
        inner_plane2 = np.array([inner_normal[0], inner_normal[1], inner_normal[2], inner_plane2_d])
        
        # Create anvil planes with 60-degree angles
        def create_rotated_normal(reference_normal, angle_degrees):
            """Create a normal vector rotated by specified angle from reference"""
            # Create rotation axis (perpendicular to reference normal)
            if abs(reference_normal[0]) < 0.9:
                temp_vec = np.array([1.0, 0.0, 0.0])
            else:
                temp_vec = np.array([0.0, 1.0, 0.0])
            
            rotation_axis = np.cross(reference_normal, temp_vec)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Create rotation matrix manually if scipy not available
            try:
                from scipy.spatial.transform import Rotation
                HAS_SCIPY = True
            except ImportError:
                HAS_SCIPY = False
                print("Warning: scipy not available, using manual rotation calculations")
            
            if HAS_SCIPY:
                rotation = Rotation.from_rotvec(np.radians(angle_degrees) * rotation_axis)
                rotated_normal = rotation.apply(reference_normal)
            else:
                # Manual rotation using Rodrigues' rotation formula
                angle_rad = np.radians(angle_degrees)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # Rodrigues' rotation formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
                k = rotation_axis
                v = reference_normal
                k_cross_v = np.cross(k, v)
                k_dot_v = np.dot(k, v)
                
                rotated_normal = (v * cos_angle + 
                                k_cross_v * sin_angle + 
                                k * k_dot_v * (1 - cos_angle))
            
            return rotated_normal / np.linalg.norm(rotated_normal)
        
        # Create anvil_plane1 (60 degrees from base_plane normal)
        anvil1_normal = create_rotated_normal(base_normal, 60.0)
        anvil_plane1 = np.array([anvil1_normal[0], anvil1_normal[1], anvil1_normal[2], anvil_plane1_d])
        
        # Create anvil_plane2 (60 degrees from anvil_plane1 normal)
        anvil2_normal = create_rotated_normal(anvil1_normal, 60.0)
        anvil_plane2 = np.array([anvil2_normal[0], anvil2_normal[1], anvil2_normal[2], anvil_plane2_d])
        
        # Debug: Print intermediate values
        # print("Debug Info:")
        # print(f"Base plane: {base_plane}")
        # print(f"Base normal: {base_normal}")
        # print(f"Orth1: {orth1}")
        # print(f"Orth2: {orth2}")
        # print(f"Anvil1 normal: {anvil1_normal}")
        # print(f"Anvil2 normal: {anvil2_normal}")
        
        # Verify constraints
        def verify_orthogonality(plane1, plane2):
            """Check if two planes are orthogonal (dot product of normals = 0)"""
            dot_product = np.dot(plane1[:3], plane2[:3])
            return abs(dot_product) < 1e-10
        
        def verify_parallelism(plane1, plane2):
            """Check if two planes are parallel (normals are identical)"""
            return np.allclose(plane1[:3], plane2[:3], atol=1e-10)
        
        def verify_angle(plane1, plane2, expected_angle_degrees):
            """Check if angle between plane normals matches expected value"""
            cos_angle = np.dot(plane1[:3], plane2[:3])
            angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            return abs(angle_degrees - expected_angle_degrees) < 1e-6
        
        def verify_separation(plane1, plane2, expected_distance):
            """Check separation between parallel planes"""
            distance = abs(plane2[3] - plane1[3])  # Since normals are unit vectors
            return abs(distance - expected_distance) < 1e-10
        
        # Perform verification
        if verify:
            print("Verification Results:")
            print(f"Outer planes orthogonal to base: {verify_orthogonality(base_plane, outer_plane1)}")
            print(f"Inner planes orthogonal to base: {verify_orthogonality(base_plane, inner_plane1)}")
            print(f"Outer planes parallel: {verify_parallelism(outer_plane1, outer_plane2)}")
            print(f"Inner planes parallel: {verify_parallelism(inner_plane1, inner_plane2)}")
            print(f"Anvil1 60° from base: {verify_angle(base_plane, anvil_plane1, 60.0)}")
            print(f"Anvil2 60° from anvil1: {verify_angle(anvil_plane1, anvil_plane2, 60.0)}")
            print(f"Inner planes separated by 25.0: {verify_separation(inner_plane1, inner_plane2, 25.0)}")
        
        created_planes = [base_plane,
                        outer_plane1,
                        outer_plane2,
                        inner_plane1,
                        inner_plane2,
                        anvil_plane1,
                        anvil_plane2]
        
        return created_planes
        
    
    
    def objective(ga_instance, solution, solution_idx):
        planes = create_planes_from_params(solution, verify=False)
        
        # n_planes = len(planes)
        loss = 0.0
        for i, plane in enumerate(planes):
            n = plane[:3]
            n /= np.linalg.norm(n)
            d = plane[3]
            pts = np.asarray(keep_inlier_clouds[i].points)
            res = pts @ n + d
            loss += np.sum(res**2)
        return 1 / loss
        
    # def on_generation(ga_instance):
    #     # This function is called after each generation.
    #     # You can print out relevant information here.
    #     print(f"\nGeneration = {ga_instance.generations_completed}")
    #     # print(f"Best solution = {ga_instance.best_solution()[0]}")
    #     print(f"Fitness of the best solution = {ga_instance.best_solution()[1]}")
    #     # print(f"Best solution index = {ga_instance.best_solution()[2]}")
    
    
    ### Perform optimization (option to show progress and results)
    n_genes = len(params)
    initial_population = []
    for _ in range(100):
        jittered = copy.deepcopy(params)
        jittered += np.random.normal(0, 0.1, size=len(jittered))
        initial_population.append(jittered)
    
    num_generations = 500
    with tqdm(total=num_generations) as pbar:
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=5,
            fitness_func=objective,
            sol_per_pop=len(initial_population),
            num_genes=n_genes,
            initial_population=initial_population,
            gene_space=gene_space,
            mutation_num_genes=3,
            mutation_type="random",
            mutation_by_replacement=True,
            crossover_type="single_point",
            parent_selection_type="tournament",
            on_generation=lambda _: pbar.update(1),
            # tqdm=True
        )
    
        print("[INFO] Running Genetic Algorithm...")
        ga_instance.run()
    
    # Plot the fitness progression
    ga_instance.plot_fitness(color='b')
    
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"\n[INFO] Optimization complete. Final fitness: {solution_fitness}")

    optimized_planes = create_planes_from_params(solution)

    optimized_plane_meshes = []
    for plane, cloud in zip(optimized_planes, keep_inlier_clouds):
        mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
        optimized_plane_meshes.append(mesh)

    return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds
    
    
    
    
    
    
    
    



def optimize_all_planes_genetic(keep_planes, keep_inlier_clouds):

    def normalize(v):
        return v / np.linalg.norm(v)

    def get_target_direction(normal, target_axis):
        normal_unit = normalize(normal)
        return target_axis if np.dot(normal_unit, target_axis) > np.dot(normal_unit, -target_axis) else -target_axis

    z_axis = np.array([0, 0, 1])
    x_axis = np.array([1, 0, 0])
    z_target = get_target_direction(keep_planes[0][:3], z_axis)
    x_targets = [get_target_direction(keep_planes[i][:3], x_axis) for i in range(1, 6)]

    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(keep_planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]

    def should_be_same_direction(normal1, normal2):
        return np.dot(normalize(normal1), normalize(normal2)) > 0

    outer_same_dir = should_be_same_direction(keep_planes[x_plane_indices[0]][:3],
                                               keep_planes[x_plane_indices[3]][:3])
    inner_same_dir = should_be_same_direction(keep_planes[x_plane_indices[1]][:3],
                                               keep_planes[x_plane_indices[2]][:3])

    flat_params = np.hstack(keep_planes)
    n_planes = len(flat_params) // 4
    n_genes = len(flat_params)

    def compute_anvil_separation(p1, p2):
        na1, da1 = normalize(p1[:3]), p1[3]
        na2, da2 = normalize(p2[:3]), p2[3]
        na_avg = normalize((na1 + na2) / 2)
        sep = np.abs((da1 - da2) / np.dot(na_avg, na1))
        return sep

    def objective_func(ga_instance, solution, solution_idx):
        planes = [solution[i * 4:(i + 1) * 4] for i in range(n_planes)]
        loss = 0.0

        for i, plane in enumerate(planes):
            n = normalize(plane[:3])
            d = plane[3]
            pts = np.asarray(keep_inlier_clouds[i].points)
            res = pts @ n + d
            loss += np.sum(res ** 2)

        penalty = 0.0

        # 1) Plane[0] normal = ±Z
        n0 = normalize(planes[0][:3])
        penalty += 1000 * (np.abs(n0[0]) + np.abs(n0[1]) + np.abs(n0[2] - z_target[2]))

        # 2) Planes[1:5] = ±X
        for i, idx in enumerate(x_plane_indices):
            ni = normalize(planes[idx][:3])
            penalty += 1000 * (np.abs(ni[1]) + np.abs(ni[2]) + np.abs(ni[0] - x_targets[idx - 1][0]))

        # 3) Outer pair parallel
        n_outer1 = normalize(planes[x_plane_indices[0]][:3])
        n_outer2 = normalize(planes[x_plane_indices[3]][:3])
        dot_outer = np.dot(n_outer1, n_outer2)
        penalty += 1000 * (np.abs(dot_outer - (1 if outer_same_dir else -1)))

        # 4) Inner pair parallel
        n_inner1 = normalize(planes[x_plane_indices[1]][:3])
        n_inner2 = normalize(planes[x_plane_indices[2]][:3])
        dot_inner = np.dot(n_inner1, n_inner2)
        penalty += 1000 * (np.abs(dot_inner - (1 if inner_same_dir else -1)))

        # 5) Planes[5] and [6] separated by 60 degrees
        n5, n6 = normalize(planes[5][:3]), normalize(planes[6][:3])
        penalty += 1000 * (np.abs(np.dot(n5, n6) - np.cos(np.radians(60))))

        # 6) Angled vs side planes = 150 deg (dot = -cos(30))
        cos150 = np.cos(np.radians(150))
        l_anvil = normalize(planes[x_plane_indices[1]][:3])
        r_anvil = normalize(planes[x_plane_indices[2]][:3])
        penalty += 1000 * (np.abs(np.dot(n5, l_anvil) - np.abs(cos150)))
        penalty += 1000 * (np.abs(np.dot(n6, r_anvil) - np.abs(cos150)))

        # 7) Unit length
        for p in planes:
            penalty += 1000 * (np.abs(np.linalg.norm(p[:3]) - 1))

        # 8) Anvil spacing = 25 mm
        sep = compute_anvil_separation(planes[x_plane_indices[1]], planes[x_plane_indices[2]])
        penalty += 1000 * (np.abs(sep - 25.0))

        return loss + penalty
    
    def on_generation(ga_instance):
        # This function is called after each generation.
        # You can print out relevant information here.
        print(f"\nGeneration = {ga_instance.generations_completed}")
        # print(f"Best solution = {ga_instance.best_solution()[0]}")
        print(f"Fitness of the best solution = {ga_instance.best_solution()[1]}")
        # print(f"Best solution index = {ga_instance.best_solution()[2]}")
    
    gene_space = [{'low': -1.0, 'high': 1.0} for _ in range(n_planes * 3)] + \
                 [{'low': -100.0, 'high': 100.0} for _ in range(n_planes)]

    initial_population = []
    for _ in range(20):
        jittered = copy.deepcopy(flat_params)
        jittered += np.random.normal(0, 0.05, size=len(jittered))
        initial_population.append(jittered)

    ga_instance = pygad.GA(
        num_generations=2000,
        num_parents_mating=5,
        fitness_func=objective_func,
        sol_per_pop=len(initial_population),
        num_genes=n_genes,
        initial_population=initial_population,
        gene_space=gene_space,
        mutation_percent_genes=10,
        mutation_type="random",
        mutation_by_replacement=True,
        crossover_type="single_point",
        parent_selection_type="rank",
        on_generation=on_generation
    )

    print("[INFO] Running Genetic Algorithm...")
    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"[INFO] Optimization complete. Final fitness: {solution_fitness}")

    optimized_planes = [solution[i*4:(i+1)*4] for i in range(n_planes)]

    optimized_plane_meshes = []
    for plane, cloud in zip(optimized_planes, keep_inlier_clouds):
        mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
        optimized_plane_meshes.append(mesh)

    return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds


def optimize_all_planes_pso(keep_planes, keep_inlier_clouds):
    """
    Optimize planes with geometric relationship constraints (no axis alignment):
    
    1. Planes[1:5] are four roughly x-aligned planes:
       - Two outer planes (far apart) - must be strictly parallel
       - Two inner planes (close together) - must be strictly parallel  
       - All four must be strictly orthogonal to Planes[0]
    
    2. Planes[0] should be roughly z-aligned (but not forced)
    
    3. Planes[5] and Planes[6] have 60° separation
    
    4. Planes[5] and Planes[6] each have 150° separation with inner planes
    
    5. Inner planes separated by 25mm
    
    The constrained parametrization ensures relationships are preserved.
    """
    
    # Identify plane relationships
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(keep_planes[1:5])
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]
    
    # Determine initial parallel relationships
    def should_be_same_direction(normal1, normal2):
        n1_unit = normalize(normal1)
        n2_unit = normalize(normal2)
        return np.dot(n1_unit, n2_unit) > 0
    
    outer_same_dir = should_be_same_direction(
        keep_planes[x_plane_indices[0]][:3], 
        keep_planes[x_plane_indices[3]][:3]
    )
    inner_same_dir = should_be_same_direction(
        keep_planes[x_plane_indices[1]][:3], 
        keep_planes[x_plane_indices[2]][:3]
    )
    
    print(f"[INFO] Outer planes same direction: {outer_same_dir}")
    print(f"[INFO] Inner planes same direction: {inner_same_dir}")
    
    def params_to_planes(params):
        """
        Convert constrained parameter space to full plane definitions.
        
        Parameter space:
        - params[0:3]: normal vector for Planes[0] (base plane)
        - params[3]: d parameter for Planes[0]
        - params[4:6]: direction vector for outer plane pair (will be normalized)
        - params[6:10]: d parameters for the four x-aligned planes
        - params[10:12]: direction vector for inner plane pair (will be normalized) 
        - params[12]: d parameter for Planes[5]
        - params[13]: d parameter for Planes[6]
        - params[14:16]: direction in plane orthogonal to inner planes for Planes[5] constraint
        - params[16:18]: direction in plane orthogonal to inner planes for Planes[6] constraint
        """
        
        planes = []
        
        # Plane 0: Base plane (roughly z-aligned but not forced)
        n0 = normalize(params[0:3])
        planes.append(np.array([n0[0], n0[1], n0[2], params[3]]))
        
        # Construct orthogonal basis to n0
        if np.abs(n0[2]) < 0.9:
            basis1 = normalize(np.cross(n0, [0, 0, 1]))
        else:
            basis1 = normalize(np.cross(n0, [1, 0, 0]))
        basis2 = normalize(np.cross(n0, basis1))
        
        # Outer plane direction (in the n0-orthogonal plane)
        outer_dir_2d = normalize(params[4:6])
        outer_dir = normalize(outer_dir_2d[0] * basis1 + outer_dir_2d[1] * basis2)
        
        # Create the two outer planes
        outer1_normal = outer_dir
        outer2_normal = outer_dir if outer_same_dir else -outer_dir
        
        inner_dir_2d = normalize(params[10:12])
        inner_dir = normalize(inner_dir_2d[0] * basis1 + inner_dir_2d[1] * basis2)

        
        # Create the two inner planes
        inner1_normal = inner_dir
        inner2_normal = inner_dir if inner_same_dir else -inner_dir
        
        # Insert the four x-aligned planes in the correct order
        temp_planes = [None] * 4
        temp_planes[0] = np.array([outer1_normal[0], outer1_normal[1], outer1_normal[2], params[6]])
        temp_planes[1] = np.array([inner1_normal[0], inner1_normal[1], inner1_normal[2], params[7]])
        temp_planes[2] = np.array([inner2_normal[0], inner2_normal[1], inner2_normal[2], params[8]])
        temp_planes[3] = np.array([outer2_normal[0], outer2_normal[1], outer2_normal[2], params[9]])
        
        # Add them in the correct global order
        for i, local_idx in enumerate([idx_outer1, idx_inner1, idx_inner2, idx_outer2]):
            planes.append(temp_planes[i])
        
        # Planes 5 and 6: Constrained by angles with inner planes
        # Use the constraint that they have 150° with inner planes and 60° with each other
        
        # Get the common direction of inner planes
        avg_inner_normal = normalize(inner1_normal + inner2_normal) if inner_same_dir else inner1_normal
        
        # Create orthogonal basis in the plane perpendicular to inner planes
        if np.abs(avg_inner_normal[2]) < 0.9:
            perp1 = normalize(np.cross(avg_inner_normal, [0, 0, 1]))
        else:
            perp1 = normalize(np.cross(avg_inner_normal, [1, 0, 0]))
        perp2 = normalize(np.cross(avg_inner_normal, perp1))
        
        # Direction for Plane 5 in the perpendicular space
        dir5_perp = normalize(params[14:16])
        dir5_in_perp_space = dir5_perp[0] * perp1 + dir5_perp[1] * perp2
        
        # Plane 5 normal satisfying 150° constraint with inner planes
        cos_150 = np.cos(np.radians(150))
        sin_150 = np.sin(np.radians(150))
        
        n5 = cos_150 * avg_inner_normal + sin_150 * normalize(dir5_in_perp_space)
        n5 = normalize(n5)
        
        # Plane 6 normal: 60° from Plane 5 and 150° from inner planes
        cos_60 = np.cos(np.radians(60))
        
        # Find n6 such that n5 · n6 = cos(60°) and n6 · avg_inner = cos(150°)
        # This constrains n6 to lie on a circle. Use the remaining parameter.
        dir6_perp = normalize(params[16:18])
        dir6_in_perp_space = dir6_perp[0] * perp1 + dir6_perp[1] * perp2
        
        # Similar construction for n6
        n6 = cos_150 * avg_inner_normal + sin_150 * normalize(dir6_in_perp_space)
        n6 = normalize(n6)
        
        # Adjust n6 to satisfy 60° constraint with n5
        # Project n6 onto the cone around n5 with 60° half-angle
        n5_cross_n6 = np.cross(n5, n6)
        if np.linalg.norm(n5_cross_n6) > 1e-6:
            axis = normalize(n5_cross_n6)
            current_angle = np.arccos(np.clip(np.dot(n5, n6), -1, 1))
            angle_diff = np.radians(60) - current_angle
            
            # Rodrigues rotation to adjust angle
            cos_diff = np.cos(angle_diff)
            sin_diff = np.sin(angle_diff)
            n6 = (n6 * cos_diff + 
                  np.cross(axis, n6) * sin_diff + 
                  axis * np.dot(axis, n6) * (1 - cos_diff))
            n6 = normalize(n6)
        
        planes.append(np.array([n5[0], n5[1], n5[2], params[12]]))
        planes.append(np.array([n6[0], n6[1], n6[2], params[13]]))
        
        return planes
    
    def check_constraints(planes):
        """Check how well constraints are satisfied"""
        errors = []
        
        # Check orthogonality of planes 1-4 with plane 0
        n0 = normalize(planes[0][:3])
        for i in range(1, 5):
            ni = normalize(planes[i][:3])
            dot_product = np.dot(n0, ni)
            errors.append(dot_product**2)  # Should be 0
        
        # Check parallelism of outer planes
        n_outer1 = normalize(planes[x_plane_indices[0]][:3])
        n_outer2 = normalize(planes[x_plane_indices[3]][:3])
        if outer_same_dir:
            errors.append((np.dot(n_outer1, n_outer2) - 1)**2)
        else:
            errors.append((np.dot(n_outer1, n_outer2) + 1)**2)
        
        # Check parallelism of inner planes
        n_inner1 = normalize(planes[x_plane_indices[1]][:3])
        n_inner2 = normalize(planes[x_plane_indices[2]][:3])
        if inner_same_dir:
            errors.append((np.dot(n_inner1, n_inner2) - 1)**2)
        else:
            errors.append((np.dot(n_inner1, n_inner2) + 1)**2)
        
        # Check 60° constraint between planes 5 and 6
        n5 = normalize(planes[5][:3])
        n6 = normalize(planes[6][:3])
        cos_60 = np.cos(np.radians(60))
        errors.append((np.dot(n5, n6) - cos_60)**2)
        
        # Check 150° constraints
        cos_150 = np.cos(np.radians(150))
        avg_inner = normalize(n_inner1 + n_inner2) if inner_same_dir else n_inner1
        errors.append((np.dot(n5, avg_inner) - cos_150)**2)
        errors.append((np.dot(n6, avg_inner) - cos_150)**2)
        
        # Check 25mm separation of inner planes
        inner_plane1 = planes[x_plane_indices[1]]
        inner_plane2 = planes[x_plane_indices[2]]
        
        na1, da1 = normalize(inner_plane1[:3]), inner_plane1[3]
        na2, da2 = normalize(inner_plane2[:3]), inner_plane2[3]
        
        # Calculate distance between parallel planes
        if inner_same_dir:
            separation = abs(da1 - da2)
        else:
            separation = abs(da1 + da2)
        
        errors.append((separation - 25.0)**2)
        
        return errors
    
    def objective_function(params_matrix):
        """Objective function combining fit error and constraint violations"""
        if len(params_matrix.shape) == 1:
            params_matrix = params_matrix.reshape(1, -1)
            
        n_particles = params_matrix.shape[0]
        fitness = np.zeros(n_particles)
        
        penalty_weight = 1e6
        
        for i in range(n_particles):
            params = params_matrix[i]
            
            try:
                planes = params_to_planes(params)
                
                # Calculate fit error
                fit_error = 0.0
                for j, plane in enumerate(planes):
                    n = normalize(plane[:3])
                    d = plane[3]
                    pts = np.asarray(keep_inlier_clouds[j].points)
                    residuals = pts @ n + d
                    fit_error += np.sum(residuals**2)
                
                # Calculate constraint violations
                constraint_errors = check_constraints(planes)
                total_violation = np.sum(constraint_errors)
                
                fitness[i] = fit_error + penalty_weight * total_violation
                
            except Exception as e:
                fitness[i] = 1e10
        
        return fitness
    
    # Extract initial parameters from current planes
    initial_params = []
    
    # Plane 0 normal and d
    initial_params.extend(keep_planes[0][:3])  # normal
    initial_params.append(keep_planes[0][3])   # d
    
    # Outer plane direction (from first outer plane)
    outer_normal = normalize(keep_planes[x_plane_indices[0]][:3])
    initial_params.extend(outer_normal[:2])  # Only need 2 components, normalize anyway
    
    # D parameters for x-aligned planes
    for idx in x_plane_indices:
        initial_params.append(keep_planes[idx][3])
    
    # Inner plane direction (from first inner plane)
    inner_normal = normalize(keep_planes[x_plane_indices[1]][:3])
    initial_params.extend(inner_normal[:2])
    
    # D parameters for planes 5 and 6
    initial_params.append(keep_planes[5][3])
    initial_params.append(keep_planes[6][3])
    
    # Direction parameters for planes 5 and 6 (initial guess)
    initial_params.extend([1.0, 0.0])  # direction for plane 5
    initial_params.extend([0.0, 1.0])  # direction for plane 6
    
    # Set bounds
    bounds_low = []
    bounds_high = []
    
    for i, param in enumerate(initial_params):
        if i < 3:  # Normal vector components
            bounds_low.append(-2.0)
            bounds_high.append(2.0)
        elif i == 3 or (6 <= i <= 9) or i == 12 or i == 13:  # D parameters
            bounds_low.append(param - 50.0)
            bounds_high.append(param + 50.0)
        else:  # Direction parameters
            bounds_low.append(-2.0)
            bounds_high.append(2.0)
    
    bounds = list(zip(bounds_low, bounds_high))
    
    print(f"[INFO] Using geometric constraint parametrization")
    print(f"[INFO] Parameter space dimension: {len(initial_params)}")
    print(f"[INFO] X-plane indices: {x_plane_indices}")
    
    # Use Differential Evolution
    def de_objective(params):
        return objective_function(params)[0]
    
    result = differential_evolution(
        de_objective,
        bounds=bounds,
        seed=42,
        maxiter=1000,
        popsize=15,
        atol=1e-10,
        tol=1e-10
    )
    
    print(f"[INFO] DE optimization completed. Success: {result.success}")
    print(f"[INFO] Final cost: {result.fun}")
    
    if not result.success:
        print(f"[WARNING] Optimization failed: {result.message}")
    
    # Convert final parameters back to planes
    optimized_planes = params_to_planes(result.x)
    
    # Verify constraints
    print("\n[INFO] Constraint verification:")
    constraint_errors = check_constraints(optimized_planes)
    
    print(f"Orthogonality errors (planes 1-4 vs plane 0): {[f'{e:.6f}' for e in constraint_errors[:4]]}")
    print(f"Outer plane parallelism error: {constraint_errors[4]:.6f}")
    print(f"Inner plane parallelism error: {constraint_errors[5]:.6f}")
    print(f"60° constraint error: {constraint_errors[6]:.6f}")
    print(f"150° constraint errors: {constraint_errors[7]:.6f}, {constraint_errors[8]:.6f}")
    print(f"25mm separation error: {constraint_errors[9]:.6f}")
    
    # Check actual angles
    n5 = normalize(optimized_planes[5][:3])
    n6 = normalize(optimized_planes[6][:3])
    angle_56 = np.degrees(np.arccos(np.clip(np.dot(n5, n6), -1, 1)))
    print(f"Actual angle between planes 5&6: {angle_56:.2f}°")
    
    # Create placeholder meshes
    optimized_plane_meshes = [None] * len(optimized_planes)
    
    return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds


def separation_between_parallel_planes(plane1, plane2):
    """
    Computes the perpendicular distance between two parallel planes.
    Planes must be in the form [a, b, c, d] for ax + by + cz + d = 0.
    """
    n1 = plane1[:3]
    d1 = plane1[3]
    d2 = plane2[3]

    # Normalize the normal vector
    n1_norm = n1 / np.linalg.norm(n1)

    # Compute signed distance from origin to each plane
    # Signed distance = -d / ||n||
    dist1 = -d1 / np.linalg.norm(n1)
    dist2 = -d2 / np.linalg.norm(n1)

    # Return the absolute separation between the planes
    return abs(dist1 - dist2)

# def create_cylinder_relative_to_planes(
#     plane1, plane2,
#     plane1_offset,
#     plane2_offset,
#     diameter, height
# ):
#     """
#     Create a cylinder aligned with the intersection of two orthogonal planes.

#     Parameters:
#     - plane1, plane2: 4-element numpy arrays [a, b, c, d] for ax + by + cz + d = 0
#     - plane1_offset, plane2_offset: Scalar distances (can be positive or negative)
#     - diameter: Cylinder diameter
#     - height: Cylinder height

#     Returns:
#     - trimesh.Trimesh cylinder mesh
#     """

#     def flip_if_major_negative(plane):
#         normal = plane[:3]
#         major_idx = np.argmax(np.abs(normal))
#         if normal[major_idx] < 0:
#             return np.concatenate([-normal, [-plane[3]]])
#         return plane

#     # Ensure normals face positive along their major axis
#     plane1 = flip_if_major_negative(plane1)
#     plane2 = flip_if_major_negative(plane2)

#     # Extract normals
#     n1 = plane1[:3]
#     n2 = plane2[:3]

#     # Axis of the cylinder = intersection line of planes
#     axis_dir = np.cross(n1, n2)
#     axis_dir /= np.linalg.norm(axis_dir)

#     # Find a point on both planes (intersection point)
#     A = np.vstack([n1, n2, axis_dir])
#     b = -np.array([plane1[3], plane2[3], 0])
#     intersection_point = np.linalg.lstsq(A, b, rcond=None)[0]

#     # Offset from each plane along its normal
#     offset_vector = n1 * plane1_offset + n2 * plane2_offset
#     base_center = intersection_point + offset_vector - axis_dir * height / 2

#     # Create default cylinder aligned with +Z
#     cyl = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)

#     # Rotate to align Z axis with desired axis_dir
#     z_axis = np.array([0, 0, 1])
#     if np.allclose(axis_dir, z_axis):
#         R = np.eye(3)
#     elif np.allclose(axis_dir, -z_axis):
#         R = -np.eye(3)
#     else:
#         rotation_axis = np.cross(z_axis, axis_dir)
#         rotation_axis /= np.linalg.norm(rotation_axis)
#         angle = np.arccos(np.clip(np.dot(z_axis, axis_dir), -1.0, 1.0))
#         R = trimesh.transformations.rotation_matrix(angle, rotation_axis)[:3, :3]

#     # Compose transform
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = base_center
#     cyl.apply_transform(T)

#     # Translate further along the cylinder's axis
#     T2 = np.eye(4)
#     T2[:3, 3] = axis_dir * (height / 2)
#     cyl.apply_transform(T2)

#     return cyl

def create_cylinder_relative_to_planes(
    plane1, plane2,
    plane1_offset,
    plane2_offset,
    diameter, height
):
    """
    Create a cylinder aligned with the intersection of two orthogonal planes.
    The cylinder's geometric center will be positioned such that:
    - It's offset from plane1 by plane1_offset in world coordinates
    - It's offset from plane2 by plane2_offset in world coordinates  
    - Its center is at Y=0 (or follows the intersection line for non-axis-aligned planes)
    
    Parameters:
    - plane1, plane2: 4-element numpy arrays [a, b, c, d] for ax + by + cz + d = 0
    - plane1_offset, plane2_offset: Scalar distances in world coordinates
      (for Z-aligned plane1: positive = more positive Z)
      (for X-aligned plane2: positive = more positive X)
    - diameter: Cylinder diameter
    - height: Cylinder height
    Returns:
    - trimesh.Trimesh cylinder mesh
    """
    
    # Extract and normalize normals (preserve original orientation)
    n1 = plane1[:3] / np.linalg.norm(plane1[:3])
    n2 = plane2[:3] / np.linalg.norm(plane2[:3])
    
    # Cylinder axis = intersection line of the two planes
    axis_dir = np.cross(n1, n2)
    axis_dir /= np.linalg.norm(axis_dir)
    
    # Find a point on the intersection line of both planes
    # We'll find the point where the intersection line crosses Y=0
    A = np.vstack([n1, n2])
    b = -np.array([plane1[3], plane2[3]])
    
    # If planes are axis-aligned, we can solve this directly
    if np.allclose(np.abs(n1), [0, 0, 1]) and np.allclose(np.abs(n2), [1, 0, 0]):
        # Z-aligned and X-aligned planes
        # Intersection line is parallel to Y-axis
        # Find intersection point at Y=0
        z_from_plane1 = -plane1[3] / n1[2]  # z where plane1 intersects at x=0, y=0
        x_from_plane2 = -plane2[3] / n2[0]  # x where plane2 intersects at y=0, z=0
        intersection_at_y0 = np.array([x_from_plane2, 0, z_from_plane1])
    else:
        # General case: find intersection line and get point at Y=0
        # Use least squares to find a point on both planes
        A_extended = np.vstack([n1, n2, [0, 1, 0]])  # Add constraint y=0
        b_extended = np.append(b, 0)
        intersection_at_y0 = np.linalg.lstsq(A_extended, b_extended, rcond=None)[0]
    
    # Apply offsets in world coordinates
    # For axis-aligned planes, determine offset direction based on desired world coordinate direction
    
    # Handle plane1 offset
    if np.allclose(np.abs(n1), [0, 0, 1]):  # plane1 is Z-aligned
        # We want positive offset to mean more positive Z
        # If normal points in +Z, offset in +Z direction
        # If normal points in -Z, offset in -Z direction (away from plane)
        plane1_offset_vector = np.array([0, 0, plane1_offset * np.sign(n1[2])])
    elif np.allclose(np.abs(n1), [1, 0, 0]):  # plane1 is X-aligned
        # We want positive offset to mean more positive X
        plane1_offset_vector = np.array([plane1_offset * np.sign(n1[0]), 0, 0])
    elif np.allclose(np.abs(n1), [0, 1, 0]):  # plane1 is Y-aligned
        # We want positive offset to mean more positive Y
        plane1_offset_vector = np.array([0, plane1_offset * np.sign(n1[1]), 0])
    else:
        # For non-axis-aligned planes, offset along the normal
        # Positive offset moves in direction of normal
        plane1_offset_vector = n1 * plane1_offset
    
    # Handle plane2 offset
    if np.allclose(np.abs(n2), [0, 0, 1]):  # plane2 is Z-aligned
        # We want positive offset to mean more positive Z
        plane2_offset_vector = np.array([0, 0, plane2_offset * np.sign(n2[2])])
    elif np.allclose(np.abs(n2), [1, 0, 0]):  # plane2 is X-aligned
        # We want positive offset to increase X coordinate
        # Always offset in +X direction regardless of normal direction
        plane2_offset_vector = np.array([plane2_offset, 0, 0])
    elif np.allclose(np.abs(n2), [0, 1, 0]):  # plane2 is Y-aligned
        # We want positive offset to mean more positive Y
        plane2_offset_vector = np.array([0, plane2_offset * np.sign(n2[1]), 0])
    else:
        # For nearly-axis-aligned planes, determine the dominant axis
        abs_n2 = np.abs(n2)
        dominant_axis = np.argmax(abs_n2)
        
        if dominant_axis == 0:  # Nearly X-aligned
            # We want positive offset to increase X coordinate
            plane2_offset_vector = np.array([plane2_offset, 0, 0])
        elif dominant_axis == 1:  # Nearly Y-aligned
            # We want positive offset to increase Y coordinate
            plane2_offset_vector = np.array([0, plane2_offset, 0])
        else:  # Nearly Z-aligned
            # We want positive offset to increase Z coordinate
            plane2_offset_vector = np.array([0, 0, plane2_offset])
        
        # For safety, also handle the truly non-axis-aligned case
        # (though you said this shouldn't happen in your use case)
        if np.max(abs_n2) < 0.9:  # Not close to any axis
            plane2_offset_vector = n2 * plane2_offset
    
    # Calculate where we want the final cylinder center to be
    desired_final_center = intersection_at_y0 + plane1_offset_vector + plane2_offset_vector
    
    # Since we'll translate by height/2 along axis_dir, we need to pre-compensate
    # so that after the translation, the center ends up at desired_final_center
    initial_cylinder_center = desired_final_center - axis_dir * (height / 2)
    
    # Create default cylinder aligned with +Z
    cyl = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)
    
    # Rotate to align Z axis with cylinder axis direction
    z_axis = np.array([0, 0, 1])
    if np.allclose(axis_dir, z_axis):
        R = np.eye(3)
    elif np.allclose(axis_dir, -z_axis):
        R = -np.eye(3)
    else:
        rotation_axis = np.cross(z_axis, axis_dir)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, axis_dir), -1.0, 1.0))
        R = trimesh.transformations.rotation_matrix(angle, rotation_axis)[:3, :3]
    
    # First transform: rotation + translation to initial position
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = initial_cylinder_center
    cyl.apply_transform(T)
    
    # Second transform: translate along the cylinder's axis by height/2
    # This will move the center to the desired final position
    T2 = np.eye(4)
    T2[:3, 3] = axis_dir * (height / 2)
    cyl.apply_transform(T2)
    
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


def align_planes_to_axes_minimal_v2(aligned_pcd, optimized_planes, planeX, planeZ):
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
    
    print(f"Original normalX: {normalX}")
    print(f"Original normalZ: {normalZ}")
    
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
    
    print("\nEvaluating alignment options:")
    for target_X, target_Z, label in options:
        angle_X = rotation_angle_between_vectors(normalX, target_X)
        angle_Z = rotation_angle_between_vectors(normalZ, target_Z)
        total_angle = angle_X + angle_Z
        
        print(f"{label}: X angle = {np.degrees(angle_X):.3f}°, Z angle = {np.degrees(angle_Z):.3f}°, Total = {np.degrees(total_angle):.3f}°")
        
        if total_angle < min_total_angle:
            min_total_angle = total_angle
            best_option = (target_X, target_Z, label, angle_X, angle_Z)
    
    final_X, final_Z, best_label, best_angle_X, best_angle_Z = best_option
    print(f"\nSelected option: {best_label}")
    print(f"Target X direction: {final_X}")
    print(f"Target Z direction: {final_Z}")
    
    # Check orthogonality of target directions
    orthogonality_check = abs(np.dot(final_X, final_Z))
    if orthogonality_check > 1e-10:
        print(f"Warning: Target directions not orthogonal! Dot product: {orthogonality_check}")
    
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
        print("Procrustes method failed, using sequential rotations")
        R = compute_sequential_minimal_rotation(normalX, normalZ, final_X, final_Z)
    
    # Verify the rotation
    final_normalX = R @ normalX
    final_normalZ = R @ normalZ
    
    print("\nAfter rotation:")
    print(f"Final normalX: {final_normalX}")
    print(f"Final normalZ: {final_normalZ}")
    
    # Check alignment quality
    alignment_X = abs(np.dot(final_normalX, final_X))
    alignment_Z = abs(np.dot(final_normalZ, final_Z))
    
    angle_error_X = np.degrees(np.arccos(np.clip(alignment_X, 0, 1)))
    angle_error_Z = np.degrees(np.arccos(np.clip(alignment_Z, 0, 1)))
    
    print(f"X alignment: {alignment_X:.6f} (error: {angle_error_X:.6f}°)")
    print(f"Z alignment: {alignment_Z:.6f} (error: {angle_error_Z:.6f}°)")
    
    # Verify rotation matrix properties
    det_R = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    
    if abs(det_R - 1.0) > 1e-10:
        print(f"Warning: Rotation matrix determinant = {det_R}, should be 1.0")
    if orthogonality_error > 1e-10:
        print(f"Warning: Rotation matrix orthogonality error = {orthogonality_error}")
    
    # Create transformation matrix and apply
    T = np.eye(4)
    T[:3, :3] = R
    rotated_pcd = aligned_pcd.transform(T)
    
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
    
    print("\nRotation summary:")
    print(f"Euler angles (XYZ): {euler_angles}")
    print(f"Total rotation angle: {rotation_angle:.3f}°")
    
    return rotated_pcd, rotated_planes, R, rotation_info


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


# Debug function to help understand what's happening
def debug_plane_orientations(planes, labels=None):
    """
    Debug function to print plane normal orientations
    """
    print("\n=== PLANE ORIENTATIONS DEBUG ===")
    for i, plane in enumerate(planes):
        normal = plane[:3] / np.linalg.norm(plane[:3])
        label = labels[i] if labels else f"Plane {i}"
        
        # Compute angles with coordinate axes
        angle_x = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [1,0,0])), 0, 1)))
        angle_y = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [0,1,0])), 0, 1)))
        angle_z = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [0,0,1])), 0, 1)))
        
        print(f"{label}: normal = {normal}")
        print(f"  Angles to axes: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°")
        
        # Show which axis is closest
        min_angle = min(angle_x, angle_y, angle_z)
        if min_angle == angle_x:
            closest = "X-axis"
        elif min_angle == angle_y:
            closest = "Y-axis"
        else:
            closest = "Z-axis"
        print(f"  Closest to: {closest} ({min_angle:.1f}°)")
    print("=" * 40)


def validate_minimal_rotation(original_planes, rotated_planes, planeX_idx, planeZ_idx, rotation_matrix):
    """
    Validate that the minimal rotation alignment worked correctly.
    """
    print("\n=== MINIMAL ROTATION VALIDATION ===")
    
    # Check the target planes
    planeX_original = original_planes[planeX_idx]
    planeZ_original = original_planes[planeZ_idx]
    planeX_rotated = rotated_planes[planeX_idx]
    planeZ_rotated = rotated_planes[planeZ_idx]
    
    # Original normals
    normalX_orig = planeX_original[:3] / np.linalg.norm(planeX_original[:3])
    normalZ_orig = planeZ_original[:3] / np.linalg.norm(planeZ_original[:3])
    
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
    
    print(f"PlaneX alignment with X-axis: {best_X_alignment:.6f} (error: {angle_error_X:.6f}°)")
    print(f"PlaneZ alignment with Z-axis: {best_Z_alignment:.6f} (error: {angle_error_Z:.6f}°)")
    
    # Check total rotation angle
    rotation_scipy = Rotation.from_matrix(rotation_matrix)
    total_angle = np.linalg.norm(rotation_scipy.as_rotvec()) * 180 / np.pi
    print(f"Total rotation angle: {total_angle:.3f}°")
    
    success = angle_error_X < 0.1 and angle_error_Z < 0.1
    print(f"Alignment successful: {success}\n\n")
    
    return success, angle_error_X, angle_error_Z, total_angle

def generate_candidate_orientations(mesh, visualize=False, target_pcd=None):
    print("[INFO] Generating candidate 180° flips...")

    # Define all required 180° rotations
    rotations = {
        "Original": np.eye(3),
        "Flip Z": Rotation.from_euler('z', 180, degrees=True).as_matrix(),
        "Flip X": Rotation.from_euler('x', 180, degrees=True).as_matrix(),
        "Flip Z+X": Rotation.from_euler('x', 180, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix()
    }

    candidates = []
    for name, rot in rotations.items():
        m = copy.deepcopy(mesh)
        m.rotate(rot, center=(0, 0, 0))
        print(f"  - Candidate: {name}")
        
        if visualize and target_pcd is not None:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
            m.paint_uniform_color([1, 0, 0])
            target_pcd.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries(
                [m, target_pcd, axis],
                window_name=f"Candidate Orientation: {name}"
            )
        
        candidates.append((m, name))

    return candidates

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

def ensure_normals_outward(mesh: trimesh.Trimesh, mesh_name=""):
    mesh_type = identify_mesh_type(mesh)
    if mesh_type=='open3d':
        mesh = o3d_to_trimesh(mesh)
    
    if not mesh.is_watertight:
        print(f"Warning: Mesh '{mesh_name}' is not watertight. Normal orientation may be unreliable.")
    
    # Attempt to fix normals
    mesh.fix_normals()

    # Optional check
    if not mesh.is_winding_consistent:
        print(f"Warning: Mesh '{mesh_name}' has inconsistent winding after fixing normals.")

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

def create_model(fixture_scan_path, specimen_scan_path, output_path):
    # mesh_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
    # print("Loading mesh...")
    # base_pcd = load_mesh_as_point_cloud(fixture_scan_path)
    
    # target_axes = [[0, 0, 1],
    #                [1, 0, 0],
    #                [np.sqrt(3)/2, 0, -0.5],
    #                [-np.sqrt(3)/2, 0, -0.5]]
    
    expected_planes = {
        0: (np.array([0, 0, 1]), 1),
        1: (np.array([1, 0, 0]), 4),
        2: (np.array([np.sqrt(3)/2, 0, -0.5]), 1),
        3: (np.array([-np.sqrt(3)/2, 0, -0.5]), 1)
    }
    
    # supports_distance = 127.66
    
    # keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid = detect_fixture_planes(base_pcd, target_axes)
    keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid = fixture_plane_fitting.create_model(fixture_scan_path, expected_planes, visualization=True)
    
    # optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes_pso(keep_planes, keep_inlier_clouds)
    # optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes_genetic(keep_planes, keep_inlier_clouds)
    optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes_genetic_v2(keep_planes, keep_inlier_clouds)
    # optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes(keep_planes, keep_inlier_clouds)
    
    # Verification of constraints
    print("")
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
            print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} ({labels[1]}) are separated by {angle_diff:.4f} degrees')
        elif len(common_elements) == 1:
            if combo[0] in common_elements:
                print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} are separated by {angle_diff:.4f} degrees')
            else:
                print(f'Planes {combo[0]} and {combo[1]} ({labels[0]}) are separated by {angle_diff:.4f} degrees')                
        else:
            print(f'Planes {combo[0]} and {combo[1]} are separated by {angle_diff:.4f} degrees')
        
        if np.abs(np.around(angle_diff,2)) <= 0.01:
            separation_dist = separation_between_parallel_planes(optimized_planes[combo[0]], optimized_planes[combo[1]])
            print(f'\tPlanes {combo[0]} and {combo[1]} are separated by a distance of {separation_dist}')
    
    
    # Re-align everything so the support planes and base plane define the world orientation
    planeX_idx = x_plane_indices[0]
    planeZ_idx = 0
    planeX = optimized_planes[planeX_idx]
    planeZ = optimized_planes[planeZ_idx]
    
    # Perform alignment
    rotated_pcd, rotated_planes, R_planes, info = align_planes_to_axes_minimal_v2(
        aligned_pcd, optimized_planes, planeX, planeZ
    )
    
    # Validate results
    success, error_X, error_Z, total_angle = validate_minimal_rotation(
        optimized_planes, rotated_planes, planeX_idx, planeZ_idx, R_planes
    )    
    
    print(f'\nPlane adjustment R matrix:\n{R_planes}')
    
    if R_flip is not None:
        R_total = R_planes @ R_90X @ R_flip @ R_pca
    else:
        R_total = R_planes @ R_90X @ R_pca
    
    print(f'\nR Total:\n{R_total}')
    
    # === STEP 1: Load the original mesh ===
    original_mesh = load_mesh(fixture_scan_path)
    original_vertices = np.asarray(original_mesh.vertices)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
    
    # === STEP 2: Translate to origin ===
    original_mesh.translate(-centroid)
    
    # === STEP 3: PCA alignment ===
    original_mesh.rotate(R_pca, center=(0, 0, 0))
    
    if R_flip is not None:
        original_mesh.rotate(R_flip, center=(0, 0, 0))
    
    # === STEP 4: Rotate 90° about +X ===
    original_mesh.rotate(R_90X, center=(0, 0, 0))
    
    # === STEP 5: Apply final minimal rotation matrix R ===
    original_mesh.rotate(R_planes, center=(0, 0, 0))
    
    # === RESULT ===
    transformed_reference_mesh = original_mesh
    
    o3d.visualization.draw_geometries([
        aligned_pcd.paint_uniform_color([0.5, 0.5, 0.5]),
        transformed_reference_mesh.paint_uniform_color([1, 0, 0]),
        axis], window_name="Transformed Reference Mesh")

    # Load and align matched specimen
    # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    matched_specimen_mesh = load_mesh(specimen_scan_path)
    
    aligned_specimen_mesh = align_tgt_to_ref_meshes(transformed_reference_mesh, matched_specimen_mesh)
    
    #### Create mesh models of support and anvil cylinders
    diameter = 10
    height = 40
    
    base_plane = rotated_planes[0]
    base_support_offset = 52
    
    support_offset = 25.4   # 1 inch
    
    anvil_plane1 = rotated_planes[5]
    anvil_plane2 = rotated_planes[6]
    anvil = create_cylinder_relative_to_planes(anvil_plane1,
                                            anvil_plane2,
                                            0,
                                            0,
                                            diameter,
                                            height)
    anvil_mesh = trimesh_to_open3d(anvil)
    
    l_support_plane = rotated_planes[x_plane_indices[0]]
    l_support_plane_offset = support_offset
    l_support = create_cylinder_relative_to_planes(
                    base_plane, l_support_plane,
                    base_support_offset,
                    l_support_plane_offset,
                    diameter, height)
    
    l_support_mesh = trimesh_to_open3d(l_support)
    
    r_support_plane = rotated_planes[x_plane_indices[3]]
    r_support_plane_offset = -support_offset
    r_support = create_cylinder_relative_to_planes(
                    base_plane, r_support_plane,
                    base_support_offset,
                    r_support_plane_offset,
                    diameter, height)
    
    r_support_mesh = trimesh_to_open3d(r_support)
    
    # Create coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])

    # Visualize optimized planes and the inlier clouds used for fitting
    cylinder_meshes = [anvil_mesh, l_support_mesh, r_support_mesh]
    o3d.visualization.draw_geometries([rotated_pcd] + [axis] + cylinder_meshes + [aligned_specimen_mesh], window_name="Aligned Cylinders")
    
    anvil = ensure_normals_outward(anvil, mesh_name="anvil")
    l_support = ensure_normals_outward(l_support, mesh_name="left_support")
    r_support = ensure_normals_outward(r_support, mesh_name="right_support")
    flex_mesh = ensure_normals_outward(aligned_specimen_mesh, mesh_name="flex_mesh")
    
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
                print(f"Applying {increment} adjustment to Z position of anvil")
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
                print(f"Applying -{increment} adjustment to Z position of supports")
                # Recheck
                updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
                if l_idx in updated or r_idx in updated:
                    still_intersecting = True
    
            if not still_intersecting:
                break
        else:
            print("Warning: Maximum adjustment iterations reached, intersection may still exist.")
    
    # Adjust for good measure
    n_increments = 2
    anvil.apply_translation([0, 0, n_increments*increment])
    l_support.apply_translation([0, 0, -n_increments*increment])
    r_support.apply_translation([0, 0, -n_increments*increment])
    
    merged_mesh = trimesh.util.concatenate(all_meshes)
    
    # output_filepath = "E:/Fixture Scans/prepared_test.stl"
    merged_mesh.export(output_path)

def create_model_v2(fixture_scan_path, specimen_scan_path, output_path):
    # mesh_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(fixture_scan_path)
    
    # Orient base_pcd with PCA and rotate so it is in close to the right orientation (Z up, X right)
    base_pcd, R_pca, R_90X, R_flip, centroid = prepare_scan_orientation(base_pcd)
    
    target_axes = [[0, 0, 1],
                   [1, 0, 0],
                   [np.sqrt(3)/2, 0, -0.5],
                   [-np.sqrt(3)/2, 0, -0.5]]
    
    all_keep_planes = detect_fixture_planes_efficient(base_pcd, target_axes)
    
    # supports_distance = 127.66
    
    # keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid = detect_fixture_planes(base_pcd, target_axes)
    
    # optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes(keep_planes, keep_inlier_clouds)
    
    # # Verification of constraints
    # print("")
    # for combo in combinations(range(len(optimized_planes)), 2):
    #     n1 = optimized_planes[combo[0]][:3]
    #     n2 = optimized_planes[combo[1]][:3]
    #     angle_diff = angular_change_between_normals(n1, n2)
        
    #     # Check if the current combo contains X-oriented planes
    #     set_combo = set(combo)
    #     set_x_plane_indices = set(x_plane_indices)
        
    #     common_elements = list(set_combo.intersection(set_x_plane_indices))
        
    #     labels = []
    #     for ele in common_elements:
    #         x_plane_pos = x_plane_indices.index(ele)
    #         if x_plane_pos == 0:
    #             labels.append("Left Support")
    #         elif x_plane_pos == 1:
    #             labels.append("Left Side Anvil")
    #         elif x_plane_pos == 2:
    #             labels.append("Right Side Anvil")
    #         elif x_plane_pos == 3:
    #             labels.append("Right Support")
    #         else:
    #             raise ValueError("Index out of range")
            
    #     if len(common_elements) == 2:
    #         print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} ({labels[1]}) are separated by {angle_diff:.4f} degrees')
    #     elif len(common_elements) == 1:
    #         if combo[0] in common_elements:
    #             print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} are separated by {angle_diff:.4f} degrees')
    #         else:
    #             print(f'Planes {combo[0]} and {combo[1]} ({labels[0]}) are separated by {angle_diff:.4f} degrees')                
    #     else:
    #         print(f'Planes {combo[0]} and {combo[1]} are separated by {angle_diff:.4f} degrees')
        
    #     if np.abs(np.around(angle_diff,2)) <= 0.01:
    #         separation_dist = separation_between_parallel_planes(optimized_planes[combo[0]], optimized_planes[combo[1]])
    #         print(f'\tPlanes {combo[0]} and {combo[1]} are separated by a distance of {separation_dist}')
    
    
    # # Re-align everything so the support planes and base plane define the world orientation
    # planeX_idx = x_plane_indices[0]
    # planeZ_idx = 0
    # planeX = optimized_planes[planeX_idx]
    # planeZ = optimized_planes[planeZ_idx]
    
    # # Perform alignment
    # rotated_pcd, rotated_planes, R_planes, info = align_planes_to_axes_minimal_v2(
    #     aligned_pcd, optimized_planes, planeX, planeZ
    # )
    
    # # Validate results
    # success, error_X, error_Z, total_angle = validate_minimal_rotation(
    #     optimized_planes, rotated_planes, planeX_idx, planeZ_idx, R_planes
    # )    
    
    # print(f'\nPlane adjustment R matrix:\n{R_planes}')
    
    # if R_flip is not None:
    #     R_total = R_planes @ R_90X @ R_flip @ R_pca
    # else:
    #     R_total = R_planes @ R_90X @ R_pca
    
    # print(f'\nR Total:\n{R_total}')
    
    # # === STEP 1: Load the original mesh ===
    # original_mesh = load_mesh(fixture_scan_path)
    # original_vertices = np.asarray(original_mesh.vertices)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
    
    # # === STEP 2: Translate to origin ===
    # original_mesh.translate(-centroid)
    
    # # === STEP 3: PCA alignment ===
    # original_mesh.rotate(R_pca, center=(0, 0, 0))
    
    # if R_flip is not None:
    #     original_mesh.rotate(R_flip, center=(0, 0, 0))
    
    # # === STEP 4: Rotate 90° about +X ===
    # original_mesh.rotate(R_90X, center=(0, 0, 0))
    
    # # === STEP 5: Apply final minimal rotation matrix R ===
    # original_mesh.rotate(R_planes, center=(0, 0, 0))
    
    # # === RESULT ===
    # transformed_reference_mesh = original_mesh
    
    # o3d.visualization.draw_geometries([
    #     aligned_pcd.paint_uniform_color([0.5, 0.5, 0.5]),
    #     transformed_reference_mesh.paint_uniform_color([1, 0, 0]),
    #     axis], window_name="Transformed Reference Mesh")

    # # Load and align matched specimen
    # # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    # matched_specimen_mesh = load_mesh(specimen_scan_path)
    
    # aligned_specimen_mesh = align_tgt_to_ref_meshes(transformed_reference_mesh, matched_specimen_mesh)
    
    # #### Create mesh models of support and anvil cylinders
    # diameter = 10
    # height = 40
    
    # base_plane = rotated_planes[0]
    # base_support_offset = 52
    
    # support_offset = 25.4   # 1 inch
    
    # anvil_plane1 = rotated_planes[5]
    # anvil_plane2 = rotated_planes[6]
    # anvil = create_cylinder_relative_to_planes(anvil_plane1,
    #                                         anvil_plane2,
    #                                         0,
    #                                         0,
    #                                         diameter,
    #                                         height)
    # anvil_mesh = trimesh_to_open3d(anvil)
    
    # l_support_plane = rotated_planes[x_plane_indices[0]]
    # l_support_plane_offset = support_offset
    # l_support = create_cylinder_relative_to_planes(
    #                 base_plane, l_support_plane,
    #                 base_support_offset,
    #                 l_support_plane_offset,
    #                 diameter, height)
    
    # l_support_mesh = trimesh_to_open3d(l_support)
    
    # r_support_plane = rotated_planes[x_plane_indices[3]]
    # r_support_plane_offset = -support_offset
    # r_support = create_cylinder_relative_to_planes(
    #                 base_plane, r_support_plane,
    #                 base_support_offset,
    #                 r_support_plane_offset,
    #                 diameter, height)
    
    # r_support_mesh = trimesh_to_open3d(r_support)
    
    # # Create coordinate frame
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])

    # # Visualize optimized planes and the inlier clouds used for fitting
    # cylinder_meshes = [anvil_mesh, l_support_mesh, r_support_mesh]
    # o3d.visualization.draw_geometries([rotated_pcd] + [axis] + cylinder_meshes + [aligned_specimen_mesh], window_name="Aligned Cylinders")
    
    # anvil = ensure_normals_outward(anvil, mesh_name="anvil")
    # l_support = ensure_normals_outward(l_support, mesh_name="left_support")
    # r_support = ensure_normals_outward(r_support, mesh_name="right_support")
    # flex_mesh = ensure_normals_outward(aligned_specimen_mesh, mesh_name="flex_mesh")
    
    # # Combine meshes
    # all_meshes = [flex_mesh, anvil, l_support, r_support]
    
    # intersecting_indices = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    
    # increment = 0.01  # Adjust the movement step as needed
    # max_iterations = 100  # Prevent infinite loops
    
    # if intersecting_indices:
    #     for _ in range(max_iterations):
    #         still_intersecting = False
    
    #         # Check and resolve intersection with anvil
    #         if all_meshes.index(anvil) in intersecting_indices:
    #             # Move anvil up in Z
    #             anvil.apply_translation([0, 0, increment])
    #             print(f"Applying {increment} adjustment to Z position of anvil")
    #             # Recheck
    #             updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    #             if all_meshes.index(anvil) in updated:
    #                 still_intersecting = True
    
    #         # Check and resolve intersection with supports
    #         l_idx = all_meshes.index(l_support)
    #         r_idx = all_meshes.index(r_support)
    #         if l_idx in intersecting_indices or r_idx in intersecting_indices:
    #             # Move both supports down in Z
    #             l_support.apply_translation([0, 0, -increment])
    #             r_support.apply_translation([0, 0, -increment])
    #             print(f"Applying -{increment} adjustment to Z position of supports")
    #             # Recheck
    #             updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    #             if l_idx in updated or r_idx in updated:
    #                 still_intersecting = True
    
    #         if not still_intersecting:
    #             break
    #     else:
    #         print("Warning: Maximum adjustment iterations reached, intersection may still exist.")
    
    # # Adjust for good measure
    # n_increments = 2
    # anvil.apply_translation([0, 0, n_increments*increment])
    # l_support.apply_translation([0, 0, -n_increments*increment])
    # r_support.apply_translation([0, 0, -n_increments*increment])
    
    # merged_mesh = trimesh.util.concatenate(all_meshes)
    
    # # output_filepath = "E:/Fixture Scans/prepared_test.stl"
    # merged_mesh.export(output_path)
    return all_keep_planes

def create_models(test_data_filepath, scanned_meshes_folder, prepared_meshes_folder):
    df_test_data = pd.read_excel(test_data_filepath)
    os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # Iterate through df_test_data and create the necessary multibody STL files for simulation
    for index, row in df_test_data.iterrows():
        fixture_scan_filename = row["Fixture Scan File"]
        fixture_scan_path = os.path.join(scanned_meshes_folder, fixture_scan_filename)
        
        specimen_scan_filename = row["Specimen Scan File"]
        specimen_scan_path = os.path.join(scanned_meshes_folder, specimen_scan_filename)
        
        specimen = row["Specimen"]
        test_num = row["Test_Num"]
        output_filename = f"{specimen}_{test_num}.stl"
        output_path = os.path.join(prepared_meshes_folder, output_filename)
        create_model(fixture_scan_path, specimen_scan_path, output_path)
        
        # Save the job name and add it to test_data.xlsx
        job_name = f"{specimen}_{test_num}"
        df_test_data.loc[index, "Job Name"] = job_name
        
        # Save the test specific mesh file name and add it to test_data.xlsx
        df_test_data.loc[index, "Test Specific Mesh File"] = output_path
        
        print(f"{output_filename} model and job creation complete")
        
    # Export changed dataframe
    df_test_data.to_excel(test_data_filepath, index=False)
        
    

if __name__ == "__main__":
    # test_data_filepath = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    # scanned_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    # prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    # create_models(test_data_filepath, scanned_meshes_folder, prepared_meshes_folder)
    
    
    # fixture_scan_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
    # fixture_scan_path = "E:/Fixture Scans/X2_1_Fixture.stl"
    fixture_scan_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Fixtures/X4_Test2_raw.stl"
    # specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    # specimen_scan_path = "E:/Fixture Scans/X2.stl"
    specimen_scan_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Specimens/X4_raw.stl"
    # output_path = "E:/Fixture Scans/X2_Test1.stl"
    output_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes/X4_Test2.stl"
    
    create_model(fixture_scan_path, specimen_scan_path, output_path)
    # all_keep_planes = create_model_v2(fixture_scan_path, specimen_scan_path, output_path)
    
    
    
    # mesh_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
    # print("Loading mesh...")
    # base_pcd = load_mesh_as_point_cloud(mesh_path)
    
    # target_axes = [[0, 0, 1],
    #                [1, 0, 0],
    #                [np.sqrt(3)/2, 0, -0.5],
    #                [-np.sqrt(3)/2, 0, -0.5]]
    
    # supports_distance = 127.66
    
    # keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, centroid = detect_fixture_planes(base_pcd, target_axes)
    
    # optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes(keep_planes, keep_inlier_clouds)
    
    # # Verification of constraints
    # print("")
    # for combo in combinations(range(len(optimized_planes)), 2):
    #     n1 = optimized_planes[combo[0]][:3]
    #     n2 = optimized_planes[combo[1]][:3]
    #     angle_diff = angular_change_between_normals(n1, n2)
        
    #     # Check if the current combo contains X-oriented planes
    #     set_combo = set(combo)
    #     set_x_plane_indices = set(x_plane_indices)
        
    #     common_elements = list(set_combo.intersection(set_x_plane_indices))
        
    #     labels = []
    #     for ele in common_elements:
    #         x_plane_pos = x_plane_indices.index(ele)
    #         if x_plane_pos == 0:
    #             labels.append("Left Support")
    #         elif x_plane_pos == 1:
    #             labels.append("Left Side Anvil")
    #         elif x_plane_pos == 2:
    #             labels.append("Right Side Anvil")
    #         elif x_plane_pos == 3:
    #             labels.append("Right Support")
    #         else:
    #             raise ValueError("Index out of range")
            
    #     if len(common_elements) == 2:
    #         print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} ({labels[1]}) are separated by {angle_diff:.4f} degrees')
    #     elif len(common_elements) == 1:
    #         if combo[0] in common_elements:
    #             print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} are separated by {angle_diff:.4f} degrees')
    #         else:
    #             print(f'Planes {combo[0]} and {combo[1]} ({labels[0]}) are separated by {angle_diff:.4f} degrees')                
    #     else:
    #         print(f'Planes {combo[0]} and {combo[1]} are separated by {angle_diff:.4f} degrees')
        
    #     if np.abs(np.around(angle_diff,2)) <= 0.01:
    #         separation_dist = separation_between_parallel_planes(optimized_planes[combo[0]], optimized_planes[combo[1]])
    #         print(f'\tPlanes {combo[0]} and {combo[1]} are separated by a distance of {separation_dist}')
    
    
    # # Re-align everything so the support planes and base plane define the world orientation
    # planeX_idx = x_plane_indices[0]
    # planeZ_idx = 0
    # planeX = optimized_planes[planeX_idx]
    # planeZ = optimized_planes[planeZ_idx]
    
    # # Perform alignment
    # rotated_pcd, rotated_planes, R_planes, info = align_planes_to_axes_minimal_v2(
    #     aligned_pcd, optimized_planes, planeX, planeZ
    # )
    
    # # Validate results
    # success, error_X, error_Z, total_angle = validate_minimal_rotation(
    #     optimized_planes, rotated_planes, planeX_idx, planeZ_idx, R_planes
    # )    
    
    # print(f'\nPlane adjustment R matrix:\n{R_planes}')
    
    # R_total = R_planes @ R_90X @ R_pca
    
    # print(f'\nR Total:\n{R_total}')
    
    # # === STEP 1: Load the original mesh ===
    # original_mesh = load_mesh(mesh_path)
    # original_vertices = np.asarray(original_mesh.vertices)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
    
    # # === STEP 2: Translate to origin ===
    # original_mesh.translate(-centroid)
    
    # # === STEP 3: PCA alignment ===
    # original_mesh.rotate(R_pca, center=(0, 0, 0))
    
    # # === STEP 4: Rotate 90° about +X ===
    # original_mesh.rotate(R_90X, center=(0, 0, 0))
    
    # # === STEP 5: Apply final minimal rotation matrix R ===
    # original_mesh.rotate(R_planes, center=(0, 0, 0))
    
    # # === RESULT ===
    # transformed_reference_mesh = original_mesh
    
    # o3d.visualization.draw_geometries([
    #     aligned_pcd.paint_uniform_color([0.5, 0.5, 0.5]),
    #     transformed_reference_mesh.paint_uniform_color([1, 0, 0]),
    #     axis], window_name="Transformed Reference Mesh")

    # # Load and align matched specimen
    # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    # matched_specimen_mesh = load_mesh(matched_specimen_scan_path)
    
    # aligned_specimen_mesh = align_tgt_to_ref_meshes(transformed_reference_mesh, matched_specimen_mesh)
    
    # #### Create mesh models of support and anvil cylinders
    # diameter = 10
    # height = 40
    
    # base_plane = rotated_planes[0]
    # base_support_offset = 52
    
    # support_offset = 25.4   # 1 inch
    
    # anvil_plane1 = rotated_planes[5]
    # anvil_plane2 = rotated_planes[6]
    # anvil = create_cylinder_relative_to_planes(anvil_plane1,
    #                                         anvil_plane2,
    #                                         0,
    #                                         0,
    #                                         diameter,
    #                                         height)
    # anvil_mesh = trimesh_to_open3d(anvil)
    
    # l_support_plane = rotated_planes[x_plane_indices[0]]
    # l_support_plane_offset = support_offset
    # l_support = create_cylinder_relative_to_planes(
    #                 base_plane, l_support_plane,
    #                 base_support_offset,
    #                 l_support_plane_offset,
    #                 diameter, height)
    
    # l_support_mesh = trimesh_to_open3d(l_support)
    
    # r_support_plane = rotated_planes[x_plane_indices[3]]
    # r_support_plane_offset = -support_offset
    # r_support = create_cylinder_relative_to_planes(
    #                 base_plane, r_support_plane,
    #                 base_support_offset,
    #                 r_support_plane_offset,
    #                 diameter, height)
    
    # r_support_mesh = trimesh_to_open3d(r_support)
    
    # # Create coordinate frame
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])

    # # Visualize optimized planes and the inlier clouds used for fitting
    # cylinder_meshes = [anvil_mesh, l_support_mesh, r_support_mesh]
    # o3d.visualization.draw_geometries([rotated_pcd] + [axis] + cylinder_meshes + [aligned_specimen_mesh], window_name="Aligned Cylinders")
    
    # anvil = ensure_normals_outward(anvil, mesh_name="anvil")
    # l_support = ensure_normals_outward(l_support, mesh_name="left_support")
    # r_support = ensure_normals_outward(r_support, mesh_name="right_support")
    # flex_mesh = ensure_normals_outward(aligned_specimen_mesh, mesh_name="flex_mesh")
    
    # # Combine meshes
    # all_meshes = [flex_mesh, anvil, l_support, r_support]
    
    # intersecting_indices = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    
    # increment = 0.01  # Adjust the movement step as needed
    # max_iterations = 100  # Prevent infinite loops
    
    # if intersecting_indices:
    #     for _ in range(max_iterations):
    #         still_intersecting = False
    
    #         # Check and resolve intersection with anvil
    #         if all_meshes.index(anvil) in intersecting_indices:
    #             # Move anvil up in Z
    #             anvil.apply_translation([0, 0, increment])
    #             print(f"Applying {increment} adjustment to Z position of anvil")
    #             # Recheck
    #             updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    #             if all_meshes.index(anvil) in updated:
    #                 still_intersecting = True
    
    #         # Check and resolve intersection with supports
    #         l_idx = all_meshes.index(l_support)
    #         r_idx = all_meshes.index(r_support)
    #         if l_idx in intersecting_indices or r_idx in intersecting_indices:
    #             # Move both supports down in Z
    #             l_support.apply_translation([0, 0, -increment])
    #             r_support.apply_translation([0, 0, -increment])
    #             print(f"Applying -{increment} adjustment to Z position of supports")
    #             # Recheck
    #             updated = check_vertex_intersections(flex_mesh, all_meshes, threshold=1e-4)
    #             if l_idx in updated or r_idx in updated:
    #                 still_intersecting = True
    
    #         if not still_intersecting:
    #             break
    #     else:
    #         print("Warning: Maximum adjustment iterations reached, intersection may still exist.")
    
    
    # merged_mesh = trimesh.util.concatenate(all_meshes)
    
    # output_filepath = "E:/Fixture Scans/prepared_test.stl"
    # merged_mesh.export(output_filepath)
    
