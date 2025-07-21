# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:03:34 2025

@author: Ryan.Larson
"""

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import copy
import os

def load_mesh_as_point_cloud(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

def create_plane_mesh(plane_model, inlier_cloud, plane_size=20.0, color=None):
    # Create a square plane oriented by the plane normal
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Get centroid of inlier points
    # centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    centroid = -d * normal

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

def detect_and_correct_pca(pcd):    
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
    
    # print(f'\nPCA Rotation Matrix:\n{R_pca}')
    # print(f'PCA Centroid:\t{centroid}')
    
    # Rotate the point cloud about X so the greatest variation aligns with Z
    rotation_angle = np.radians(90)
    axis_angle = np.array([rotation_angle, 0, 0])
    R_90X = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(R_90X, center=(0,0,0))
    
    return pcd, R_pca, R_90X, R_flip, centroid

def angle_between_vectors(v1, v2):
    """Returns the minimum angle between two vectors in radians, adjusting for 180° ambiguity."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot)
    return min(angle, np.pi - angle)

def fit_ransac_plane(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        return plane_model, inliers
    except:
        return None, []

def remove_close_planes(planes, angle_threshold=np.radians(5), dist_threshold=0.01):
    """Filter planes that are similar in normal direction and offset."""
    filtered = []
    for n1, d1 in planes:
        keep = True
        for n2, d2 in filtered:
            if angle_between_vectors(n1, n2) < angle_threshold and (d1 - d2) < dist_threshold:
            # if angle_between_vectors(n1, n2) < angle_threshold and abs(d1 - d2) < dist_threshold:
                keep = False
                break
        if keep:
            filtered.append((n1, d1))
    return filtered

def detect_planes_by_axis_clustering(pcd, expected_counts,
                                     max_planes=30,
                                     angle_threshold_deg=20,
                                     min_inliers=100,
                                     distance_threshold=0.01,
                                     num_iterations=1000):
    """
    Fit multiple planes to the full point cloud and cluster them by axis alignment.

    Args:
        pcd: Open3D point cloud
        expected_counts: dict of {axis_index: (target_axis_vector, expected_count)}
        max_planes: max number of RANSAC planes to try extracting
        angle_threshold_deg: angular tolerance for axis alignment (in degrees)
        min_inliers: minimum inliers required to accept a plane
        distance_threshold: RANSAC inlier distance threshold
        num_iterations: number of RANSAC iterations

    Returns:
        final_planes: dict of {axis_idx: list of (normal, d) tuples}
        final_inliers: dict of {axis_idx: list of Open3D point clouds}
    """
    remaining = pcd
    remaining_indices = np.arange(len(pcd.points))  # Track original indices
    all_planes_with_inliers = []

    # Step 1: Run sequential RANSAC
    for _ in range(max_planes):
        model, inliers = fit_ransac_plane(np.asarray(remaining.points), distance_threshold, num_iterations=num_iterations)
        if model is None or len(inliers) < min_inliers:
            break
        
        normal = np.array(model[:3])
        d = model[3]
        
        # Map relative indices back to original indices
        original_inliers = remaining_indices[inliers]
        all_planes_with_inliers.append((normal, d, original_inliers))
        
        # Update remaining points and indices
        remaining = remaining.select_by_index(inliers, invert=True)
        remaining_indices = remaining_indices[np.setdiff1d(np.arange(len(remaining_indices)), inliers)]

    # Step 2: Assign planes to closest axis clusters
    axis_plane_groups = {axis_idx: [] for axis_idx in expected_counts}
    angle_threshold_rad = np.radians(angle_threshold_deg)

    for normal, d, original_inliers in all_planes_with_inliers:
        for axis_idx, (axis_vec, expected_count) in expected_counts.items():
            angle = angle_between_vectors(normal, axis_vec)
            if angle < angle_threshold_rad:
                axis_plane_groups[axis_idx].append((normal, d, original_inliers))
                break  # Assign to first matching axis only

    # Step 3: Filter and verify counts
    final_planes = {}
    final_inliers = {}

    def remove_close_planes_with_inliers(plane_data, angle_threshold, dist_threshold):
        """Filter (normal, d, inliers) tuples based on angle and distance."""
        filtered = []
        for n1, d1, inl1 in plane_data:
            keep = True
            for n2, d2, _ in filtered:
                if angle_between_vectors(n1, n2) < angle_threshold and abs(d1 - d2) < dist_threshold:
                    keep = False
                    break
            if keep:
                filtered.append((n1, d1, inl1))
        return filtered

    for axis_idx, plane_data in axis_plane_groups.items():
        filtered_data = remove_close_planes_with_inliers(
            plane_data,
            angle_threshold=angle_threshold_rad,
            dist_threshold=distance_threshold
        )
        
        # Remove all but the lowest plane for target axis [0, 0, 1]
        if axis_idx == 0:
            centroids = [-d * normal for normal, d, _ in filtered_data]
            centroids = np.asarray(centroids)
            min_index = np.argmin(centroids[:,2])
            filtered_data = [filtered_data[min_index]]
            
        expected = expected_counts[axis_idx][1]
        # if len(filtered_data) != expected:
        #     print(f"[WARN] Axis {axis_idx}: expected {expected} planes, found {len(filtered_data)} after filtering.")
        if len(filtered_data) > expected:
            # print(f"[WARN] Axis {axis_idx}: expected {expected} planes, found {len(filtered_data)} after filtering.")
            filtered_data = [filtered_data[0]]
        elif len(filtered_data) < expected:
            raise Exception(f"Did not find {expected} planes for Axis {axis_idx}")
        else:
            print(f"[INFO] Axis {axis_idx}: found {len(filtered_data)} planes, as expected (SUCCESS)")

        final_planes[axis_idx] = [(n, d) for (n, d, _) in filtered_data]
        final_inliers[axis_idx] = [pcd.select_by_index(inl) for (_, _, inl) in filtered_data]

    return final_planes, final_inliers

def create_model(fixture_scan_path, expected_planes, visualization=True):
    print(f'Loading {os.path.basename(fixture_scan_path)}...')
    pcd = load_mesh_as_point_cloud(fixture_scan_path)
    
    print('Reorienting point cloud...')
    aligned_pcd, R_pca, R_90X, R_flip, centroid = prepare_scan_orientation(pcd, is_point_cloud=True)
    
    print('Detecting planes...')
    final_planes, final_inliers = detect_planes_by_axis_clustering(aligned_pcd,
                                                      expected_planes,
                                                      max_planes=200,
                                                      min_inliers=50,
                                                      distance_threshold=0.2,
                                                      num_iterations=2000)
    all_plane_models = []
    all_inlier_clouds = []
    plane_meshes = []
    for key in final_planes:
        plane_models = []
        inlier_clouds = []
        for i in range(len(final_planes[key])):
            [a, b, c] = final_planes[key][i][0]
            d = final_planes[key][i][1]
            plane_model = [a, b, c, d]
            all_plane_models.append(plane_model)
            
            # plane_model = final_planes[key][i]
            inlier_cloud = final_inliers[key][i]
            all_inlier_clouds.append(inlier_cloud)
            
            # plane_model = planes_by_axis[key]
            plane_mesh = create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0)
            plane_meshes.append(plane_mesh)
    
    if visualization:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([axis] + all_inlier_clouds + plane_meshes)
        # o3d.visualization.draw_geometries([aligned_pcd, axis] + plane_meshes)
        
    return all_plane_models, all_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid

if __name__ == "__main__":
    fixture_scan_path = "E:/Fixture Scans/X2_1_Fixture.stl"
    specimen_scan_path = "E:/Fixture Scans/X2_positive_quad.stl"
    output_path = "E:/Fixture Scans/X2_Test1.stl"
    
    expected_planes = {
        0: (np.array([0, 0, 1]), 1),
        1: (np.array([1, 0, 0]), 4),
        2: (np.array([np.sqrt(3)/2, 0, -0.5]), 1),
        3: (np.array([-np.sqrt(3)/2, 0, -0.5]), 1)
    }
    
    keep_planes, keep_inlier_clouds, aligned_pcd, R_pca, R_90X, R_flip, centroid = create_model(fixture_scan_path, expected_planes, visualization=True)
    
    
    
    # # fixture_scan_path = "E:/Fixture Scans/X2_1_Fixture.stl"
    # print('Loading mesh...')
    # pcd = load_mesh_as_point_cloud(fixture_scan_path)
    
    # expected_planes = {
    #     0: (np.array([1, 0, 0]), 4),
    #     1: (np.array([0, 0, 1]), 1),
    #     2: (np.array([np.sqrt(3)/2, 0, -0.5]), 1),
    #     3: (np.array([-np.sqrt(3)/2, 0, -0.5]), 1)
    # }
    
    # print('Reorienting point cloud...')
    # pcd, R_pca, R_90X, R_flip, centroid = prepare_scan_orientation(pcd, is_point_cloud=True)
    
    # print('Detecting planes...')
    # final_planes, final_inliers = detect_planes_by_axis_clustering(pcd,
    #                                                   expected_planes,
    #                                                   max_planes=200,
    #                                                   min_inliers=50,
    #                                                   distance_threshold=0.2,
    #                                                   num_iterations=2000)
    # all_plane_models = []
    # all_inlier_clouds = []
    # plane_meshes = []
    # for key in final_planes:
    #     plane_models = []
    #     inlier_clouds = []
    #     for i in range(len(final_planes[key])):
    #         [a, b, c] = final_planes[key][i][0]
    #         d = final_planes[key][i][1]
    #         plane_model = [a, b, c, d]
    #         all_plane_models.append(plane_model)
            
    #         # plane_model = final_planes[key][i]
    #         inlier_cloud = final_inliers[key][i]
    #         all_inlier_clouds.append(inlier_cloud)
            
    #         # plane_model = planes_by_axis[key]
    #         plane_mesh = create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0)
    #         plane_meshes.append(plane_mesh)
    
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis] + plane_meshes)