# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:53:22 2025

@author: Ryan.Larson
"""

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from itertools import combinations

def align_mesh_with_pca(mesh_path, sample_points=300000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    points = np.asarray(pcd.points)

    centroid = points.mean(axis=0)
    centered = points - centroid

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # [X, Y, Z] = [Vt[0], Vt[2], Vt[1]]
    new_basis = np.vstack([Vt[0], Vt[2], Vt[1]]).T
    aligned = centered @ new_basis

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
    aligned_pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    # o3d.visualization.draw_geometries([aligned_pcd, coord_frame])

    return aligned_pcd, new_basis, centroid


def estimate_normals(pcd, radius=10, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    return pcd


def classify_normal(normal, threshold=0.9):
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
        return -1  # Invalid
    normal = normal / np.linalg.norm(normal)
    axis_vecs = {
        0: np.array([1, 0, 0]),  # X
        1: np.array([0, 1, 0]),  # Y
        2: np.array([0, 0, 1])   # Z
    }
    for axis_id, axis in axis_vecs.items():
        if abs(np.dot(normal, axis)) >= threshold:
            return axis_id
    return -1


def cluster_point_normals(pcd, n_clusters=6, min_cluster_size=1000):
    normals = np.asarray(pcd.normals)
    valid_mask = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 0)
    normals = normals[valid_mask]
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    cluster_ids = kmeans.fit_predict(normals)
    
    # Compute representative normal (mean of unit vectors in cluster)
    representative_normals = np.zeros((n_clusters, 3))
    for cid in range(n_clusters):
        cluster_normals = normals[cluster_ids == cid]
        mean_normal = np.mean(cluster_normals, axis=0)
        norm = np.linalg.norm(mean_normal)
        representative_normals[cid] = mean_normal / norm if norm > 0 else [0, 0, 0]

    # Prepare colors
    color_palette = plt.cm.get_cmap("tab10", n_clusters)
    full_colors = np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1))  # Default gray
    cluster_id_full = -np.ones(len(pcd.points), dtype=int)

    j = 0  # index into filtered (valid) points
    for i in range(len(pcd.points)):
        if valid_mask[i]:
            cid = cluster_ids[j]
            cluster_id_full[i] = cid
            full_colors[i] = color_palette(cid)[:3]
            j += 1

    pcd.colors = o3d.utility.Vector3dVector(full_colors)
    o3d.visualization.draw_geometries([pcd])
    
    return pcd, cluster_id_full, representative_normals

# def cluster_by_dimension(pcd, axis=0, n_clusters=3, min_cluster_size=1000):
#     pts = np.asarray(pcd.points)
#     axis_pos = pts[:, axis]     # Values on the axis for each point

def compute_plane(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None, None
    normal /= norm
    d = -np.dot(normal, p1)
    return normal, d

def angle_between_normals(n1, n2):
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.arccos(dot) * 180.0 / np.pi  # degrees

def custom_ransac_plane(points, target_normal, angle_tol_deg=10, distance_thresh=0.01, num_iterations=1000):
    best_inliers = []
    best_model = None

    target_normal = target_normal / np.linalg.norm(target_normal)
    n_points = points.shape[0]

    for _ in range(num_iterations):
        idx = random.sample(range(n_points), 3)
        p1, p2, p3 = points[idx]
        normal, d = compute_plane(p1, p2, p3)
        if normal is None:
            continue

        angle = angle_between_normals(normal, target_normal)
        if angle > angle_tol_deg and abs(180 - angle) > angle_tol_deg:
            continue  # Skip if not aligned (or anti-aligned) with target

        # Compute distance of all points to plane
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (normal, d)

    return best_model, best_inliers

def detect_planes(pcd, distance_threshold=0.5, ransac_n=3, num_iterations=1000, min_ratio=0.01):
    """
    Detects multiple planes in the point cloud using RANSAC.
    Returns a list of (plane_model, inliers), where:
      - plane_model is [a, b, c, d] for ax + by + cz + d = 0
      - inliers are indices in pcd.points
    """
    planes = []
    remaining_pcd = pcd
    n_points = len(pcd.points)
    while True:
        if len(remaining_pcd.points) < n_points * min_ratio:
            break
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        if len(inliers) < n_points * min_ratio:
            break
        planes.append((plane_model, inliers))
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    return planes

def are_normals_orthogonal(n1, n2, angle_tol=np.deg2rad(5)):
    """
    Returns True if n1 and n2 are orthogonal within angle_tol.
    """
    dot = np.abs(np.dot(n1, n2))
    return np.abs(dot) < np.sin(angle_tol)

def filter_orthogonal_planes(planes, angle_tol_deg=2):
    """
    Returns plane-inlier pairs for planes in mutually orthogonal triads.
    """
    angle_tol = np.deg2rad(angle_tol_deg)
    normals = [np.array(p[0][:3]) / np.linalg.norm(p[0][:3]) for p in planes]

    triads = []
    for i, j, k in combinations(range(len(planes)), 3):
        n1, n2, n3 = normals[i], normals[j], normals[k]
        if (
            are_normals_orthogonal(n1, n2, angle_tol) and
            are_normals_orthogonal(n1, n3, angle_tol) and
            are_normals_orthogonal(n2, n3, angle_tol)
        ):
            triads.append((i, j, k))

    seen = set()
    unique_planes = []
    for triad in triads:
        for idx in triad:
            key = plane_key(planes[idx][0])
            if key not in seen:
                seen.add(key)
                unique_planes.append(planes[idx])  # Keep both model and inliers

    return unique_planes

def create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0, color=[1, 0, 0]):
    """
    Create a visual mesh (rectangle) representing a plane model over the inlier cloud.
    """
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)

    # Create two orthogonal vectors on the plane
    if np.abs(normal[2]) > 1e-3:
        tangent1 = np.cross(normal, [1, 0, 0])
    else:
        tangent1 = np.cross(normal, [0, 0, 1])
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    tangent2 /= np.linalg.norm(tangent2)

    # Create a quad representing the plane
    points = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            pt = centroid + dx * plane_size * tangent1 + dy * plane_size * tangent2
            points.append(pt)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 1, 3]])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh
    
def plane_key(plane_model, tol=1e-3):
    """
    Returns a tuple that uniquely identifies a plane up to tolerance.
    Normalized normal and rounded offset.
    """
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    d = plane_model[3]
    key = tuple(np.round(np.append(normal, d), int(-np.log10(tol))))
    return key

def get_unique_planes_from_pairs(orthogonal_pairs, tol=1e-3):
    """
    From orthogonal_pairs, extract a list of unique plane models.
    """
    seen = set()
    unique_planes = []
    for (p1, _), (p2, _) in orthogonal_pairs:
        for p in (p1, p2):
            key = plane_key(p, tol)
            if key not in seen:
                seen.add(key)
                unique_planes.append(p)
    return unique_planes

def find_inlier_cloud_for_plane(pcd, plane_model, all_planes, all_inliers):
    """
    Find the inlier cloud for the given plane model.
    """
    for (pm, inlier_idx) in zip(all_planes, all_inliers):
        if np.allclose(pm, plane_model, atol=1e-4):
            return pcd.select_by_index(inlier_idx)
    return None
        
def visualize_unique_planes(pcd, orthogonal_pairs, all_planes, max_planes=20):
    """
    Visualize only the unique planes in the orthogonal pairs.
    """
    # Separate plane models and inliers
    all_plane_models = [p[0] for p in all_planes]
    all_inliers = [p[1] for p in all_planes]

    unique_plane_models = get_unique_planes_from_pairs(orthogonal_pairs)
    print(f"Found {len(unique_plane_models)} unique planes")

    vis_elems = [pcd]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max_planes))[:, :3]

    for i, plane_model in enumerate(unique_plane_models[:max_planes]):
        inlier_cloud = find_inlier_cloud_for_plane(pcd, plane_model, all_plane_models, all_inliers)
        if inlier_cloud:
            mesh = create_plane_mesh(plane_model, inlier_cloud, color=colors[i % len(colors)])
            vis_elems.append(mesh)

    o3d.visualization.draw_geometries(vis_elems)

def visualize_planes_from_list(pcd, filtered_planes, max_planes=20):
    """
    Visualize given list of (plane_model, inliers) with unique colors.
    """
    vis_elems = [pcd]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max_planes))[:, :3]

    for i, (plane_model, inliers) in enumerate(filtered_planes[:max_planes]):
        inlier_cloud = pcd.select_by_index(inliers)
        mesh = create_plane_mesh(plane_model, inlier_cloud, color=colors[i % len(colors)])
        vis_elems.append(mesh)

    o3d.visualization.draw_geometries(vis_elems)

if __name__ == "__main__":
    directory = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization"
    output_filename = directory + "/" + "Plane_analysis.xlsx"
    
    mesh_a = directory + "/" + "Orthogonal_Plane_Test.stl"
    # mesh_a = directory + "/" + "Merge_01_mesh.stl"
    
    # Step 1: PCA Align
    aligned_pcd, basis_matrix, center = align_mesh_with_pca(mesh_a)

    # Step 2: Estimate normals
    aligned_pcd = estimate_normals(aligned_pcd, radius=10, max_nn=30)

    # # Step 3: Classify by normal direction
    # clustered_pcd, cluster_ids, representative_normals = cluster_point_normals(aligned_pcd, n_clusters=6)
    
    # # Define direction and run
    # target_normal = np.array([1, 0, 0])  # Looking for X-aligned plane
    # model, inliers = custom_ransac_plane(np.asarray(aligned_pcd.points), target_normal, angle_tol_deg=10, distance_thresh=0.01)
    
    # # Visualize
    # if model is not None:
    #     print(f"{len(inliers)} inliers found.")
    #     aligned_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    #     plane_cloud = aligned_pcd.select_by_index(inliers, invert=False)
    #     plane_cloud.paint_uniform_color([1, 0, 0])
    #     o3d.visualization.draw_geometries([aligned_pcd, plane_cloud])
    # else:
    #     print("No valid plane found.")
    
    # Assume `aligned_pcd` is your aligned open3d.geometry.PointCloud object
    planes = detect_planes(aligned_pcd, distance_threshold=0.2, ransac_n=3, num_iterations=1000, min_ratio=0.01)
    
    # # Instead of orthogonal_pairs, we now get filtered_planes directly
    # filtered_planes = filter_orthogonal_planes(planes)
    
    # # To reuse visualization, simulate orthogonal_pairs structure
    # orthogonal_pairs = [((p, np.array(p[:3]) / np.linalg.norm(p[:3])), (p, np.array(p[:3]) / np.linalg.norm(p[:3]))) for p, _ in filtered_planes]
    
    # Visualize
    # visualize_unique_planes(aligned_pcd, orthogonal_pairs, planes)
    visualize_planes_from_list(aligned_pcd, planes)
    # visualize_planes_from_list(aligned_pcd, filtered_planes)

    