# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:53:22 2025

@author: Ryan.Larson
"""

import open3d as o3d
import numpy as np

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

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    o3d.visualization.draw_geometries([aligned_pcd, coord_frame])

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


def cluster_point_normals(pcd, threshold=0.9):
    normals = np.asarray(pcd.normals)
    cluster_ids = np.array([classify_normal(n, threshold) for n in normals])

    # Map cluster to color
    color_map = {
        0: [1.0, 0.0, 0.0],  # X = red
        1: [0.0, 1.0, 0.0],  # Y = green
        2: [0.0, 0.0, 1.0],  # Z = blue
        -1: [0.5, 0.5, 0.5]  # unclassified = gray
    }
    colors = np.array([color_map.get(cid, [0.5, 0.5, 0.5]) for cid in cluster_ids])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    return pcd, cluster_ids

if __name__ == "__main__":
    directory = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization"
    output_filename = directory + "/" + "Plane_analysis.xlsx"
    
    mesh_a = directory + "/" + "Merge_01_mesh.stl"
    
    # Step 1: PCA Align
    aligned_pcd, basis_matrix, center = align_mesh_with_pca(mesh_a)

    # Step 2: Estimate normals
    aligned_pcd = estimate_normals(aligned_pcd, radius=10, max_nn=30)

    # Step 3: Classify by normal direction
    clustered_pcd, cluster_ids = cluster_point_normals(aligned_pcd, threshold=0.9)