# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:31:41 2025

@author: Ryan.Larson
"""

import open3d as o3d
import numpy as np
import sys


def load_and_sample(filepath, num_points=5000):
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return mesh, pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals()
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source, target, voxel_size):
    src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )
    return result


def refine_with_icp(source, target, init_trans, threshold=5.0):
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())


def align_meshes(ref_path, tgt_path, output_path, visualize=True):
    print("Loading meshes...")
    mesh_ref, pcd_ref = load_and_sample(ref_path)
    mesh_tgt, pcd_tgt = load_and_sample(tgt_path)

    print("Running global alignment (RANSAC)...")
    voxel_size = 5.0
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print("RANSAC fitness:", result_ransac.fitness)

    if result_ransac.fitness == 0.0:
        print("Warning: RANSAC failed to find good alignment.")
    
    print("Running ICP refinement...")
    result_icp = refine_with_icp(pcd_tgt, pcd_ref, result_ransac.transformation, threshold=10.0)
    print("ICP fitness:", result_icp.fitness)
    print("ICP inlier RMSE:", result_icp.inlier_rmse)

    # Apply final transformation to mesh B
    mesh_tgt.transform(result_icp.transformation)

    # Save aligned mesh B
    o3d.io.write_triangle_mesh(output_path, mesh_tgt)
    print(f"Aligned mesh B saved to {output_path}")

    if visualize:
        mesh_ref.paint_uniform_color([1, 0.706, 0])  # Yellow
        mesh_tgt.paint_uniform_color([0, 0.651, 0.929])  # Blue
        o3d.visualization.draw_geometries([mesh_ref, mesh_tgt])


if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("Usage: python align_meshes.py <mesh_A_path> <mesh_B_path> <output_B_aligned_path>")
    #     sys.exit(1)

    # mesh_a = sys.argv[1]
    # mesh_b = sys.argv[2]
    # output_b = sys.argv[3]
    
    directory = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization"
    
    mesh_a = directory + "/" + "Merge_01_mesh.stl"
    mesh_b = directory + "/" + "0613_02_mesh.stl"
    output_b = directory + "/" + "0613_02_mesh_aligned.stl"
    
    align_meshes(mesh_a, mesh_b, output_b)
