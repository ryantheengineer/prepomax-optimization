# -*- coding: utf-8 -*-
"""
Enhanced mesh alignment with multi-scale ICP refinement for sub-0.1mm accuracy
Created on Fri Jun 13 10:31:41 2025
@author: Ryan.Larson
"""
import open3d as o3d
import numpy as np
import sys
import copy

def load_and_sample(filepath, num_points=5000):
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return mesh, pcd

# def load_mesh(filepath):
#     mesh = o3d.io.read_triangle_mesh(filepath)
#     mesh.compute_vertex_normals()
#     return mesh

def print_verbose(printstring, verbose=False):
    if verbose:
        print(printstring)

def load_mesh(filepath, verbose=False):
    """
    Load a mesh from file and ensure all normals point outward.
    
    Parameters:
    - filepath: Path to the mesh file
    
    Returns:
    - Open3D TriangleMesh with outward-facing normals
    """
    mesh = o3d.io.read_triangle_mesh(filepath)
    
    # Check if mesh loaded successfully
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {filepath}")
    
    # # Remove degenerate triangles and duplicated vertices
    # mesh.remove_degenerate_triangles()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_duplicated_vertices()
    # mesh.remove_non_manifold_edges()
    
    # Ensure consistent triangle orientation (outward normals)
    # This works best for closed, manifold meshes
    try:
        mesh.orient_triangles()
        print_verbose("[INFO] Triangle orientation corrected", verbose)
    except Exception as e:
        print_verbose(f"[WARNING] Could not orient triangles automatically: {e}", verbose)
    
    
    # Optional: If you know the mesh should be a closed surface,
    # you can verify and potentially fix orientation using this approach:
    if mesh.is_watertight():
        # Compute vertex normals after orientation
        mesh.compute_vertex_normals()
        
        # For additional robustness, you can also compute triangle normals
        mesh.compute_triangle_normals()
        print_verbose("[INFO] Mesh is watertight - normals should be correctly oriented", verbose)
    else:
        print_verbose("[WARNING] Mesh is not watertight - normal orientation may be inconsistent", verbose)
        
        # Alternative approach for non-watertight meshes:
        # Try to orient normals based on majority direction or centroid
        try:
            # Get mesh centroid
            centroid = mesh.get_center()
            vertices = np.asarray(mesh.vertices)
            normals = np.asarray(mesh.vertex_normals)
            
            # For each vertex, check if normal points away from centroid
            vectors_to_centroid = centroid - vertices
            dot_products = np.sum(normals * vectors_to_centroid, axis=1)
            
            # If majority of normals point toward centroid, flip all normals
            inward_count = np.sum(dot_products > 0)
            total_count = len(dot_products)
            
            if inward_count > total_count / 2:
                print_verbose("[INFO] Flipping normals to point outward from centroid", verbose)
                # Flip triangle orientation
                triangles = np.asarray(mesh.triangles)
                mesh.triangles = o3d.utility.Vector3iVector(triangles[:, [0, 2, 1]])
                # Recompute normals
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                
        except Exception as e:
            print_verbose(f"[WARNING] Could not apply centroid-based normal correction: {e}", verbose)
    
    return mesh

def mesh_to_pcd(mesh, num_points=5000):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd    

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals()
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source, target, voxel_size):
    src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)
    
    # Enhanced RANSAC parameters for better accuracy
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.2,  # Tighter correspondence
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),  # Stricter edge length check
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.2),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 2000)  # More iterations
    )
    return result

def multi_scale_icp(source, target, init_trans, thresholds=[2.0, 1.0, 0.5, 0.2, 0.1], verbose=False):
    """
    Perform multi-scale ICP refinement with progressively tighter thresholds
    """
    current_trans = init_trans
    
    for i, threshold in enumerate(thresholds):
        print_verbose(f"ICP refinement step {i+1}/{len(thresholds)}, threshold: {threshold}mm", verbose)
        
        # Use point-to-plane for better convergence in later stages
        if i >= 2:  # Use point-to-plane for final refinements
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, current_trans,
            estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8,
                relative_rmse=1e-8,
                max_iteration=200
            )
        )
        
        current_trans = result.transformation
        print_verbose(f"  Fitness: {result.fitness:.6f}, RMSE: {result.inlier_rmse:.6f}mm", verbose)
        
        # Early termination if convergence is very good
        if result.inlier_rmse < 0.05:
            print_verbose(f"  Early termination - excellent convergence achieved", verbose)
            break
    
    return result

def colored_icp_refinement(source, target, init_trans, threshold=0.2, verbose=False):
    """
    Apply colored ICP for final refinement if meshes have color information
    """
    # Check if both point clouds have colors
    if not source.has_colors() or not target.has_colors():
        print_verbose("Colored ICP not applicable (point clouds don't have color information)", verbose)
        return None
    
    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            source, target, threshold, init_trans,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8,
                relative_rmse=1e-8,
                max_iteration=100
            )
        )
        print_verbose(f"Colored ICP - Fitness: {result.fitness:.6f}, RMSE: {result.inlier_rmse:.6f}mm", verbose)
        return result
    except Exception as e:
        print_verbose(f"Colored ICP failed: {str(e)}", verbose)
        return None

def compute_alignment_metrics(source, target, transformation=None, max_distance=1.0):
    """
    Compute detailed alignment metrics
    """
    if transformation is not None:
        source_transformed = source.transform(transformation)
    else:
        source_transformed = source
    distances = np.asarray(source_transformed.compute_point_cloud_distance(target))
    
    # Filter out outliers for more meaningful statistics
    valid_distances = distances[distances < max_distance]
    
    metrics = {
        'mean_distance': np.mean(valid_distances),
        'median_distance': np.median(valid_distances),
        'std_distance': np.std(valid_distances),
        'max_distance': np.max(valid_distances),
        'percentile_95': np.percentile(valid_distances, 95),
        'percentile_99': np.percentile(valid_distances, 99),
        'num_valid_points': len(valid_distances),
        'total_points': len(distances)
    }
    
    return metrics

def align_meshes_path(ref_path, tgt_path, target_accuracy=0.1):
    print("Loading meshes...")
    mesh_ref, pcd_ref = load_and_sample(ref_path, num_points=10000)  # More points for better accuracy
    mesh_tgt, pcd_tgt = load_and_sample(tgt_path, num_points=10000)
    
    # Ensure normals are computed for point-to-plane ICP
    pcd_ref.estimate_normals()
    pcd_tgt.estimate_normals()
    
    print("Running global alignment (Enhanced RANSAC)...")
    voxel_size = 2.0  # Smaller voxel size for better initial alignment
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
    if result_ransac.fitness < 0.1:
        print("Warning: RANSAC fitness is low. Consider adjusting parameters or checking mesh compatibility.")
    
    print("\nRunning multi-scale ICP refinement...")
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation)
    
    # Try colored ICP if available
    print("\nAttempting colored ICP refinement...")
    result_colored = colored_icp_refinement(pcd_tgt, pcd_ref, result_icp.transformation)
    
    # Use the best result
    final_result = result_colored if result_colored and result_colored.inlier_rmse < result_icp.inlier_rmse else result_icp
    final_transformation = final_result.transformation
    
    print(f"\nFinal alignment metrics:")
    print(f"  Final fitness: {final_result.fitness:.6f}")
    print(f"  Final RMSE: {final_result.inlier_rmse:.6f}mm")
    
    # Compute detailed alignment metrics
    print("\nComputing detailed alignment metrics...")
    pcd_tgt_copy = copy.deepcopy(pcd_tgt)
    pcd_tgt_copy.transform(final_transformation)
    metrics = compute_alignment_metrics(pcd_tgt_copy, pcd_ref)
    
    print(f"Alignment Quality Report:")
    print(f"  Mean distance: {metrics['mean_distance']:.4f}mm")
    print(f"  Median distance: {metrics['median_distance']:.4f}mm")
    print(f"  Std deviation: {metrics['std_distance']:.4f}mm")
    print(f"  95th percentile: {metrics['percentile_95']:.4f}mm")
    print(f"  99th percentile: {metrics['percentile_99']:.4f}mm")
    print(f"  Max distance: {metrics['max_distance']:.4f}mm")
    print(f"  Valid points: {metrics['num_valid_points']}/{metrics['total_points']}")
    
    # Check if target accuracy is achieved
    if metrics['percentile_95'] <= target_accuracy:
        print(f"✓ Target accuracy of {target_accuracy}mm achieved!")
    else:
        print(f"⚠ Target accuracy of {target_accuracy}mm not achieved. Consider:")
        print("  - Increasing point cloud density")
        print("  - Using smaller voxel sizes")
        print("  - Checking mesh quality and compatibility")
    
    # Apply final transformation to mesh
    mesh_tgt.transform(final_transformation)
    
    return mesh_tgt, mesh_ref, final_transformation, metrics

def align_meshes(mesh_ref, mesh_tgt, target_accuracy=0.1, verbose=False):
    print_verbose("Loading meshes...", verbose)
    # print("Loading meshes...")
    pcd_ref = mesh_to_pcd(mesh_ref, num_points=10000)
    pcd_tgt = mesh_to_pcd(mesh_tgt, num_points=10000)
    
    # mesh_ref, pcd_ref = load_and_sample(ref_path, num_points=10000)  # More points for better accuracy
    # mesh_tgt, pcd_tgt = load_and_sample(tgt_path, num_points=10000)
    
    # Ensure normals are computed for point-to-plane ICP
    pcd_ref.estimate_normals()
    pcd_tgt.estimate_normals()
    
    # print("Running global alignment (Enhanced RANSAC)...")
    print_verbose("Running global alignment (Enhanced RANSAC)...", verbose)
    voxel_size = 2.0  # Smaller voxel size for better initial alignment
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    # print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    print_verbose(f"RANSAC fitness: {result_ransac.fitness:.6f}", verbose)
    
    if result_ransac.fitness < 0.1:
        print_verbose("Warning: RANSAC fitness is low. Consider adjusting parameters or checking mesh compatibility.", verbose)
        # print("Warning: RANSAC fitness is low. Consider adjusting parameters or checking mesh compatibility.")
    
    print_verbose("\nRunning multi-scale ICP refinement...", verbose)
    # print("\nRunning multi-scale ICP refinement...")
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation, verbose=verbose)
    
    # Try colored ICP if available
    print_verbose("\nAttempting colored ICP refinement...", verbose)
    result_colored = colored_icp_refinement(pcd_tgt, pcd_ref, result_icp.transformation, verbose)
    
    # Use the best result
    final_result = result_colored if result_colored and result_colored.inlier_rmse < result_icp.inlier_rmse else result_icp
    final_transformation = final_result.transformation
    
    print_verbose(f"\nFinal alignment metrics:", verbose)
    print_verbose(f"  Final fitness: {final_result.fitness:.6f}", verbose)
    print_verbose(f"  Final RMSE: {final_result.inlier_rmse:.6f}mm", verbose)
    
    # Compute detailed alignment metrics
    print_verbose("\nComputing detailed alignment metrics...", verbose)
    pcd_tgt_copy = copy.deepcopy(pcd_tgt)
    pcd_tgt_copy.transform(final_transformation)
    metrics = compute_alignment_metrics(pcd_tgt_copy, pcd_ref)
    
    print_verbose(f"Alignment Quality Report:", verbose)
    print_verbose(f"  Mean distance: {metrics['mean_distance']:.4f}mm", verbose)
    print_verbose(f"  Median distance: {metrics['median_distance']:.4f}mm", verbose)
    print_verbose(f"  Std deviation: {metrics['std_distance']:.4f}mm", verbose)
    print_verbose(f"  95th percentile: {metrics['percentile_95']:.4f}mm", verbose)
    print_verbose(f"  99th percentile: {metrics['percentile_99']:.4f}mm", verbose)
    print_verbose(f"  Max distance: {metrics['max_distance']:.4f}mm", verbose)
    print_verbose(f"  Valid points: {metrics['num_valid_points']}/{metrics['total_points']}", verbose)
    
    # Check if target accuracy is achieved
    if metrics['percentile_95'] <= target_accuracy:
        print_verbose(f"✓ Target accuracy of {target_accuracy}mm achieved!", verbose)
    else:
        print_verbose(f"⚠ Target accuracy of {target_accuracy}mm not achieved. Consider:", verbose)
        print_verbose("  - Increasing point cloud density", verbose)
        print_verbose("  - Using smaller voxel sizes", verbose)
        print_verbose("  - Checking mesh quality and compatibility", verbose)
    
    # Apply final transformation to mesh
    mesh_tgt.transform(final_transformation)
    
    return mesh_tgt, mesh_ref, final_transformation, metrics

def align_mesh_to_pcd(pcd_ref, mesh_tgt, target_accuracy=0.1):
    print("Loading meshes...")
    # pcd_ref = mesh_to_pcd(mesh_ref, num_points=10000)
    pcd_tgt = mesh_to_pcd(mesh_tgt, num_points=10000)
    
    # mesh_ref, pcd_ref = load_and_sample(ref_path, num_points=10000)  # More points for better accuracy
    # mesh_tgt, pcd_tgt = load_and_sample(tgt_path, num_points=10000)
    
    # Ensure normals are computed for point-to-plane ICP
    pcd_ref.estimate_normals()
    pcd_tgt.estimate_normals()
    
    print("Running global alignment (Enhanced RANSAC)...")
    voxel_size = 2.0  # Smaller voxel size for better initial alignment
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
    if result_ransac.fitness < 0.1:
        print("Warning: RANSAC fitness is low. Consider adjusting parameters or checking mesh compatibility.")
    
    print("\nRunning multi-scale ICP refinement...")
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation)
    
    # Try colored ICP if available
    print("\nAttempting colored ICP refinement...")
    result_colored = colored_icp_refinement(pcd_tgt, pcd_ref, result_icp.transformation)
    
    # Use the best result
    final_result = result_colored if result_colored and result_colored.inlier_rmse < result_icp.inlier_rmse else result_icp
    final_transformation = final_result.transformation
    
    print(f"\nFinal alignment metrics:")
    print(f"  Final fitness: {final_result.fitness:.6f}")
    print(f"  Final RMSE: {final_result.inlier_rmse:.6f}mm")
    
    # Compute detailed alignment metrics
    print("\nComputing detailed alignment metrics...")
    pcd_tgt_copy = copy.deepcopy(pcd_tgt)
    pcd_tgt_copy.transform(final_transformation)
    metrics = compute_alignment_metrics(pcd_tgt_copy, pcd_ref)
    
    print(f"Alignment Quality Report:")
    print(f"  Mean distance: {metrics['mean_distance']:.4f}mm")
    print(f"  Median distance: {metrics['median_distance']:.4f}mm")
    print(f"  Std deviation: {metrics['std_distance']:.4f}mm")
    print(f"  95th percentile: {metrics['percentile_95']:.4f}mm")
    print(f"  99th percentile: {metrics['percentile_99']:.4f}mm")
    print(f"  Max distance: {metrics['max_distance']:.4f}mm")
    print(f"  Valid points: {metrics['num_valid_points']}/{metrics['total_points']}")
    
    # Check if target accuracy is achieved
    if metrics['percentile_95'] <= target_accuracy:
        print(f"✓ Target accuracy of {target_accuracy}mm achieved!")
    else:
        print(f"⚠ Target accuracy of {target_accuracy}mm not achieved. Consider:")
        print("  - Increasing point cloud density")
        print("  - Using smaller voxel sizes")
        print("  - Checking mesh quality and compatibility")
    
    # Apply final transformation to mesh
    mesh_tgt.transform(final_transformation)
    
    return mesh_tgt, pcd_ref, final_transformation, metrics

def align_meshes_ultra_precise_path(ref_path, tgt_path):
    """
    Ultra-precise alignment using the finest settings
    """
    print("Loading meshes with high density sampling...")
    mesh_ref, _ = load_and_sample(ref_path, num_points=1)  # We'll create our own sampling
    mesh_tgt, _ = load_and_sample(tgt_path, num_points=1)
    
    # High-density sampling for maximum accuracy
    pcd_ref = mesh_ref.sample_points_uniformly(number_of_points=20000)
    pcd_tgt = mesh_tgt.sample_points_uniformly(number_of_points=20000)
    
    # Ensure high-quality normals
    pcd_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd_tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    print("Running ultra-precise global alignment...")
    voxel_size = 1.0  # Very fine voxel size
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
    print("\nRunning ultra-precise multi-scale ICP...")
    # Ultra-fine thresholds for sub-0.1mm accuracy
    ultra_fine_thresholds = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation, ultra_fine_thresholds)
    
    # Final metrics
    print(f"\nUltra-precise alignment complete:")
    print(f"  Final RMSE: {result_icp.inlier_rmse:.6f}mm")
    
    mesh_tgt.transform(result_icp.transformation)
    
    return mesh_tgt, mesh_ref, result_icp.transformation

def align_meshes_ultra_precise(mesh_ref, mesh_tgt, verbose=False):
    """
    Ultra-precise alignment using the finest settings
    """
    print_verbose("Loading meshes with high density sampling...", verbose)
    # mesh_ref, _ = load_and_sample(ref_path, num_points=1)  # We'll create our own sampling
    # mesh_tgt, _ = load_and_sample(tgt_path, num_points=1)
    
    # High-density sampling for maximum accuracy
    pcd_ref = mesh_ref.sample_points_uniformly(number_of_points=20000)
    pcd_tgt = mesh_tgt.sample_points_uniformly(number_of_points=20000)
    
    # Ensure high-quality normals
    pcd_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd_tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    print_verbose("Running ultra-precise global alignment...", verbose)
    voxel_size = 1.0  # Very fine voxel size
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print_verbose(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
    print_verbose("\nRunning ultra-precise multi-scale ICP...", verbose)
    # Ultra-fine thresholds for sub-0.1mm accuracy
    ultra_fine_thresholds = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation, ultra_fine_thresholds, verbose=verbose)
    
    # Final metrics
    print_verbose("\nUltra-precise alignment complete:", verbose)
    print_verbose(f"  Final RMSE: {result_icp.inlier_rmse:.6f}mm", verbose)
    
    mesh_tgt.transform(result_icp.transformation)
    
    return mesh_tgt, mesh_ref, result_icp.transformation

# def align_mesh_to_pcd_ultra_precise(pcd_ref, mesh_tgt):
#     """
#     Ultra-precise alignment using the finest settings
#     """
#     print("Loading meshes with high density sampling...")
#     # mesh_ref, _ = load_and_sample(ref_path, num_points=1)  # We'll create our own sampling
#     # mesh_tgt, _ = load_and_sample(tgt_path, num_points=1)
    
#     # High-density sampling for maximum accuracy
#     # pcd_ref = mesh_ref.sample_points_uniformly(number_of_points=20000)
#     pcd_tgt = mesh_tgt.sample_points_uniformly(number_of_points=20000)
    
#     # Ensure high-quality normals
#     pcd_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
#     pcd_tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
#     print("Running ultra-precise global alignment...")
#     voxel_size = 1.0  # Very fine voxel size
#     result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
#     print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
#     print("\nRunning ultra-precise multi-scale ICP...")
#     # Ultra-fine thresholds for sub-0.1mm accuracy
#     ultra_fine_thresholds = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
#     result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation, ultra_fine_thresholds)
    
#     # Final metrics
#     print(f"\nUltra-precise alignment complete:")
#     print(f"  Final RMSE: {result_icp.inlier_rmse:.6f}mm")
    
#     mesh_tgt.transform(result_icp.transformation)
    
#     return mesh_tgt, pcd_ref, result_icp.transformation

def align_mesh_to_pcd_ultra_precise(pcd_ref, mesh_tgt):
    """
    Ultra-precise alignment using the finest settings
    """
    print("Loading meshes with high density sampling...")
    pcd_tgt = mesh_tgt.sample_points_uniformly(number_of_points=20000)
    
    pcd_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd_tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    print("Running ultra-precise global alignment...")
    voxel_size = 1.0
    result_ransac = execute_global_registration(pcd_tgt, pcd_ref, voxel_size)
    print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    
    print("\nRunning ultra-precise multi-scale ICP...")
    ultra_fine_thresholds = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
    result_icp = multi_scale_icp(pcd_tgt, pcd_ref, result_ransac.transformation, ultra_fine_thresholds)

    print(f"\nUltra-precise alignment complete:")
    print(f"  Final RMSE: {result_icp.inlier_rmse:.6f}mm")

    # Compute metrics
    pcd_tgt_copy = copy.deepcopy(pcd_tgt)
    pcd_tgt_copy.transform(result_icp.transformation)
    metrics = compute_alignment_metrics(pcd_tgt_copy, pcd_ref)

    print("Alignment Quality Report (Ultra-Precise):")
    print(f"  Mean distance: {metrics['mean_distance']:.4f}mm")
    print(f"  Median distance: {metrics['median_distance']:.4f}mm")
    print(f"  Std deviation: {metrics['std_distance']:.4f}mm")
    print(f"  95th percentile: {metrics['percentile_95']:.4f}mm")
    print(f"  99th percentile: {metrics['percentile_99']:.4f}mm")
    print(f"  Max distance: {metrics['max_distance']:.4f}mm")
    print(f"  Valid points: {metrics['num_valid_points']}/{metrics['total_points']}")

    mesh_tgt.transform(result_icp.transformation)
    
    return mesh_tgt, pcd_ref, result_icp.transformation, metrics


def align_tgt_to_ref_paths(ref_path, tgt_path, output_path=None, visualize=True):
    standard=True
    
    print("=== Standard Enhanced Alignment ===")
    transformed_mesh, mesh_ref, transformation, metrics = align_meshes_path(ref_path, tgt_path, target_accuracy=0.1)
    
    # If standard alignment doesn't meet requirements, try ultra-precise
    if metrics['percentile_95'] > 0.1:
        standard=False
        print("\n=== Ultra-Precise Alignment ===")
        transformed_mesh, mesh_ref, ultra_transformation = align_meshes_ultra_precise_path(ref_path, tgt_path)
    
    if output_path:
        o3d.io.write_triangle_mesh(output_path, transformed_mesh)
        print(f"Aligned mesh saved to {output_path}")
        
    if visualize:
        mesh_ref.paint_uniform_color([1, 0.706, 0])
        transformed_mesh.paint_uniform_color([0, 0.651, 0.929])
        if standard:
            o3d.visualization.draw_geometries([mesh_ref, transformed_mesh], window_name='Standard Alignment')
        else:
            o3d.visualization.draw_geometries([mesh_ref, transformed_mesh], window_name='Precise Alignment')
            
def align_tgt_to_ref_meshes(mesh_ref, mesh_tgt, output_path=None, visualize=True, verbose=False):
    standard=True
    
    print_verbose("=== Standard Enhanced Alignment ===", verbose)
    transformed_mesh, mesh_ref, transformation, metrics = align_meshes(mesh_ref, mesh_tgt, target_accuracy=0.1, verbose=verbose)
    
    # If standard alignment doesn't meet requirements, try ultra-precise
    if metrics['percentile_95'] > 0.1:
        standard=False
        print_verbose("\n=== Ultra-Precise Alignment ===", verbose)
        transformed_mesh, mesh_ref, ultra_transformation = align_meshes_ultra_precise(mesh_ref, mesh_tgt, verbose=verbose)
    
    if output_path:
        o3d.io.write_triangle_mesh(output_path, transformed_mesh)
        print_verbose(f"Aligned mesh saved to {output_path}", verbose)
        
    if visualize:
        mesh_ref.paint_uniform_color([1, 0.706, 0])
        transformed_mesh.paint_uniform_color([0, 0.651, 0.929])
        if standard:
            o3d.visualization.draw_geometries([mesh_ref, transformed_mesh], window_name='Standard Alignment')
        else:
            o3d.visualization.draw_geometries([mesh_ref, transformed_mesh], window_name='Precise Alignment')
            
    return transformed_mesh

# def align_tgt_mesh_to_ref_pcd(pcd_ref, mesh_tgt, output_path=None, visualize=True):
#     standard=True
    
#     print("=== Standard Enhanced Alignment ===")
#     transformed_mesh, pcd_ref, transformation, metrics = align_mesh_to_pcd(pcd_ref, mesh_tgt, target_accuracy=0.1)
    
#     # If standard alignment doesn't meet requirements, try ultra-precise
#     if metrics['percentile_95'] > 0.1:
#         standard=False
#         print("\n=== Ultra-Precise Alignment ===")
#         transformed_mesh, pcd_ref, ultra_transformation = align_mesh_to_pcd_ultra_precise(pcd_ref, mesh_tgt)
    
#     if output_path:
#         o3d.io.write_triangle_mesh(output_path, transformed_mesh)
#         print(f"Aligned mesh saved to {output_path}")
        
#     if visualize:
#         pcd_ref.paint_uniform_color([1, 0.706, 0])
#         transformed_mesh.paint_uniform_color([0, 0.651, 0.929])
#         if standard:
#             o3d.visualization.draw_geometries([pcd_ref, transformed_mesh], window_name='Standard Alignment')
#         else:
#             o3d.visualization.draw_geometries([pcd_ref, transformed_mesh], window_name='Precise Alignment')
            
#     return transformed_mesh

def align_tgt_mesh_to_ref_pcd(pcd_ref, mesh_tgt, output_path=None, visualize=True):
    standard = True

    print("=== Standard Enhanced Alignment ===")
    transformed_mesh, pcd_ref, transformation, metrics = align_mesh_to_pcd(pcd_ref, mesh_tgt, target_accuracy=0.1)

    # If standard alignment doesn't meet requirements, try ultra-precise
    if metrics['percentile_95'] > 0.1:
        standard = False
        print("\n=== Ultra-Precise Alignment ===")
        transformed_mesh, pcd_ref, transformation, metrics = align_mesh_to_pcd_ultra_precise(pcd_ref, mesh_tgt)

    if output_path:
        o3d.io.write_triangle_mesh(output_path, transformed_mesh)
        print(f"Aligned mesh saved to {output_path}")

    if visualize:
        pcd_ref.paint_uniform_color([1, 0.706, 0])
        transformed_mesh.paint_uniform_color([0, 0.651, 0.929])
        window_title = 'Standard Alignment' if standard else 'Precise Alignment'
        o3d.visualization.draw_geometries([pcd_ref, transformed_mesh], window_name=window_title)

    return transformed_mesh, metrics


if __name__ == "__main__":
    directory = "E:/Fixture Scans"
    
    ref_path = directory + "/" + "scan_1_with_specimen.stl"
    tgt_path = directory + "/" + "specimen.stl"
    output_path = directory + "/" + "aligned_specimen.stl"
    # output_path = None
    visualize=True
    standard=True
    
    align_tgt_to_ref_paths(ref_path, tgt_path, output_path)
            