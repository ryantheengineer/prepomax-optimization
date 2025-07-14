# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:04:51 2025

@author: Ryan.Larson
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import copy
import pyswarms as ps
from scipy.spatial import cKDTree

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0,0,0])

# ---------- PCA ALIGNMENT ----------
def pca_align_mesh(mesh):
    print("[INFO] Running PCA alignment...")
    points = np.asarray(mesh.vertices)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    R_align = eigvecs.T
    mesh_copy = mesh.translate(-centroid, relative=False)
    mesh_copy.rotate(R_align, center=(0, 0, 0))
    return mesh_copy, R_align, centroid

def align_mesh_with_pca(mesh):
    print("[INFO] Running PCA alignment...")

    # Convert to numpy array
    vertices = np.asarray(mesh.vertices)

    # Compute centroid and center the mesh
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered)
    axes = pca.components_

    # Sort axes by explained variance (descending)
    order = np.argsort(pca.explained_variance_)[::-1]
    ordered_axes = axes[order]

    # Target basis: identity matrix (align to X, Y, Z)
    target_axes = np.eye(3)

    # Compute rotation matrix
    rotation_matrix = ordered_axes.T @ target_axes

    # Rotate centered vertices
    aligned_vertices = centered @ rotation_matrix

    # Create a new mesh
    mesh_aligned = o3d.geometry.TriangleMesh(mesh)  # deep copy
    mesh_aligned.vertices = o3d.utility.Vector3dVector(aligned_vertices)
    mesh_aligned.compute_vertex_normals()

    return mesh_aligned

# ---------- PREPARE MESHES ------------
def rotate_mesh_about_axis(mesh, angle_degrees, axis='z', center=(0, 0, 0)):
    """
    Rotates a mesh by a given angle around a specified axis.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The mesh to rotate.
        angle_degrees (float): The angle of rotation in degrees.
        axis (str or array-like): The axis to rotate around ('x', 'y', 'z' or a 3-element array).
        center (tuple or np.ndarray): The center of rotation (default: origin).

    Returns:
        o3d.geometry.TriangleMesh: A rotated copy of the mesh.
    """
    # Handle axis input
    if isinstance(axis, str):
        axis = axis.lower()
        axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Use 'x', 'y', 'z', or a 3-element vector.")
        axis_vector = axis_map[axis]
    else:
        axis_vector = np.asarray(axis)

    # Normalize axis vector
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Generate rotation matrix
    rotation_vector = np.radians(angle_degrees) * axis_vector
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    # Copy mesh and apply rotation
    rotated_mesh = o3d.geometry.TriangleMesh(mesh)
    rotated_mesh.rotate(rotation_matrix, center=center)
    rotated_mesh.compute_vertex_normals()

    return rotated_mesh

def filter_target_faces_by_normal(mesh, direction=np.array([0, 0, 1]), angle_threshold_degrees=15):
    """
    Filters the mesh to retain only faces whose normals are within angle_threshold_degrees of the given direction.
    Args:
        mesh: open3d.geometry.TriangleMesh
        direction: target normal direction (default is +Y)
        angle_threshold_degrees: angle threshold in degrees (default 15)
    Returns:
        A new TriangleMesh with only the filtered faces.
    """
    mesh.compute_triangle_normals()
    direction = direction / np.linalg.norm(direction)
    angle_threshold_cos = np.cos(np.radians(angle_threshold_degrees))

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.triangle_normals)

    dot_products = normals @ direction
    keep_face_mask = np.abs(dot_products) >= angle_threshold_cos
    keep_indices = np.where(keep_face_mask)[0]

    # Extract the triangles to keep and rebuild mesh
    new_triangles = triangles[keep_indices]
    used_vertices = np.unique(new_triangles)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    remapped_triangles = np.vectorize(index_map.get)(new_triangles)
    new_vertices = vertices[used_vertices]

    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    filtered_mesh.compute_vertex_normals()

    return filtered_mesh

def filter_points_near_z_peak(pcd, peak_z, tolerance=0.5):
    """
    Filter points in a point cloud to retain only those near the peak Z level.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        peak_z (float): Z-coordinate of the detected peak.
        tolerance (float): Maximum distance from peak_z to retain points.

    Returns:
        filtered_pcd (o3d.geometry.PointCloud): Filtered point cloud.
    """
    points = np.asarray(pcd.points)
    mask = np.abs(points[:, 2] - peak_z) <= tolerance

    if not np.any(mask):
        print("No points found within tolerance of peak_z.")
        return o3d.geometry.PointCloud()

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])

    # Retain colors or normals if they exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return filtered_pcd

# ---------- DISTANCE METRICS ----------
def compute_rmse(source_mesh, target_pcd):
    source_points = np.asarray(source_mesh.vertices)
    dists = target_pcd.compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points)))
    return np.sqrt(np.mean(np.square(dists)))

def compute_median_error(source_mesh, target_pcd):
    source_points = np.asarray(source_mesh.vertices)
    dists = target_pcd.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    )
    return np.median(dists)

def compute_pct_within_tol(source_mesh, target_pcd, tol=0.1):
    source_points = np.asarray(source_mesh.vertices)
    dists = target_pcd.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    )
    dists = np.asarray(dists)
    count_tol = np.sum(dists < tol)
    pct_tol = count_tol / len(dists)
    return pct_tol
    

# ---------- TRANSLATION SCAN ----------
def scan_translation(source_mesh, target_pcd, axis=2, values=np.linspace(-50, 50, 100), visualize=False):
    axis_name = ['X', 'Y', 'Z'][axis]
    print(f"[INFO] Scanning translation along {axis_name}-axis...")
    best_rmse = np.inf
    best_median_error = np.inf
    best_pct = 0.0
    best_mesh = source_mesh
    best_value = 0.0
    tol = 0.1

    for val in values:
        trial = source_mesh.translate((val if axis==0 else 0, val if axis==1 else 0, val if axis==2 else 0), relative=False)
        pct_tol = compute_pct_within_tol(trial, target_pcd, tol=tol)
        if pct_tol > best_pct:
            best_pct = pct_tol
            best_mesh = trial
            best_value = val
            print(f"[INFO] Current best % within tol={tol}: {best_pct:.4f} at translation {best_value:.4f}")
            
        # median_error = compute_median_error(trial, target_pcd)
        # if median_error < best_median_error:
        #     best_median_error = median_error
        #     best_mesh = trial
        #     best_value = val
        #     print(f"[INFO] Current best median error {best_median_error:.4f} at translation {best_value:.4f}")
        # rmse = compute_rmse(trial, target_pcd)
        # if rmse < best_rmse:
        #     best_rmse = rmse
        #     best_mesh = trial
        #     best_value = val
        #     print(f"[INFO] Current best RMSE {best_rmse:.4f} at translation {best_value:.4f}")

    print(f"[RESULT] Best translation along {axis_name} = {best_value:.2f}, % of points within tol={tol}: {best_pct:.4f}")
    best_error = best_pct
    # print(f"[RESULT] Best translation along {axis_name} = {best_value:.2f}, Median Error = {best_median_error:.4f}")
    # best_error = best_median_error
    # print(f"[RESULT] Best translation along {axis_name} = {best_value:.2f}, RMSE = {best_rmse:.4f}")
    # best_error = best_rmse
    
    if visualize:
        o3d.visualization.draw_geometries([best_mesh, target_pcd],
            window_name=f"Best Fit - Translate {axis_name}")

    return best_mesh, best_error, best_value

def find_highest_z_peak(pcd, bin_width=0.25, min_count=20):
    """
    Find the highest Z-coordinate peak in a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        bin_width (float): Width of histogram bins along Z.
        min_count (int): Minimum number of points to consider a bin a peak.
    
    Returns:
        peak_z_center (float): The Z value at the center of the highest peak bin.
        count (int): The number of points in that bin.
    """
    z_vals = np.asarray(pcd.points)[:, 2]
    z_min, z_max = np.min(z_vals), np.max(z_vals)

    bins = np.arange(z_min, z_max + bin_width, bin_width)
    hist, edges = np.histogram(z_vals, bins=bins)

    # Only consider bins with a meaningful number of points
    peak_idx = None
    for i in reversed(range(len(hist))):  # Search from highest Z downward
        if hist[i] >= min_count:
            peak_idx = i
            break

    if peak_idx is None:
        print("No significant peak found.")
        return None, 0

    peak_z_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    return peak_z_center, hist[peak_idx]

def apply_translation(source_mesh, target_pcd, distance, axis=2, visualize=False):
    axis_name = ['X', 'Y', 'Z'][axis]
    print(f"[INFO] Applying translation along {axis_name}-axis...")
    translated_source_mesh = source_mesh.translate((distance if axis==0 else 0, distance if axis==1 else 0, distance if axis==2 else 0), relative=False)
    
    if visualize:
        o3d.visualization.draw_geometries([translated_source_mesh, target_pcd],
            window_name=f"Translate {axis_name}")
    return translated_source_mesh

# ---------- ORIENTATION FLIPS ----------
def generate_candidate_orientations(mesh, visualize=False, target_pcd=None):
    print("[INFO] Generating candidate 180° flips...")
    identities = {
        "Original": np.eye(3),
        "Flip Z": R.from_euler('z', 180, degrees=True).as_matrix(),
        "Flip X": R.from_euler('x', 180, degrees=True).as_matrix(),
        "Flip Z+X": R.from_euler('zx', [180, 180], degrees=True).as_matrix()
    }

    candidates = []
    for name, rot in identities.items():
        m = copy.deepcopy(mesh)
        m.rotate(rot, center=(0, 0, 0))
        print(f"  - Candidate: {name}")
        if visualize and target_pcd:
            o3d.visualization.draw_geometries([
                m.paint_uniform_color([1, 0, 0]),
                target_pcd.paint_uniform_color([0, 1, 0]),
                axis
            ], window_name=f"Candidate Orientation: {name}")
        candidates.append((m, name))
    return candidates

# ---------- FEATURE FITTING WITH RANSAC -----------
def align_source_mesh_to_target_pcd(source_mesh, target_pcd, voxel_size=2.0):
    """
    Aligns a source mesh to a target point cloud using multi-scale ICP only.
    
    Args:
        source_mesh (o3d.geometry.TriangleMesh): The mesh to align.
        target_pcd (o3d.geometry.PointCloud): The fixed target point cloud (assumed to be in place).
        voxel_size (float): Base voxel size for downsampling and threshold tuning.
        
    Returns:
        aligned_mesh (o3d.geometry.TriangleMesh): The transformed source mesh.
        result_icp (RegistrationResult): The final ICP registration result.
    """
    # Step 1: Convert source mesh to point cloud
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=20000)
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    # Step 2: Multi-scale ICP
    thresholds = [voxel_size * 2, voxel_size, voxel_size * 0.5]
    iterations = [50, 30, 14]

    current_transformation = np.eye(4)
    for i, (thresh, iter_count) in enumerate(zip(thresholds, iterations)):
        source_down = source_pcd.voxel_down_sample(voxel_size * (2.0 / (i + 1)))
        source_down.estimate_normals()
        target_down = target_pcd.voxel_down_sample(voxel_size * (2.0 / (i + 1)))
        target_down.estimate_normals()

        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, thresh, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter_count)
        )
        current_transformation = result_icp.transformation

    # Step 3: Apply final transformation to source mesh
    aligned_mesh = copy.deepcopy(source_mesh)
    aligned_mesh.transform(current_transformation)

    return aligned_mesh, result_icp


# def align_source_mesh_to_target_pcd(source_mesh, target_pcd, voxel_size=2.0):
#     """
#     Align a source mesh to a target point cloud using RANSAC-based global registration.
#     The transformation is applied to the original (dense) source mesh.

#     Args:
#         source_mesh (o3d.geometry.TriangleMesh): The mesh to align.
#         target_pcd (o3d.geometry.PointCloud): The fixed target point cloud (not downsampled).
#         voxel_size (float): Voxel size for downsampling the source and computing features.

#     Returns:
#         transformed_mesh (o3d.geometry.TriangleMesh): The source mesh transformed to match the target.
#         result (RegistrationResult): The Open3D registration result object (contains transformation).
#     """
#     # Sample source mesh to point cloud and downsample it
#     source_pcd = source_mesh.sample_points_uniformly(number_of_points=20000)
#     source_pcd.estimate_normals()

#     src_down = source_pcd.voxel_down_sample(voxel_size)
#     src_down.estimate_normals()

#     # Compute FPFH for source
#     src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         src_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 7, max_nn=150)
#     )

#     # Preprocess target point cloud (dense, no downsampling)
#     target_pcd.estimate_normals()

#     tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         target_pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 7, max_nn=150)
#     )

#     # Run RANSAC global registration
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         src_down, target_pcd, src_fpfh, tgt_fpfh,
#         mutual_filter=False,
#         max_correspondence_distance=voxel_size * 1.5,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=4,
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2.0),
#         ],
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
#     )

#     # Apply the transformation to the full-resolution mesh
#     transformed_mesh = copy.deepcopy(source_mesh.translate((0, 0, 0), relative=False))
#     transformed_mesh.transform(result.transformation)

#     return transformed_mesh, result

# ---------- LOCAL OPTIMIZATION ----------
def constrained_optimization(source_mesh, target_pcd, max_translation=20.0, max_rotation_deg=5.0, visualize=False):
    print("[INFO] Starting local constrained optimization...")
    source_points = np.asarray(source_mesh.vertices)

    def transform_mesh(params):
        tx, ty, tz, rx, ry, rz = params
        R_mat = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        transformed = (R_mat @ source_points.T).T + np.array([tx, ty, tz])
        return transformed

    def objective(params):
        transformed_pts = transform_mesh(params)
        temp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_pts))
        dists = target_pcd.compute_point_cloud_distance(temp_pcd)
        return np.sqrt(np.mean(np.square(dists)))

    bounds = [(-max_translation, max_translation)] * 3 + [(-max_rotation_deg, max_rotation_deg)] * 3
    x0 = np.zeros(6)
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    print(f"[RESULT] Optimization RMSE: {result.fun:.4f}")
    print(f"[INFO] Optimal Parameters: Trans {result.x[:3]}, Rot (deg) {result.x[3:]}")

    final_transformed = transform_mesh(result.x)
    final_mesh = o3d.geometry.TriangleMesh()
    final_mesh.vertices = o3d.utility.Vector3dVector(final_transformed)
    final_mesh.triangles = source_mesh.triangles
    final_mesh.compute_vertex_normals()

    if visualize:
        o3d.visualization.draw_geometries([
            final_mesh.paint_uniform_color([1, 0, 0]),
            target_pcd.paint_uniform_color([0, 1, 0]),
            axis
        ], window_name="Final Optimized Alignment")

    return final_mesh, result


def transform_mesh(mesh, angles_xyz, translation):
    """Applies a 6D transform (rotation angles in radians + translation) to a mesh."""
    rx, ry, rz = angles_xyz
    R = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.transform(T)
    return mesh_copy


def compute_fit_error(mesh, target_pcd, num_sample_points=50000):
    """Samples points from mesh and computes average distance to nearest target point."""
    sampled = mesh.sample_points_uniformly(number_of_points=num_sample_points)
    source_points = np.asarray(sampled.points)
    target_tree = cKDTree(np.asarray(target_pcd.points))
    dists, _ = target_tree.query(source_points, k=1)
    tol = 10
    
    pct_close = np.sum(dists < tol) / len(dists)
    
    return 1 - pct_close
    
    # return np.median(dists ** 2)
    # return np.mean(dists ** 2)


def create_objective_function(source_mesh, target_pcd):
    """Returns a function that takes a 6D vector and outputs a single error value."""
    def objective(x):
        # x: (num_particles, 6)
        results = []
        for row in x:
            rx, ry, rz, tx, ty, tz = row
            mesh_transformed = transform_mesh(source_mesh, [rx, ry, rz], [tx, ty, tz])
            error = compute_fit_error(mesh_transformed, target_pcd)
            results.append(error)
        return np.array(results)
    return objective


def align_mesh_with_pso(source_mesh, target_pcd, voxel_size=2.0, num_particles=30, iters=100):
    """
    Uses PSO to align source_mesh to target_pcd by minimizing surface-to-cloud error.
    
    Returns:
        aligned_mesh: Transformed mesh in best-fit pose
        best_cost: Final objective value
        best_pose: [rx, ry, rz, tx, ty, tz]
    """
    bounds = (
        np.array([-np.pi, -np.radians(10), -np.pi, -10, -10, -10]),  # min bounds
        np.array([ np.pi,  np.radians(10),  np.pi,  10,  10,  10])   # max bounds
    )

    objective_fn = create_objective_function(source_mesh, target_pcd)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=6,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.8},
        bounds=bounds
    )

    best_cost, best_pose = optimizer.optimize(objective_fn, iters)

    rx, ry, rz, tx, ty, tz = best_pose
    aligned_mesh = transform_mesh(source_mesh, [rx, ry, rz], [tx, ty, tz])

    return aligned_mesh, best_cost, best_pose


def align_dense_to_sparse(source_mesh, target_mesh, visualize=True):
    print("[INFO] === Starting Full Alignment ===")

    source_aligned = align_mesh_with_pca(source_mesh)
    target_aligned = align_mesh_with_pca(target_mesh)
    
    # Rotate target mesh 90 degrees about +X
    target_aligned = rotate_mesh_about_axis(target_aligned, 90.0, axis='x', center=(0, 0, 0))
    
    target_aligned = filter_target_faces_by_normal(target_aligned, direction=np.array([0, 0, 1]), angle_threshold_degrees=15)
    
    target_pcd = target_aligned.sample_points_uniformly(number_of_points=100000)
    
    # source_aligned, _, _ = pca_align_mesh(source_mesh)
    # target_aligned, _, _ = pca_align_mesh(target_mesh)

    # if visualize:
    #     o3d.visualization.draw_geometries([
    #         source_aligned.paint_uniform_color([1, 0, 0]),
    #         target_aligned.paint_uniform_color([0, 1, 0]),
    #         axis
    #     ], window_name="After PCA Alignment")
    
    # Find the likely Z position and move the specimen there
    peak_z, count = find_highest_z_peak(target_pcd)
    filtered_target_pcd = filter_points_near_z_peak(target_pcd, peak_z, tolerance=0.5)
    z_aligned = apply_translation(source_aligned, filtered_target_pcd, peak_z, axis=2, visualize=False)
    
    best_mesh, best_cost, best_pose = align_mesh_with_pso(z_aligned, filtered_target_pcd, voxel_size=0.5, num_particles=100, iters=10)
    
    # z_aligned, _, _ = scan_translation(source_aligned, target_pcd, axis=2, values=np.linspace(0, 40, 100), visualize=visualize)
    # xz_aligned, _, _ = scan_translation(z_aligned, target_pcd, axis=0, values=np.linspace(-30, 30, 100), visualize=visualize)

    # candidates = generate_candidate_orientations(z_aligned, visualize=False)
    # best_mesh = None
    # best_fitness = 0.0
    # for mesh, label in candidates:
    #     transformed_source_mesh, result = align_source_mesh_to_target_pcd(mesh, filtered_target_pcd, voxel_size=2.0)
    #     fitness = result.fitness
    #     if fitness > best_fitness:
    #         best_fitness = fitness
    #         best_mesh = transformed_source_mesh
    #         print(f'Found better candidate: {label}, Fitness={best_fitness}')
            
    

    # best_candidate = None
    # best_pct = 0.0
    # best_label = ""

    # for mesh, label in candidates:
    #     cz_aligned, _, _ = scan_translation(mesh, target_pcd, axis=2, values=np.linspace(-2, 2, 50))
    #     cxz_aligned, pct, _ = scan_translation(cz_aligned, target_pcd, axis=0, values=np.linspace(-20, 20, 30))
    #     print(f"[CANDIDATE] {label}: % close to 0 error = {pct:.4f}")
    #     if pct > best_pct:
    #         best_pct = pct
    #         best_candidate = cxz_aligned
    #         best_label = label

    # print(f"[INFO] Best orientation: {best_label}")

    if visualize:
        o3d.visualization.draw_geometries([
            best_mesh.paint_uniform_color([1, 0, 0]),
            target_pcd.paint_uniform_color([0, 1, 0]),
            axis
        ], window_name=f"RANSAC Fitting Result")

    # final_mesh, result = constrained_optimization(best_candidate, target_pcd, visualize=visualize)

    print("[INFO] === Alignment Complete ===")
    return best_mesh


if __name__ == "__main__":
    directory = "E:/Fixture Scans"
    source_filepath = directory + "/" + "specimen.stl"
    target_filepath = directory + "/" + "scan_1_partial.stl"    
    
    source_mesh = o3d.io.read_triangle_mesh(source_filepath)
    target_mesh = o3d.io.read_triangle_mesh(target_filepath)
    
    align_dense_to_sparse(source_mesh, target_mesh)
    # aligned_mesh = align_dense_to_sparse(source_mesh, target_mesh)
    
    # Visualize result
    # o3d.visualization.draw_geometries([aligned_mesh.paint_uniform_color([1, 0, 0]), target_mesh.paint_uniform_color([0, 1, 0])])
