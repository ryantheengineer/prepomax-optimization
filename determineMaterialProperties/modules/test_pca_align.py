# -*- coding: utf-8 -*-
"""
Corrected PCA alignment script
Created on Mon Jun 30 09:27:38 2025
@author: Ryan.Larson
"""
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import copy

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

# === Load the mesh ===
mesh_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.orient_triangles()
mesh.compute_vertex_normals()

# === Sample a point cloud from it ===
pcd = mesh.sample_points_poisson_disk(5000)

# === Get PCA transformation from point cloud ===
T_pca, R_pca, centroid = align_with_pca(pcd, is_point_cloud=True)

# === Apply the SAME transformation to both geometries ===
# Transform the original mesh
mesh_transformed = copy.deepcopy(mesh)
mesh_transformed.transform(T_pca)

# Transform the point cloud
pcd_transformed = copy.deepcopy(pcd)
pcd_transformed.transform(T_pca)

# === Create coordinate axes for visualization ===
def create_coordinate_axes(scale=1.0):
    """Create coordinate axes (X=red, Y=green, Z=blue)"""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    return axes

# === VISUALIZATION 1: Original mesh and point cloud with WCS axes ===
print("Showing original geometry...")
mesh_original_vis = copy.deepcopy(mesh)
pcd_original_vis = copy.deepcopy(pcd)

# Color the geometries
mesh_original_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray mesh
pcd_original_vis.paint_uniform_color([0, 0, 1])         # Blue points

# Create axes scaled appropriately for the geometry
bbox = mesh.get_axis_aligned_bounding_box()
axes_scale = np.linalg.norm(bbox.get_extent()) * 0.2
original_axes = create_coordinate_axes(axes_scale)

o3d.visualization.draw_geometries([mesh_original_vis, pcd_original_vis, original_axes],
                                  window_name="Original Geometry with World Coordinate System",
                                  width=1000, height=800)

# === Apply PCA transformation using separated method ===
T_translate, T_rotate, T_combined, R_3x3, centroid = align_with_pca_separated(pcd, is_point_cloud=True)

# === Also create the manual alignment for comparison ===
pcd_manual, R_manual, centroid_manual = align_point_cloud_with_pca_manual(pcd)

# === Apply transformations in different ways ===

# Method 1: Combined transformation (should match original working version)
mesh_combined = copy.deepcopy(mesh)
mesh_combined.transform(T_combined)
pcd_combined = copy.deepcopy(pcd)
pcd_combined.transform(T_combined)

# Method 2: Separate transformations (translate first, then rotate)
mesh_separated = copy.deepcopy(mesh)
mesh_separated.transform(T_translate)  # First translate to origin
mesh_separated.transform(T_rotate)     # Then rotate
pcd_separated = copy.deepcopy(pcd)
pcd_separated.transform(T_translate)
pcd_separated.transform(T_rotate)

# Method 3: Just rotation (for downstream processes that handle translation separately)
mesh_rotation_only = copy.deepcopy(mesh)
# First manually translate to centroid
vertices = np.asarray(mesh_rotation_only.vertices)
vertices_centered = vertices - centroid
mesh_rotation_only.vertices = o3d.utility.Vector3dVector(vertices_centered)
# Then apply only rotation
mesh_rotation_only.transform(T_rotate)

# === VISUALIZATION 2: PCA aligned geometry with WCS axes ===
print("Showing PCA-aligned geometry...")
mesh_combined.paint_uniform_color([1, 0, 0])  # Red mesh
pcd_combined.paint_uniform_color([0, 1, 1])   # Cyan points

# Create new axes for the aligned view
aligned_bbox = mesh_combined.get_axis_aligned_bounding_box()
aligned_axes_scale = np.linalg.norm(aligned_bbox.get_extent()) * 0.2
aligned_axes = create_coordinate_axes(aligned_axes_scale)

o3d.visualization.draw_geometries([pcd_combined, mesh_combined, aligned_axes],
                                  window_name="PCA-Aligned Geometry with World Coordinate System",
                                  width=1000, height=800)

# === VISUALIZATION 3: Compare all methods ===
print("Showing comparison: Manual PCA (blue), Combined transform (cyan), Separated transform (red)...")
pcd_manual.paint_uniform_color([0, 0, 1])       # Blue - your manual method
pcd_combined.paint_uniform_color([0, 1, 1])     # Cyan - combined transform
pcd_separated.paint_uniform_color([1, 0, 0])    # Red - separated transforms

comparison_axes = create_coordinate_axes(aligned_axes_scale)
o3d.visualization.draw_geometries([pcd_manual, pcd_combined, pcd_separated, comparison_axes],
                                  window_name="Comparison: Manual vs Combined vs Separated Transforms",
                                  width=1000, height=800)

# === Verification metrics ===
print("\n=== VERIFICATION METRICS ===")

# 1. Check transformation matrices
print("Transformation matrices:")
print(f"Translation matrix T_translate:\n{T_translate}")
print(f"\nRotation matrix T_rotate:\n{T_rotate}")
print(f"\nCombined matrix T_combined:\n{T_combined}")
print(f"\n3x3 Rotation matrix for downstream processes:\n{R_3x3}")

# 2. Verify matrix properties
det_rotation_3x3 = np.linalg.det(R_3x3)
det_rotation_4x4 = np.linalg.det(T_rotate[:3, :3])
print(f"\n3x3 rotation matrix determinant: {det_rotation_3x3:.8f}")
print(f"4x4 rotation matrix determinant: {det_rotation_4x4:.8f}")
print(f"Both should be close to 1.0 for proper rotations")

# 3. Compare all methods
methods = [
    ("Manual", pcd_manual),
    ("Combined", pcd_combined), 
    ("Separated", pcd_separated)
]

print(f"\nCentroid comparison:")
for name, pcd_method in methods:
    centroid_method = np.mean(np.asarray(pcd_method.points), axis=0)
    print(f"{name:10s}: [{centroid_method[0]:.6f}, {centroid_method[1]:.6f}, {centroid_method[2]:.6f}]")

# 4. Check if all methods produce identical results
print(f"\nMethod equivalence checks:")
combined_vs_manual = pcd_combined.compute_point_cloud_distance(pcd_manual)
separated_vs_manual = pcd_separated.compute_point_cloud_distance(pcd_manual)
separated_vs_combined = pcd_separated.compute_point_cloud_distance(pcd_combined)

print(f"Combined vs Manual mean distance: {np.mean(combined_vs_manual):.10f}")
print(f"Separated vs Manual mean distance: {np.mean(separated_vs_manual):.10f}")
print(f"Separated vs Combined mean distance: {np.mean(separated_vs_combined):.10f}")
print(f"All methods equivalent: {np.mean(combined_vs_manual) < 1e-10 and np.mean(separated_vs_manual) < 1e-10}")

# 5. Show what your downstream processes would use
print(f"\n=== FOR DOWNSTREAM PROCESSES ===")
print(f"Centroid to translate to origin: {centroid}")
print(f"3x3 Rotation matrix to use:\n{R_3x3}")
print(f"Or 4x4 rotation-only matrix:\n{T_rotate}")

print(f"\nTo replicate the PCA alignment in your downstream process:")
print(f"1. Translate points by subtracting centroid: points - centroid")
print(f"2. Apply rotation: (points - centroid) @ R_3x3")
print(f"   OR apply 4x4 rotation matrix to already-centered geometry")

# 5. Show PCA explained variance ratios and axis alignment
pca_temp = PCA(n_components=3)
points_temp = np.asarray(pcd.points)
centroid_temp = np.mean(points_temp, axis=0)
pca_temp.fit(points_temp - centroid_temp)
print(f"\nPCA explained variance ratios:")
print(f"1st PC (now X-axis): {pca_temp.explained_variance_ratio_[0]:.4f}")
print(f"2nd PC (now Y-axis): {pca_temp.explained_variance_ratio_[1]:.4f}")  
print(f"3rd PC (now Z-axis): {pca_temp.explained_variance_ratio_[2]:.4f}")
print(f"Total variance explained: {np.sum(pca_temp.explained_variance_ratio_):.4f}")

# 6. Verify the alignment by checking the spread along each axis
transformed_points = np.asarray(pcd_transformed.points)
x_spread = np.max(transformed_points[:, 0]) - np.min(transformed_points[:, 0])
y_spread = np.max(transformed_points[:, 1]) - np.min(transformed_points[:, 1])
z_spread = np.max(transformed_points[:, 2]) - np.min(transformed_points[:, 2])
print(f"\nSpread along aligned axes:")
print(f"X-axis spread (1st PC): {x_spread:.4f}")
print(f"Y-axis spread (2nd PC): {y_spread:.4f}")
print(f"Z-axis spread (3rd PC): {z_spread:.4f}")
print(f"Ratio X:Y:Z = {x_spread/z_spread:.2f}:{y_spread/z_spread:.2f}:1.00")

print("\n=== Visualizations complete ===")
print("Close each visualization window to proceed to the next one.")