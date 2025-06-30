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

def align_with_pca(geometry, is_point_cloud=True):
    """
    Create a 4x4 transformation matrix that produces the same result as the manual PCA method
    
    Manual method does: (points - centroid) @ rotation_matrix
    We need to create T such that when Open3D applies it, we get the same result
    """
    if is_point_cloud:
        points = np.asarray(geometry.points)
    else:
        # For mesh, use vertices
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
    
    # The rotation matrix from the manual method
    rotation_matrix_manual = ordered_axes.T
    
    # For Open3D transform: result = R @ points + t
    # We want: result = (points - centroid) @ rotation_matrix_manual
    # Which is: result = points @ rotation_matrix_manual - centroid @ rotation_matrix_manual
    # So: R = rotation_matrix_manual.T, t = -centroid @ rotation_matrix_manual
    
    R_transform = rotation_matrix_manual.T
    t_transform = -centroid @ rotation_matrix_manual
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_transform
    T[:3, 3] = t_transform
    
    return T, rotation_matrix_manual, centroid

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

# === Apply PCA transformation using corrected method ===
T_pca, R_pca, centroid = align_with_pca(pcd, is_point_cloud=True)

# === Also create the manual alignment for comparison ===
pcd_manual, R_manual, centroid_manual = align_point_cloud_with_pca_manual(pcd)

# === Apply the transformation matrix to both geometries ===
mesh_transformed = copy.deepcopy(mesh)
mesh_transformed.transform(T_pca)

pcd_transformed = copy.deepcopy(pcd)
pcd_transformed.transform(T_pca)

# === VISUALIZATION 2: PCA aligned geometry with WCS axes ===
print("Showing PCA-aligned geometry...")
mesh_transformed.paint_uniform_color([1, 0, 0])  # Red mesh
pcd_transformed.paint_uniform_color([0, 1, 1])   # Cyan points

# Create new axes for the aligned view
aligned_bbox = mesh_transformed.get_axis_aligned_bounding_box()
aligned_axes_scale = np.linalg.norm(aligned_bbox.get_extent()) * 0.2
aligned_axes = create_coordinate_axes(aligned_axes_scale)

o3d.visualization.draw_geometries([pcd_transformed, mesh_transformed, aligned_axes],
                                  window_name="PCA-Aligned Geometry with World Coordinate System",
                                  width=1000, height=800)

# === VISUALIZATION 3: Compare manual vs transform method ===
print("Showing comparison: Manual PCA (blue) vs Transform method (cyan)...")
pcd_manual.paint_uniform_color([0, 0, 1])       # Blue - your manual method
pcd_transformed.paint_uniform_color([0, 1, 1])  # Cyan - transform method

comparison_axes = create_coordinate_axes(aligned_axes_scale)
o3d.visualization.draw_geometries([pcd_manual, pcd_transformed, comparison_axes],
                                  window_name="Comparison: Manual PCA vs Transform Method",
                                  width=1000, height=800)

# === Verification metrics ===
print("\n=== VERIFICATION METRICS ===")

# 1. Check transformation matrix properties
det_rotation = np.linalg.det(T_pca[:3, :3])
print(f"Rotation matrix determinant: {det_rotation:.8f} (should be close to 1.0)")

# 2. Compare rotation matrices
print(f"\nRotation matrix from transform method:")
print(R_pca)
print(f"\nRotation matrix from manual method:")
print(R_manual)
print(f"Are they equal? {np.allclose(R_pca, R_manual)}")

# 3. Compare centroids
original_mesh_centroid = np.mean(np.asarray(mesh.vertices), axis=0)
original_pcd_centroid = np.mean(np.asarray(pcd.points), axis=0)
transformed_mesh_centroid = np.mean(np.asarray(mesh_transformed.vertices), axis=0)
transformed_pcd_centroid = np.mean(np.asarray(pcd_transformed.points), axis=0)
manual_pcd_centroid = np.mean(np.asarray(pcd_manual.points), axis=0)

print(f"\nCentroid comparison:")
print(f"Original mesh: [{original_mesh_centroid[0]:.4f}, {original_mesh_centroid[1]:.4f}, {original_mesh_centroid[2]:.4f}]")
print(f"Original pcd:  [{original_pcd_centroid[0]:.4f}, {original_pcd_centroid[1]:.4f}, {original_pcd_centroid[2]:.4f}]")
print(f"Transform mesh: [{transformed_mesh_centroid[0]:.4f}, {transformed_mesh_centroid[1]:.4f}, {transformed_mesh_centroid[2]:.4f}]")
print(f"Transform pcd:  [{transformed_pcd_centroid[0]:.4f}, {transformed_pcd_centroid[1]:.4f}, {transformed_pcd_centroid[2]:.4f}]")
print(f"Manual pcd:     [{manual_pcd_centroid[0]:.4f}, {manual_pcd_centroid[1]:.4f}, {manual_pcd_centroid[2]:.4f}]")

# 4. Check alignment between point cloud and mesh
distances = pcd_transformed.compute_point_cloud_distance(mesh_transformed.sample_points_poisson_disk(10000))
mean_distance = np.mean(distances)
max_distance = np.max(distances)
std_distance = np.std(distances)

print(f"\nAlignment verification (transform method):")
print(f"Mean distance between aligned point cloud and mesh surface: {mean_distance:.6f}")
print(f"Max distance: {max_distance:.6f}")
print(f"Standard deviation: {std_distance:.6f}")

# 5. Check if manual and transform methods produce same result
manual_vs_transform_distances = pcd_manual.compute_point_cloud_distance(pcd_transformed)
mean_diff = np.mean(manual_vs_transform_distances)
print(f"\nDifference between manual and transform methods:")
print(f"Mean point-to-point distance: {mean_diff:.8f}")
print(f"Methods are equivalent: {mean_diff < 1e-10}")

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