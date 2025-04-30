# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:50:34 2025

@author: Ryan.Larson
"""

# "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/aligned_scans/Merge_03_mesh_aligned_positive.stl"
# C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl

import open3d as o3d
import numpy as np
import copy

# === Settings ===
mesh_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl"
output_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/flattened_edge_planes.stl"

# Parameters
edge_face_extent_threshold = 20.0   # mm — how far from extreme point to consider
plane_fit_distance_threshold = 5.0  # mm — for RANSAC plane fitting
projection_distance_threshold = 1.0  # mm — max distance to project onto plane
alignment_cosine_threshold = 0.85    # dot product between direction and normal

directions = {
    "+X": np.array([1, 0, 0]),
    "-X": np.array([-1, 0, 0]),
    "+Y": np.array([0, 1, 0]),
    "-Y": np.array([0, -1, 0]),
}

# === Load Mesh and Extract All Vertices as Point Cloud ===
mesh = o3d.io.read_triangle_mesh(mesh_path)
vertices = np.asarray(mesh.vertices)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

# === Utilities ===
def find_extreme_face_points(points, direction, extent_threshold):
    proj = points @ direction
    extreme_value = np.max(proj)
    mask = (extreme_value - proj) < extent_threshold
    return np.where(mask)[0]

def fit_plane_and_validate(points, expected_dir, distance_thresh, align_thresh):
    subset_pcd = o3d.geometry.PointCloud()
    subset_pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = subset_pcd.segment_plane(
        distance_threshold=distance_thresh,
        ransac_n=3,
        num_iterations=1000,
    )
    normal = np.array(model[:3])
    normal /= np.linalg.norm(normal)
    if abs(np.dot(normal, expected_dir)) < align_thresh:
        return None, None
    return model, inliers

def project_points_onto_plane(points, model):
    a, b, c, d = model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    distances = points @ normal + d
    return points - np.outer(distances, normal)

def draw_debug_scene(title, geometries):
    print(f"Showing: {title}")
    o3d.visualization.draw_geometries(geometries, window_name=title)

# === Modify Mesh Vertices ===
modified_mesh = copy.deepcopy(mesh)
all_vertices = np.asarray(modified_mesh.vertices)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

for label, dir_vec in directions.items():
    print(f"\n--- Processing direction: {label} ---")
    
    # Step 1: Identify candidate extreme points
    candidate_indices = find_extreme_face_points(all_vertices, dir_vec, edge_face_extent_threshold)
    if len(candidate_indices) < 10:
        print(f"Not enough points in {label} direction.")
        continue
    candidate_points = all_vertices[candidate_indices]
    
    subset_pcd = o3d.geometry.PointCloud()
    subset_pcd.points = o3d.utility.Vector3dVector(candidate_points)
    subset_pcd.paint_uniform_color([1, 0.7, 0])  # Orange
    draw_debug_scene(f"{label} - Candidate Extreme Points", [mesh, subset_pcd])

    # Step 2: Fit RANSAC plane
    plane_model, inliers = fit_plane_and_validate(candidate_points, dir_vec, plane_fit_distance_threshold, alignment_cosine_threshold)
    if plane_model is None:
        print(f"Plane in {label} direction rejected due to misalignment.")
        continue
    inlier_indices = np.array(candidate_indices)[inliers]
    inlier_points = all_vertices[inlier_indices]

    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
    inlier_pcd.paint_uniform_color([0, 1, 0])  # Green
    draw_debug_scene(f"{label} - RANSAC Inliers", [mesh, inlier_pcd])

    # Step 3: Filter points near the plane
    a, b, c, d = plane_model
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    distances = inlier_points @ normal + d
    close_mask = np.abs(distances) < projection_distance_threshold
    final_indices = inlier_indices[close_mask]
    final_points = all_vertices[final_indices]

    if len(final_indices) == 0:
        print(f"No points close enough to plane in {label} direction.")
        continue

    print(f"Using {len(final_indices)} vertices from {label} direction.")

    # Step 4: Project and visualize
    projected_points = project_points_onto_plane(final_points, plane_model)

    # Create lines for projection
    line_points = []
    lines = []
    for i, (p_orig, p_proj) in enumerate(zip(final_points, projected_points)):
        line_points.extend([p_orig, p_proj])
        lines.append([2*i, 2*i + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([1, 0, 0])  # Red lines

    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    projected_pcd.paint_uniform_color([0, 0, 1])  # Blue points

    draw_debug_scene(f"{label} - Projection Vectors", [mesh, inlier_pcd, projected_pcd, line_set])

    # Step 5: Replace vertex positions
    all_vertices[final_indices] = projected_points

# === Finalize and Save ===
modified_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
modified_mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(output_path, modified_mesh)
print(f"\n✅ Mesh saved to: {output_path}")





















# import open3d as o3d
# import numpy as np
# import copy

# # Load the original mesh
# mesh = o3d.io.read_triangle_mesh("C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl")
# pcd = mesh.sample_points_uniformly(number_of_points=20000)

# # Parameters
# distance_threshold = 0.2
# normal_filter_axis = np.array([0, 0, 1])  # Z-axis (adjust based on your alignment)
# alignment_threshold = 0.9  # cosine of angle to treat as "top/bottom"
# distinct_colors = [
#     [1, 0, 0],     # Red
#     [0, 1, 0],     # Green
#     [0, 0, 1],     # Blue
#     [1, 1, 0],     # Yellow
#     [1, 0, 1],     # Magenta
#     [0, 1, 1]      # Cyan
# ]

# # RANSAC with filtering (to detect planes)
# plane_models = []
# plane_inliers = []
# colored_planes = []
# remaining = pcd
# i = 0
# for _ in range(10):  # Try more iterations to get 4 usable edge planes
#     plane_model, inliers = remaining.segment_plane(
#         distance_threshold=distance_threshold,
#         ransac_n=3,
#         num_iterations=1000,
#     )

#     normal = np.array(plane_model[:3])
#     normal_unit = normal / np.linalg.norm(normal)
#     dot = np.abs(np.dot(normal_unit, normal_filter_axis))  # how aligned with "up" axis?

#     if dot < alignment_threshold:
#         # Keep this plane — it's not too aligned with the top/bottom direction
#         plane_models.append(plane_model)
#         plane = remaining.select_by_index(inliers)
#         color = distinct_colors[i % len(distinct_colors)]
#         plane.paint_uniform_color(color)
#         colored_planes.append(plane)
#         plane_inliers.append(inliers)
#         print(f"Accepted plane normal: {normal_unit}")
#         i += 1
#     else:
#         print(f"Skipped top/bottom-aligned plane: {normal_unit}")

#     remaining = remaining.select_by_index(inliers, invert=True)

#     if len(plane_models) >= 4:
#         break

# # --- Use the plane models to project the edge faces ---
# # Create a new mesh to modify (copy the original mesh data)
# modified_mesh = o3d.geometry.TriangleMesh()
# modified_mesh.vertices = mesh.vertices
# modified_mesh.triangles = mesh.triangles

# # Visualize
# o3d.visualization.draw_geometries([modified_mesh])

# # --- Function to project points onto a plane ---
# def project_points_onto_plane(points, plane_model):
#     a, b, c, d = plane_model
#     normal = np.array([a, b, c])
#     normal = normal / np.linalg.norm(normal)
#     distances = points @ normal + d
#     return points - np.outer(distances, normal)

# # --- KDTree from mesh vertices ---
# mesh_vertices_array = np.asarray(mesh.vertices)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(mesh_vertices_array)
# mesh_tree = o3d.geometry.KDTreeFlann(pcd)
# # mesh_tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(mesh_vertices_array))

# # --- Modify the mesh by flattening the edge faces ---
# modified_mesh = copy.deepcopy(mesh)

# for i, inliers in enumerate(plane_inliers):
#     # Get the inlier points (from point cloud)
#     inlier_points = np.asarray(pcd.select_by_index(inliers).points)
    
#     # Find closest mesh vertex indices for those inliers
#     matched_vertex_indices = set()
#     for p in inlier_points:
#         _, idx, _ = mesh_tree.search_knn_vector_3d(p, 1)
#         matched_vertex_indices.add(idx[0])
#     matched_vertex_indices = list(matched_vertex_indices)

#     # Get original vertex positions
#     matched_vertices = mesh_vertices_array[matched_vertex_indices]

#     # Project those vertices onto the plane
#     projected_vertices = project_points_onto_plane(matched_vertices, plane_models[i])

#     # Debug visualization lines (original -> projected)
#     debug_line_points = []
#     debug_lines = []
#     for orig, proj in zip(matched_vertices, projected_vertices):
#         start_idx = len(debug_line_points)
#         debug_line_points.extend([orig, proj])
#         debug_lines.append([start_idx, start_idx + 1])
    
#     debug_line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(debug_line_points),
#         lines=o3d.utility.Vector2iVector(debug_lines)
#     )
#     debug_line_set.paint_uniform_color([1, 0, 0])  # Red lines

#     # Show the original mesh with red projection lines
#     # o3d.visualization.draw_geometries([colored_planes[i], debug_line_set])
#     o3d.visualization.draw_geometries([mesh, colored_planes[i], debug_line_set])

#     # Apply the projected positions to the modified mesh
#     modified_vertices = np.asarray(modified_mesh.vertices)
#     for j, idx in enumerate(matched_vertex_indices):
#         modified_vertices[idx] = projected_vertices[j]
#     modified_mesh.vertices = o3d.utility.Vector3dVector(modified_vertices)


# # --- Compute normals for the modified mesh ---
# modified_mesh.compute_vertex_normals()

# # --- Save the modified mesh as a watertight STL file ---
# output_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/flattened_edge_planes.stl"
# o3d.io.write_triangle_mesh(output_path, modified_mesh)
# print(f"Exported mesh to: {output_path}")