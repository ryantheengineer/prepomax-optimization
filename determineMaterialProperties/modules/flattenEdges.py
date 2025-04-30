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

# Filepath references (unchanged)
input_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl"
output_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/flattened_edge_planes.stl"

# Parameters
edge_face_extent_threshold = 20.0  # distance from extreme direction point to consider
plane_fit_distance_threshold = 0.2  # for RANSAC
projection_distance_threshold = 2.0  # for applying projection
alignment_dot_threshold = 0.85  # must be roughly aligned with main axis
grid_resolution = 1.0  # spacing for occlusion grid

# Load mesh
mesh = o3d.io.read_triangle_mesh(input_path)
mesh_vertices = np.asarray(mesh.vertices)
mesh_triangles = np.asarray(mesh.triangles)
point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_vertices))

# Directions to process
directions = {
    "+X": np.array([1, 0, 0]),
    "-X": np.array([-1, 0, 0]),
    "+Y": np.array([0, 1, 0]),
    "-Y": np.array([0, -1, 0]),
}

# Utility: Project 3D points onto plane
def project_points_onto_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    distances = points @ normal + d
    return points - np.outer(distances, normal)

# Utility: Return only non-occluded points from candidate set
def filter_occluded_points(candidate_points, plane_normal, grid_resolution=1.0):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Create orthonormal basis: [u, v, n]
    if abs(plane_normal[2]) < 0.9:
        arbitrary = np.array([0, 0, 1])
    else:
        arbitrary = np.array([1, 0, 0])
    u = np.cross(plane_normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)

    projected_2d = np.stack([candidate_points @ u, candidate_points @ v], axis=1)
    depths = candidate_points @ plane_normal

    min_uv = projected_2d.min(axis=0)
    max_uv = projected_2d.max(axis=0)
    grid_size = np.ceil((max_uv - min_uv) / grid_resolution).astype(int)

    bin_indices = np.floor((projected_2d - min_uv) / grid_resolution).astype(int)
    bin_dict = {}

    for i, (ix, iy) in enumerate(bin_indices):
        key = (ix, iy)
        d = depths[i]
        if key not in bin_dict or d < bin_dict[key][0]:
            bin_dict[key] = (d, i)

    visible_indices = [val[1] for val in bin_dict.values()]
    return candidate_points[visible_indices], visible_indices

# Initialize modified mesh
modified_mesh = copy.deepcopy(mesh)
modified_vertices = np.asarray(modified_mesh.vertices)

# Process each direction
for direction_name, axis_vector in directions.items():
    # Step 1: Select extreme points near the outermost in this direction
    dots = mesh_vertices @ axis_vector
    max_val = np.max(dots)
    keep_indices = np.where((max_val - dots) < edge_face_extent_threshold)[0]
    candidate_points = mesh_vertices[keep_indices]
    candidate_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidate_points))

    print(f"\nDirection {direction_name} — Candidate points: {len(candidate_points)}")

    # Step 2: RANSAC to fit a plane
    plane_model, inliers = candidate_pcd.segment_plane(
        distance_threshold=plane_fit_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    dot_alignment = np.abs(np.dot(normal, axis_vector))
    print(f"  Plane normal: {normal}, alignment with axis: {dot_alignment:.3f}")
    if dot_alignment < alignment_dot_threshold:
        print(f"  Skipping plane: not aligned with expected axis.")
        continue

    inlier_points = candidate_points[inliers]
    print(f"  RANSAC inliers: {len(inlier_points)}")

    # Step 3: Apply occlusion-aware filtering
    visible_points, visible_indices = filter_occluded_points(candidate_points, normal, grid_resolution=grid_resolution)
    print(f"  Projecting {len(visible_points)} visible points onto plane.")

    # Step 4: Project those visible points into the plane
    projected_points = project_points_onto_plane(visible_points, plane_model)

    # Step 5: Map back to mesh and update vertices
    mesh_tree = o3d.geometry.KDTreeFlann(point_cloud)
    modified_indices = set()
    for orig in visible_points:
        _, idx, _ = mesh_tree.search_knn_vector_3d(orig, 1)
        modified_indices.add(idx[0])
    modified_indices = list(modified_indices)

    # Replace vertices with projected positions
    for i, idx in enumerate(modified_indices):
        modified_vertices[idx] = projected_points[i]
    modified_mesh.vertices = o3d.utility.Vector3dVector(modified_vertices)

    # Step 6: Visualization
    plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_points))
    plane_pcd.paint_uniform_color([1, 0, 0])
    vis_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(visible_points))
    vis_points.paint_uniform_color([0, 1, 0])
    proj_lines = []
    all_proj_points = []
    for a, b in zip(visible_points, projected_points):
        i = len(all_proj_points)
        all_proj_points.extend([a, b])
        proj_lines.append([i, i+1])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_proj_points),
        lines=o3d.utility.Vector2iVector(proj_lines)
    )
    line_set.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([mesh, plane_pcd, vis_points, line_set])

# # Finalize
# modified_mesh.compute_vertex_normals()
# o3d.io.write_triangle_mesh(output_path, modified_mesh)
# print(f"\n✅ Mesh exported to: {output_path}")

# Track updated vertex positions
print("\nUpdating vertex positions")
updated_vertex_map = {}  # index in original -> new position

for orig, proj in zip(visible_points, projected_points):
    _, idx, _ = mesh_tree.search_knn_vector_3d(orig, 1)
    updated_vertex_map[idx[0]] = proj

# Construct new point cloud from updated vertices
print("\nConstructing new point cloud")
new_vertices = []
for i, v in enumerate(mesh_vertices):
    if i in updated_vertex_map:
        new_vertices.append(updated_vertex_map[i])
    else:
        new_vertices.append(v)
new_vertices = np.array(new_vertices)

new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_vertices)

# Visualize the updated point cloud for quality
o3d.visualization.draw_geometries([new_pcd], window_name="New Point Cloud")

# # Estimate and orient normals
# new_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
# new_pcd.orient_normals_consistent_tangent_plane(k=10)

# # Downsample the point cloud for faster performance
# print("\nDownsampling the point cloud using voxel downsampling")
# voxel_size = 1.0  # Adjust voxel size based on your model scale
# downsampled_pcd = new_pcd.voxel_down_sample(voxel_size)

# Estimate and orient normals for the downsampled point cloud
print("\nEstimating and orienting normals")
new_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
new_pcd.orient_normals_consistent_tangent_plane(k=10)
# downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
# downsampled_pcd.orient_normals_consistent_tangent_plane(k=10)

# Ball Pivoting with smaller radii and downsampled point cloud
print("\nBall pivoting...")
radii = o3d.utility.DoubleVector([2.0, 3.0, 4.0, 10.0, 15.0])  # Try with a single radius
mesh_out = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(new_pcd, radii)
# mesh_out = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downsampled_pcd, radii)

# Compute triangle normals
print("\nComputing triangle normals")
mesh_out.compute_triangle_normals()

# Save the mesh
print("\nSaving mesh")
success = o3d.io.write_triangle_mesh(output_path, mesh_out)
if success:
    print(f"\n✅ Mesh exported to: {output_path}")
else:
    print(f"\n❌ Export failed. Output path: {output_path}")

# # Ball Pivoting reconstruction
# radii = o3d.utility.DoubleVector([2.0])  # Adjust these based on feature size
# # radii = o3d.utility.DoubleVector([2.0, 3.0, 4.0])  # Adjust these based on feature size
# mesh_out = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(new_pcd, radii)

# # Optional: Simplify or clean up if needed
# # mesh_out = mesh_out.simplify_quadric_decimation(50000)

# # Compute triangle normals
# mesh_out.compute_triangle_normals()

# # Save the mesh
# success = o3d.io.write_triangle_mesh(output_path, mesh_out)
# if success:
#     print(f"\nMesh exported to: {output_path}")
# else:
#     print(f"\nExport failed. Output path: {output_path}")






















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