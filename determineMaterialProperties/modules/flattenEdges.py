# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:50:34 2025

@author: Ryan.Larson
"""

# "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/aligned_scans/Merge_03_mesh_aligned_positive.stl"
# C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl

import open3d as o3d
import numpy as np

# Load and sample point cloud
mesh = o3d.io.read_triangle_mesh("C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl")
pcd = mesh.sample_points_uniformly(number_of_points=20000)

# Axis-aligned bounding box
aabb = pcd.get_axis_aligned_bounding_box()
min_bound = aabb.min_bound
max_bound = aabb.max_bound

# Slab thickness near each end in mm
slab_thickness = 10

# Bounding boxes for ±X slabs
crop_xmin = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[min_bound[0], min_bound[1], min_bound[2]],
    max_bound=[min_bound[0] + slab_thickness, max_bound[1], max_bound[2]]
)

crop_xmax = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[max_bound[0] - slab_thickness, min_bound[1], min_bound[2]],
    max_bound=[max_bound[0], max_bound[1], max_bound[2]]
)

# Bounding boxes for ±Y slabs
crop_ymin = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[min_bound[0], min_bound[1], min_bound[2]],
    max_bound=[max_bound[0], min_bound[1] + slab_thickness, max_bound[2]]
)

crop_ymax = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[min_bound[0], max_bound[1] - slab_thickness, min_bound[2]],
    max_bound=[max_bound[0], max_bound[1], max_bound[2]]
)

# Crop slabs from the full point cloud
pcd_xmin = pcd.crop(crop_xmin)
pcd_xmax = pcd.crop(crop_xmax)
pcd_ymin = pcd.crop(crop_ymin)
pcd_ymax = pcd.crop(crop_ymax)

# Plane fitting function with axis alignment filter
def fit_plane_and_color(pcd, color, axis, alignment_threshold=0.9, label=""):
    if len(pcd.points) < 3:
        print(f"Not enough points in {label} slice")
        return None

    plane_model, inliers = pcd.segment_plane(distance_threshold=1.0, ransac_n=3, num_iterations=1000)
    normal = np.array(plane_model[:3])
    normal_unit = normal / np.linalg.norm(normal)

    alignment = np.abs(np.dot(normal_unit, axis))
    if alignment < alignment_threshold:
        print(f"{label} plane not aligned with axis: {normal_unit}")
        return None

    print(f"{label} plane found: normal = {normal_unit}, alignment = {alignment}")
    plane = pcd.select_by_index(inliers)
    plane.paint_uniform_color(color)
    return plane

# Detect and color planes
plane_xmin = fit_plane_and_color(pcd_xmin, [1, 0, 0], axis=[1, 0, 0], label="Negative X")
plane_xmax = fit_plane_and_color(pcd_xmax, [0, 1, 0], axis=[1, 0, 0], label="Positive X")
plane_ymin = fit_plane_and_color(pcd_ymin, [0, 0, 1], axis=[0, 1, 0], label="Negative Y")
plane_ymax = fit_plane_and_color(pcd_ymax, [1, 1, 0], axis=[0, 1, 0], label="Positive Y")

# Visualize detected planes only
planes_to_show = [p for p in [plane_xmin, plane_xmax, plane_ymin, plane_ymax] if p is not None]

if planes_to_show:
    o3d.visualization.draw_geometries(planes_to_show)
else:
    print("No valid planes detected.")










# import open3d as o3d
# import numpy as np

# # Load and sample point cloud from the mesh
# mesh = o3d.io.read_triangle_mesh("C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl")
# pcd = mesh.sample_points_uniformly(number_of_points=20000)
# pcd = pcd.voxel_down_sample(voxel_size=0.5)

# # print("Mesh bounding box:", mesh.get_axis_aligned_bounding_box())

# distinct_colors = [
#     [1, 0, 0],     # Red
#     [0, 1, 0],     # Green
#     [0, 0, 1],     # Blue
#     [1, 1, 0],     # Yellow
#     [1, 0, 1],     # Magenta
#     [0, 1, 1]      # Cyan
# ]

# # Parameters
# distance_threshold = 1.5
# normal_filter_axis = np.array([0, 0, 1])  # Y-axis (adjust based on your alignment)
# alignment_threshold = 0.9  # cosine of angle to treat as "top/bottom"

# # RANSAC with filtering
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

# # Add coordinate frame and remaining cloud for context
# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
# remaining.paint_uniform_color([0.6, 0.6, 0.6])
# remaining_down = remaining.voxel_down_sample(voxel_size=0.5)

# # Visualize
# o3d.visualization.draw_geometries(colored_planes + [remaining_down, coord_frame])