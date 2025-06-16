import numpy as np
import trimesh
import open3d as o3d

def trimesh_to_o3d(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    
    # Uniformly downsample to ~20k points
    original_count = len(pcd.points)
    target_count = 20000
    pcd = downsample_point_cloud_uniform(pcd, target_count)
    # pcd.paint_uniform_color([0.8, 0.8, 0.8])
    print(f"Downsampled from {original_count} to {len(pcd.points)} points.")
    return pcd

def pca_align(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    R = Vt.T
    aligned = centered @ R
    return aligned, R, centroid

def extract_region(points, x_center, window):
    half_window = window / 2
    mask = (points[:, 0] >= (x_center - half_window)) & (points[:, 0] <= (x_center + half_window))
    return points[mask]


def fit_cylinder_ransac(points, iterations=500, distance_threshold=0.5, min_inliers=100):
    best_inliers = []
    best_model = None

    n_points = len(points)
    if n_points < 2:
        return None, None, None

    for _ in range(iterations):
        # Randomly sample 2 points to define cylinder axis direction
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]

        axis = p2 - p1
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            continue
        axis /= axis_len

        # Project all points to axis line (get distances along axis)
        diffs = points - p1
        projections = np.dot(diffs, axis)

        # Closest points on axis line
        closest = p1 + np.outer(projections, axis)

        # Radial distances from axis
        radial_vecs = points - closest
        radial_dist = np.linalg.norm(radial_vecs, axis=1)

        # Guess radius as median of radial distances of all points
        radius_guess = np.median(radial_dist)

        # Find inliers: points with radial distance close to radius_guess within threshold
        inliers = np.where(np.abs(radial_dist - radius_guess) < distance_threshold)[0]

        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_model = (p1, axis, radius_guess)

    if best_model is None:
        return None, None, None

    # Refine cylinder parameters using inliers
    inlier_points = points[best_inliers]
    p1, axis, radius = best_model

    # Recompute axis using PCA on inliers to improve orientation
    centroid = inlier_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_points - centroid)
    axis = Vt[0]
    if np.dot(axis, best_model[1]) < 0:
        axis = -axis  # Keep same direction

    # Recompute radius as mean radial distance
    diffs = inlier_points - centroid
    proj_len = np.dot(diffs, axis)
    proj_points = diffs - np.outer(proj_len, axis)
    radius = np.mean(np.linalg.norm(proj_points, axis=1))

    return centroid, axis, radius

def create_trimesh_cylinder(center, axis, radius, height=20.0, sections=64, color=(1,0,0,0.6)):
    # Create cylinder along Z axis
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    # Calculate rotation matrix from Z axis to cylinder axis
    z_axis = np.array([0,0,1])
    axis = axis / np.linalg.norm(axis)

    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v)**2))

    cyl.apply_transform(trimesh.transformations.rotation_matrix(angle=0, direction=[1,0,0]))
    cyl.apply_transform(np.vstack((np.hstack((R, np.zeros((3,1)))), [0,0,0,1])))
    # Translate cylinder so its center aligns (cylinder origin is at center along height)
    cyl.apply_translation(center - axis * height / 2)

    # cyl.visual.face_colors = trimesh.visual.color.hex_to_rgba(trimesh.visual.color.color_to_hex(color))

    return cyl

def downsample_point_cloud_uniform(pcd, target_count):
    pts = np.asarray(pcd.points)
    if len(pts) <= target_count:
        return pcd
    indices = np.random.choice(len(pts), size=target_count, replace=False)
    down_pcd = pcd.select_by_index(indices)
    return down_pcd

def visualize_point_clouds(full_points, window_points, window_center):
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_points)
    full_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray full cloud

    window_pcd = o3d.geometry.PointCloud()
    window_pcd.points = o3d.utility.Vector3dVector(window_points)
    window_pcd.paint_uniform_color([1, 0, 0])  # Red window points

    print(f"Showing windowed points near x = {window_center} (red) over full cloud (gray)")
    o3d.visualization.draw_geometries([full_pcd, window_pcd])

def main(mesh_path, expected_x, window=15.0):
    mesh = trimesh.load(mesh_path)
    pcd = trimesh_to_o3d(mesh)

    original_count = len(pcd.points)
    target_count = 50000
    pcd = downsample_point_cloud_uniform(pcd, target_count)
    print(f"Downsampled from {original_count} to {len(pcd.points)} points.")

    points = np.asarray(pcd.points)
    # aligned_points, R, centroid = pca_align(points)

    cylinders = []
    for x in expected_x:
        region_points = extract_region(points, x_center=x, window=window)
        # region_points = extract_region(aligned_points, x_center=x, window=window)
        if len(region_points) < 200:
            print(f"Skipping window at x={x}, not enough points")
            continue

        # Visual validation of windowing:
        visualize_point_clouds(points, region_points, x)
        # visualize_point_clouds(aligned_points, region_points, x)

        c, axis, r = fit_cylinder_ransac(region_points)
        if c is None:
            print(f"No cylinder found near x={x}")
            continue

        # c_world = c @ R.T + centroid
        # axis_world = axis @ R.T

        print(f"Detected cylinder near x={x}: center={c}, axis={axis}, radius={r:.3f}")
        # print(f"Detected cylinder near x={x}: center={c_world}, axis={axis_world}, radius={r:.3f}")

        cyl = create_trimesh_cylinder(c, axis, r)
        # cyl = create_trimesh_cylinder(c_world, axis_world, r)
        cylinders.append(cyl)

    scene = trimesh.Scene([mesh] + cylinders)
    scene.show()




if __name__ == "__main__":
    # Example usage
    directory = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization"
    mesh_path = directory + "/5 - Flexural Test Meshes/" + "X2_smoothed_Test1.stl"
    # mesh_path = "path/to/your/mesh.stl"  # change this
    expected_x_positions = [-33, 0, 43]  # example expected positions after PCA alignment
    main(mesh_path, expected_x_positions, window=40.0)


# directory = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization"
# mesh_path = directory + "/5 - Flexural Test Meshes/" + "X2_smoothed_Test1.stl"

# # Load and convert mesh
# mesh = trimesh.load(mesh_path)
# pcd = trimesh_to_o3d(mesh)

# # Uniformly downsample to ~20k points
# original_count = len(pcd.points)
# target_count = 20000
# pcd = downsample_point_cloud_uniform(pcd, target_count)
# pcd.paint_uniform_color([0.8, 0.8, 0.8])
# print(f"Downsampled from {original_count} to {len(pcd.points)} points.")

# # Align with PCA
# pcd_aligned, R, mean = pca_align(pcd)

# # Detect cylinders
# expected_x = [-33, 0, 42]
# detected = []
# for x in expected_x:
#     sub = extract_region(pcd_aligned, x_center=x, x_tol=20)
#     if len(sub.points) < 100:
#         continue
#     axis, center, radius = fit_axis_radius(sub)
#     center_world = center @ R.T + mean
#     axis_world = axis @ R.T
#     detected.append((center_world, axis_world, radius))

# # Create cylinders as point clouds
# cylinder_pcds = [
#     create_cylinder_point_cloud(center=c, axis=a, radius=r, height=10.0)
#     for c, a, r in detected
# ]

# # Visualize all as point clouds
# o3d.visualization.draw_geometries([pcd] + cylinder_pcds)
