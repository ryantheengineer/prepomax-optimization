import open3d as o3d
import numpy as np
# import random
# from scipy.spatial.transform import Rotation as R
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar, minimize
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import pyswarms as ps
# import copy
# from itertools import combinations
import time

def load_mesh_as_point_cloud(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

def align_point_cloud_with_pca(pcd):
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

    # Align principal components with +X, +Y, +Z
    target_axes = np.eye(3)
    rotation_matrix = ordered_axes.T @ target_axes

    # Apply rotation
    rotated_points = centered_points @ rotation_matrix

    # Create new point cloud
    pcd_pca = o3d.geometry.PointCloud()
    pcd_pca.points = o3d.utility.Vector3dVector(rotated_points)

    # Copy over colors if they exist
    if pcd.has_colors():
        pcd_pca.colors = pcd.colors
    if pcd.has_normals():
        # Recompute normals since they may no longer be valid
        pcd_pca.estimate_normals()
        
    # o3d.visualization.draw_geometries([pcd_pca], window_name="PCA Point Cloud")
    
    return pcd_pca

def create_plane_mesh(plane_model, inlier_cloud, plane_size=20.0, color=None):
    # Create a square plane oriented by the plane normal
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Get centroid of inlier points
    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)

    # Build local coordinate system (normal + two orthogonal in-plane axes)
    z = normal
    if np.allclose(z, [0, 0, 1]):
        x = np.array([1, 0, 0])
    else:
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
    y = np.cross(z, x)

    # Build the 4 corners of the plane
    hw = plane_size / 2.0
    corners = [
        centroid + hw * x + hw * y,
        centroid - hw * x + hw * y,
        centroid - hw * x - hw * y,
        centroid + hw * x - hw * y,
    ]
    corners = np.array(corners)

    # Create mesh from corners
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color if color else np.random.rand(3))
    return mesh

def filter_duplicate_planes_by_alignment(planes, axis=np.array([1, 0, 0]), angle_thresh_deg=5.0, dist_thresh=1.0):
    axis = axis / np.linalg.norm(axis)
    angle_thresh_rad = np.radians(angle_thresh_deg)
    cos_thresh = np.cos(angle_thresh_rad)

    planes = list(planes)  # Ensure it's indexable
    used = set()
    final_planes = []
    retained_indices = []

    for i in range(len(planes)):
        if i in used:
            continue

        a1, b1, c1, d1 = planes[i]
        n1 = np.array([a1, b1, c1])
        n1 /= np.linalg.norm(n1)
        group = [i]

        for j in range(i + 1, len(planes)):
            if j in used:
                continue

            a2, b2, c2, d2 = planes[j]
            n2 = np.array([a2, b2, c2])
            n2 /= np.linalg.norm(n2)

            # Check angle between normals
            dot_product = np.dot(n1, n2)
            if abs(dot_product) >= cos_thresh:
                # Check distance between offsets
                if abs(d1 - d2) < dist_thresh:
                    group.append(j)

        # Keep the best-aligned plane in the group
        best_idx = max(
            group,
            key=lambda k: abs(np.dot(np.array(planes[k][:3]) / np.linalg.norm(planes[k][:3]), axis))
        )
        final_planes.append(planes[best_idx])
        retained_indices.append(best_idx)
        used.update(group)

    return final_planes, retained_indices

def normalize_plane(plane):
    normal = plane[:3]
    d = plane[3]
    norm = np.linalg.norm(normal)
    return normal / norm, d / norm

def plane_similarity(plane1, plane2):
    # Normalize both
    n1, d1 = normalize_plane(plane1)
    n2, d2 = normalize_plane(plane2)

    # Check both directions (n2 or -n2)
    dot_same = np.dot(n1, n2)
    dot_flipped = np.dot(n1, -n2)
    if abs(dot_same) > abs(dot_flipped):
        normal_diff = np.linalg.norm(n1 - n2)
        d_diff = abs(d1 - d2)
    else:
        normal_diff = np.linalg.norm(n1 + n2)
        d_diff = abs(d1 + d2)

    return normal_diff + d_diff  # Lower = more similar

def filter_duplicate_planes(planes, target_axis):
    planes = np.array(planes)
    n = len(planes)
    used = [False] * n
    keep = []

    for i in range(n):
        if used[i]:
            continue
        current_group = [i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            sim = plane_similarity(planes[i], planes[j])
            # We find the closest neighbor based on similarity
            if sim < 0.05:  # Very tight, not arbitrary — adjust if needed
                current_group.append(j)

        # Select best-aligned with target_axis
        best_idx = max(
            current_group,
            key=lambda idx: abs(np.dot(planes[idx][:3] / np.linalg.norm(planes[idx][:3]), target_axis))
        )
        keep.append(planes[best_idx])
        for idx in current_group:
            used[idx] = True

    return keep

def detect_planes(base_pcd, target_axis=[1, 0, 0], angle_deg=3, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    # Align the point cloud with PCA
    pcd = align_point_cloud_with_pca(base_pcd)
    
    # Rotate the point cloud about X so the greatest variation aligns with Z
    rotation_angle = np.radians(90)
    axis_angle = np.array([rotation_angle, 0, 0])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(rotation_matrix, center=(0,0,0))    
    
    target_axis = np.array(target_axis)
    # x_axis = np.array([1, 0, 0])
    angle_rad = np.radians(angle_deg)
    cos_angle_thresh = np.cos(angle_rad)
    planes = []
    plane_meshes = []

    pcd_copy = pcd.select_by_index([], invert=True)  # clone for modification
    
    inlier_clouds = []

    while np.asarray(pcd_copy.points).shape[0] >= ransac_n:
        plane_model, inliers = pcd_copy.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        if len(inliers) == 0:
            break

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        # Check if the normal is within ±angle_deg of the X axis (either direction)
        dot = np.abs(np.dot(normal, target_axis))
        if dot < cos_angle_thresh:
            # Not close enough to X axis — skip
            pcd_copy = pcd_copy.select_by_index(inliers, invert=True)
            continue
        
        inlier_cloud = pcd_copy.select_by_index(inliers)
        inlier_clouds.append(inlier_cloud)
        
        # Generate plane mesh and save it
        plane_mesh = create_plane_mesh(plane_model, inlier_cloud, plane_size=50.0)
        plane_meshes.append(plane_mesh)
        planes.append(plane_model)

        # Remove inliers and continue
        pcd_copy = pcd_copy.select_by_index(inliers, invert=True)
    
    filtered_planes, retained_idxs = filter_duplicate_planes_by_alignment(
        planes, axis=target_axis, angle_thresh_deg=5.0, dist_thresh=1.0
    )
    
    filtered_inlier_clouds = [inlier_clouds[i] for i in retained_idxs]
    
    print(f"\nPlanes found for target_axis {target_axis}:")
    if len(filtered_planes) < len(planes):
        print(f"Kept {len(filtered_planes)}/{len(planes)} planes")
    
    for plane in filtered_planes:
        print(plane)
    time.sleep(2)
    
    # Combine all for visualization
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis] + [plane_meshes[i] for i in retained_idxs])
    
    filtered_plane_meshes = [plane_meshes[i] for i in retained_idxs]
    return pcd, filtered_planes, retained_idxs, filtered_plane_meshes, filtered_inlier_clouds

def detect_fixture_planes(base_pcd, target_axes):
    keep_planes = []
    keep_plane_meshes = []
    keep_inlier_clouds = []
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    for target_axis in target_axes:
        max_retries = 5
        success = False
    
        for attempt in range(max_retries):
            try:
                pcd, filtered_planes, retained_idxs, plane_meshes, filtered_inlier_clouds = detect_planes(
                    base_pcd, target_axis=target_axis)
    
                prev_len = len(filtered_planes)
    
                tmp_planes = []
                tmp_meshes = []
                tmp_clouds = []
    
                if target_axis == [0, 0, 1]:
                    base_plane = max(filtered_planes, key=lambda x: x[3])
                    base_idx = filtered_planes.index(base_plane)
                    # original_idx = retained_idxs[base_idx]
    
                    tmp_planes.append(base_plane)
                    tmp_meshes.append(plane_meshes[base_idx])
                    tmp_clouds.append(filtered_inlier_clouds[base_idx])
    
                elif target_axis == [1, 0, 0]:
                    if len(filtered_planes) > 4:
                        filtered_planes = filter_duplicate_planes(filtered_planes, target_axis)
                        print(f'Further filtered from {prev_len} to {len(filtered_planes)} planes')
    
                    if len(filtered_planes) == 4:
                        for i, plane in enumerate(filtered_planes):
                            tmp_planes.append(plane)
                            tmp_meshes.append(plane_meshes[retained_idxs[i]])
                            tmp_clouds.append(filtered_inlier_clouds[retained_idxs[i]])
                    else:
                        raise ValueError(f"Expected 4 planes detected, but found {len(filtered_planes)}")
    
                else:
                    for i, plane in enumerate(filtered_planes):
                        tmp_planes.append(plane)
                        tmp_meshes.append(plane_meshes[retained_idxs[i]])
                        tmp_clouds.append(filtered_inlier_clouds[retained_idxs[i]])
    
                # All good, commit results
                keep_planes.extend(tmp_planes)
                keep_plane_meshes.extend(tmp_meshes)
                keep_inlier_clouds.extend(tmp_clouds)
    
                success = True
                break
    
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Failed with error: {e}")
    
        if not success:
            print(f"Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")

        
    time.sleep(2)
    
    # Combine all for visualization
    o3d.visualization.draw_geometries([pcd, axis] + keep_plane_meshes)
    # o3d.visualization.draw_geometries([pcd, axis] + [plane_meshes[i] for i in retained_idxs])
    
    return keep_planes, keep_inlier_clouds, pcd
            

def normalize_plane_append(plane):
    n = plane[:3]
    d = plane[3]
    norm = np.linalg.norm(n)
    return np.append(n / norm, d / norm)

def total_residual_loss(flat_params, inlier_clouds, plane_counts):
    num_planes = len(flat_params) // 4
    planes = [flat_params[i*4:(i+1)*4] for i in range(num_planes)]

    # Map each plane to its cloud index
    cloud_assignments = []
    for cloud_idx, count in enumerate(plane_counts):
        cloud_assignments.extend([cloud_idx] * count)

    loss = 0.0
    for i, plane in enumerate(planes):
        cloud = inlier_clouds[cloud_assignments[i]]
        n = plane[:3]
        n /= np.linalg.norm(n)
        d = plane[3]
        points = np.asarray(cloud.points)
        distances = points @ n + d
        loss += np.sum(distances ** 2)
    return loss

def constraint_orthogonal(plane1, plane2):
    n1 = plane1[:3] / np.linalg.norm(plane1[:3])
    n2 = plane2[:3] / np.linalg.norm(plane2[:3])
    return np.dot(n1, n2)

def constraint_unit_norm(plane):
    return np.linalg.norm(plane[:3]) - 1

def check_plane_point_residuals(planes, inlier_clouds, verbose=True, threshold=0.1):
    """
    For each plane and associated inlier cloud, compute the residual distance of each point to the plane.

    Args:
        planes (list of [a, b, c, d]): Plane coefficients.
        inlier_clouds (list of o3d.geometry.PointCloud): Point clouds associated with each plane.
        verbose (bool): If True, print mean and max residuals.
        threshold (float): Flag any planes with mean residuals exceeding this value.

    Returns:
        residual_stats (list of dict): Mean and max residuals per plane.
    """
    residual_stats = []

    for i, (plane, cloud) in enumerate(zip(planes, inlier_clouds)):
        n = np.array(plane[:3])
        d = plane[3]
        n = n / np.linalg.norm(n)
        points = np.asarray(cloud.points)

        distances = points @ n + d  # signed distances
        mean_res = np.mean(np.abs(distances))
        max_res = np.max(np.abs(distances))

        residual_stats.append({
            "plane_index": i,
            "mean_residual": mean_res,
            "max_residual": max_res,
            "num_points": len(points)
        })

        if verbose:
            status = "✅" if mean_res <= threshold else "⚠️"
            print(f"[{status}] Plane {i}: mean = {mean_res:.4f}, max = {max_res:.4f}, points = {len(points)}")

    return residual_stats

# def optimize_parallel_planes(plane1, plane2, cloud1, cloud2):
#     """
#     Optimize two planes to be parallel and best fit their respective point clouds.
#     Returns optimized (plane1, plane2) definitions.
#     """

#     def loss(params):
#         n = params[:3]
#         n = n / np.linalg.norm(n)
#         d0 = params[3]
#         d1 = d0 + params[4]

#         pts1 = np.asarray(cloud1.points)
#         pts2 = np.asarray(cloud2.points)

#         res1 = pts1 @ n + d0
#         res2 = pts2 @ n + d1

#         return np.sum(res1**2) + np.sum(res2**2)

#     def unit_norm_constraint(x):
#         return np.linalg.norm(x[:3]) - 1

#     initial_n = plane1[:3] / np.linalg.norm(plane1[:3])
#     initial_d0 = plane1[3]
#     initial_d1 = plane2[3]
#     initial_offset = initial_d1 - initial_d0

#     x0 = np.concatenate([initial_n, [initial_d0, initial_offset]])

#     cons = [{'type': 'eq', 'fun': unit_norm_constraint}]

#     result = minimize(
#         fun=loss,
#         x0=x0,
#         constraints=cons,
#         method='SLSQP',
#         options={'maxiter': 300, 'disp': True}
#     )

#     if not result.success:
#         print("Optimization failed:", result.message)

#     # Extract optimized values
#     n_opt = result.x[:3] / np.linalg.norm(result.x[:3])
#     d0_opt = result.x[3]
#     d1_opt = d0_opt + result.x[4]

#     plane1_opt = np.append(n_opt, d0_opt)
#     plane2_opt = np.append(n_opt, d1_opt)

#     return plane1_opt, plane2_opt

def optimize_planes_with_fixed_angle(plane1, plane2, cloud1, cloud2, target_angle_deg=60):
    """
    Optimize two planes to fit their point clouds,
    and constrain their normals to be separated by target_angle_deg degrees.
    Returns optimized (plane1, plane2) definitions.
    """

    target_cos = np.cos(np.radians(target_angle_deg))

    def loss(params):
        n1 = params[:3]
        n1 = n1 / np.linalg.norm(n1)
        n2 = params[3:6]
        n2 = n2 / np.linalg.norm(n2)
        d1 = params[6]
        d2 = params[7]

        pts1 = np.asarray(cloud1.points)
        pts2 = np.asarray(cloud2.points)

        res1 = pts1 @ n1 + d1
        res2 = pts2 @ n2 + d2

        return np.sum(res1**2) + np.sum(res2**2)

    def unit_norm_constraint1(params):
        return np.linalg.norm(params[:3]) - 1

    def unit_norm_constraint2(params):
        return np.linalg.norm(params[3:6]) - 1

    def angle_constraint(params):
        n1 = params[:3] / np.linalg.norm(params[:3])
        n2 = params[3:6] / np.linalg.norm(params[3:6])
        return np.dot(n1, n2) - target_cos

    # Initial guesses from current planes
    n1_init = plane1[:3] / np.linalg.norm(plane1[:3])
    n2_init = plane2[:3] / np.linalg.norm(plane2[:3])
    d1_init = plane1[3]
    d2_init = plane2[3]

    x0 = np.hstack([n1_init, n2_init, d1_init, d2_init])

    constraints = [
        {'type': 'eq', 'fun': unit_norm_constraint1},
        {'type': 'eq', 'fun': unit_norm_constraint2},
        {'type': 'eq', 'fun': angle_constraint},
    ]

    result = minimize(
        loss,
        x0=x0,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 500, 'disp': True}
    )

    if not result.success:
        print("Optimization failed:", result.message)

    n1_opt = result.x[:3] / np.linalg.norm(result.x[:3])
    n2_opt = result.x[3:6] / np.linalg.norm(result.x[3:6])
    d1_opt = result.x[6]
    d2_opt = result.x[7]

    plane1_opt = np.append(n1_opt, d1_opt)
    plane2_opt = np.append(n2_opt, d2_opt)

    return plane1_opt, plane2_opt

def angular_change_between_normals(n_orig, n_opt):
    n_orig = n_orig / np.linalg.norm(n_orig)
    n_opt = n_opt / np.linalg.norm(n_opt)
    dot_product = np.clip(np.dot(n_orig, n_opt), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def identify_planes_along_x(keep_planes, keep_inlier_clouds):
    """
    Identify and return sorted indices of keep_planes[1:5] from min to max real-world X.

    Returns:
        sorted_indices (list of int): Indices in keep_planes corresponding to sorted X position.
    """
    sub_planes = keep_planes[1:5]
    sub_clouds = keep_inlier_clouds[1:5]

    x_axis = np.array([1, 0, 0])
    centroids = [np.mean(np.asarray(cloud.points), axis=0) for cloud in sub_clouds]

    # Project each centroid onto X axis
    projected_x = [np.dot(centroid, x_axis) for centroid in centroids]

    # Get original indices from keep_planes
    original_indices = list(range(1, 5))

    # Sort based on projected X values
    sorted_tuples = sorted(zip(original_indices, projected_x), key=lambda t: t[1])
    sorted_indices = [idx for idx, _ in sorted_tuples]

    return sorted_indices

if __name__ == "__main__":
    mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3_point_bend_flat_surfaces.stl"
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(mesh_path)
    
    target_axes = [[0, 0, 1],
                   [1, 0, 0],
                   [np.sqrt(3)/2, 0, -0.5],
                   [-np.sqrt(3)/2, 0, -0.5]]
    
    keep_planes, keep_inlier_clouds, aligned_pcd = detect_fixture_planes(base_pcd, target_axes)
    
    o3d.visualization.draw_geometries([aligned_pcd] + keep_inlier_clouds, window_name='Keep Inlier Clouds')
    
    # Identify the X-aligned planes
    sorted_idxs = identify_planes_along_x(keep_planes, keep_inlier_clouds)
    print("Planes sorted along X (min to max):", sorted_idxs)
    sorted_planes = [keep_planes[i] for i in sorted_idxs]
    sorted_clouds = [keep_inlier_clouds[i] for i in sorted_idxs]
    
    
    
    print("\nOptimizing parallel planes for keep_planes[5] and keep_planes[6]...\n")
    
    p5_opt, p6_opt = optimize_planes_with_fixed_angle(
        keep_planes[5], keep_planes[6],
        keep_inlier_clouds[5], keep_inlier_clouds[6],
        target_angle_deg=60
    )
    
    optimized_plane_meshes = [
        create_plane_mesh(p5_opt, keep_inlier_clouds[5], plane_size=50.0),
        create_plane_mesh(p6_opt, keep_inlier_clouds[6], plane_size=50.0)
    ]
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [aligned_pcd, axis] + optimized_plane_meshes,
        window_name="Optimized Parallel Planes"
    )

    orig_n1 = keep_planes[5][:3]
    orig_n2 = keep_planes[6][:3]
    
    angle_change1 = angular_change_between_normals(orig_n1, p5_opt[:3])
    angle_change2 = angular_change_between_normals(orig_n2, p6_opt[:3])
    
    print(f"Plane 5 normal changed by {angle_change1:.2f} degrees")
    print(f"Plane 6 normal changed by {angle_change2:.2f} degrees")
    
    # residuals = check_plane_point_residuals(keep_planes, keep_inlier_clouds, threshold=0.1)
    
    # plane_counts = [1, 4, 1, 1]

    # # Construct cloud_assignments based on plane_counts
    # cloud_assignments = []
    # for cloud_idx, count in enumerate(plane_counts):
    #     cloud_assignments.extend([cloud_idx] * count)
    
    # initial_planes = [normalize_plane_append(p) for p in keep_planes]
    # flat_initial = np.concatenate(initial_planes)
    
    # # Build constraints
    # constraints = []
    
    # # Unit normal constraints
    # for i in range(len(keep_planes)):
    #     constraints.append({
    #         'type': 'eq',
    #         'fun': lambda x, i=i: constraint_unit_norm(x[i*4:(i+1)*4])
    #     })
    
    # # Orthogonality between first plane and next four (fixture walls)
    # for i in range(1, 5):
    #     constraints.append({
    #         'type': 'eq',
    #         'fun': (lambda i: lambda x: constraint_orthogonal(x[0:4], x[i*4:(i+1)*4]))(i)
    #     })

    
    # # Optimize
    # res = minimize(
    #     fun=total_residual_loss,
    #     x0=flat_initial,
    #     args=(keep_inlier_clouds, plane_counts),
    #     constraints=constraints,
    #     method='SLSQP',
    #     options={'maxiter': 500, 'disp': True}
    # )
    
    # # Parse result
    # optimized_planes = [res.x[i*4:(i+1)*4] for i in range(len(keep_planes))]
    
    # optimized_plane_meshes = []
    # for i, plane in enumerate(optimized_planes):
    #     inlier_cloud = keep_inlier_clouds[cloud_assignments[i]]
    #     optimized_plane_mesh = create_plane_mesh(plane, inlier_cloud, plane_size=50.0)
    #     optimized_plane_meshes.append(optimized_plane_mesh)
    
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([base_pcd, axis] + optimized_plane_meshes, window_name='Optimized Planes')

    
    
    
    
    # NEXT STEPS:
    # * Use the known relative angles and positions of the found datum planes
    #   to adjust the found datum planes into perfect relative position, while
    #   optimizing for the best fit of the scan data.
    
    # * Use the datum planes to determine the location of the supports and
    #   anvil. The anvil should be allowed to be at a slightly different angle
    #   than the supports, but it should have its axis in a plane parallel to
    #   the XY plane.
    
    # * Adjust the entire point cloud and the fit cylinders so the fit supports
    #   have identical Z coordinates and their axes are parallel to the XY plane.
    #   Place one of the centers of a support at the origin. Since we are not
    #   assuming that the anvil is parallel to the supports, we do not use this
    #   as a locating feature as before.
    
    # * Align the corresponding full specimen mesh to the scanned portion of
    #   the specimen in the test setup scan.
    
    # * Ensure there are no intersections between the specimen mesh and the
    #   suppors or anvil meshes. Adjust positions in the Z axis if necessary,
    #   but only by the minimum amount.
    
    # * Export the aligned specimen, anvil, left support, and right support as
    #   a new mesh.
    