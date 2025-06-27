import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from itertools import combinations
import time
import trimesh
from align_meshes import align_tgt_to_ref_meshes, load_mesh
import copy

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

def align_mesh_with_pca(mesh):
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError("Input must be an Open3D TriangleMesh.")

    # Convert vertices to numpy array
    vertices = np.asarray(mesh.vertices)

    # Center the mesh at the origin
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_vertices)
    axes = pca.components_

    # Sort axes by explained variance (descending)
    order = np.argsort(pca.explained_variance_)[::-1]
    ordered_axes = axes[order]

    # Align principal components with +X, +Y, +Z
    target_axes = np.eye(3)
    rotation_matrix = ordered_axes.T @ target_axes

    # Apply rotation
    rotated_vertices = centered_vertices @ rotation_matrix

    # Create a new mesh with rotated vertices and same faces
    aligned_mesh = o3d.geometry.TriangleMesh()
    aligned_mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    aligned_mesh.triangles = mesh.triangles

    # Recompute normals since geometry has changed
    aligned_mesh.compute_vertex_normals()

    return aligned_mesh


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

# def filter_duplicate_planes(planes, target_axis):
#     planes = np.array(planes)
#     n = len(planes)
#     used = [False] * n
#     keep = []

#     for i in range(n):
#         if used[i]:
#             continue
#         current_group = [i]
#         for j in range(i + 1, n):
#             if used[j]:
#                 continue
#             sim = plane_similarity(planes[i], planes[j])
#             # We find the closest neighbor based on similarity
#             if sim < 0.05:  # Very tight, not arbitrary — adjust if needed
#                 current_group.append(j)

#         # Select best-aligned with target_axis
#         best_idx = max(
#             current_group,
#             key=lambda idx: abs(np.dot(planes[idx][:3] / np.linalg.norm(planes[idx][:3]), target_axis))
#         )
#         keep.append(planes[best_idx])
#         for idx in current_group:
#             used[idx] = True

#     return keep

def filter_duplicate_planes(planes, target_axis, d_threshold=1.0):
    """
    Filters out planes with similar 'd' values (i.e., close positions along their shared normal direction).
    Uses DBSCAN to cluster by 'd' and picks the plane in each cluster whose normal is most aligned with target_axis.

    Args:
        planes: (N, 4) numpy array of plane definitions [a, b, c, d]
        target_axis: list or np.array of shape (3,), the preferred alignment direction
        d_threshold: float, max absolute difference in 'd' value for clustering (adjust as needed)

    Returns:
        List of filtered plane definitions (each is a 4-element numpy array)
    """
    planes = np.array(planes)
    normals = planes[:, :3]
    d_values = planes[:, 3].reshape(-1, 1)

    # Cluster based on the d value only
    clustering = DBSCAN(eps=d_threshold, min_samples=1).fit(d_values)
    labels = clustering.labels_

    keep = []
    target_axis = np.array(target_axis) / np.linalg.norm(target_axis)

    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        best_idx = max(
            cluster_indices,
            key=lambda idx: abs(np.dot(normals[idx] / np.linalg.norm(normals[idx]), target_axis))
        )
        keep.append(planes[best_idx])

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
        
    print(f'{len(planes)} planes detected for axis {target_axis}')
    
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
    
    # while len(keep_planes) != 7:
    while True:
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
                        # base_plane = max(filtered_planes, key=lambda x: x[3])
                        # base_idx = filtered_planes.index(base_plane)
                        
                        base_idx = max(range(len(filtered_planes)), key=lambda i: filtered_planes[i][3])
                        base_plane = filtered_planes[base_idx]
                        
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
                # print(f"Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")
                raise Exception(f"Failed to detect valid planes for axis {target_axis} after {max_retries} attempts.")
                
                
        break
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

def identify_planes_along_x(planes, clouds):
    """
    planes: list of plane coefficients
    clouds: list of corresponding point clouds
    
    Returns indices of planes in order from min X to max X.
    So returns 4 indices: outer_min, inner_min, inner_max, outer_max
    """
    # Compute centroid X for each cloud
    centroids = [np.mean(np.asarray(c.points), axis=0) for c in clouds]
    x_coords = [c[0] for c in centroids]
    
    # Sort indices by X coordinate
    sorted_indices = sorted(range(len(x_coords)), key=lambda i: x_coords[i])
    
    for idx in sorted_indices:
        print(f'\nX Plane Fit Coordinate: {x_coords[idx]}')    
        print(f'Plane Definition: {planes[idx]}')
    
    # Return all four sorted indices
    return tuple(sorted_indices)  # will be 4 indices



###############################################################################
# OPTIMIZE FOR ALL CONSTRAINTS

def normalize(v):
    return v / np.linalg.norm(v)


def residual_loss(flat_params, inlier_clouds):
    loss = 0.0
    for i in range(len(inlier_clouds)):
        n = normalize(flat_params[i * 4:i * 4 + 3])
        d = flat_params[i * 4 + 3]
        pts = np.asarray(inlier_clouds[i].points)
        distances = pts @ n + d
        loss += np.sum(distances ** 2)
    return loss


def define_constraints():
    cons = []

    # Plane 0 orthogonal to planes 1 to 4
    for i in range(1, 5):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.dot(
                normalize(x[0:3]), normalize(x[i * 4:i * 4 + 3])
            )
        })

    # Planes 1 & 4 parallel: normals must match or be opposite
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[1 * 4:1 * 4 + 3]), normalize(x[4 * 4:4 * 4 + 3])
        ) - 1
    })

    # Planes 2 & 3 parallel
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[2 * 4:2 * 4 + 3]), normalize(x[3 * 4:3 * 4 + 3])
        ) - 1
    })

    # Planes 5 and 6: 60 degree angle between normals
    target_angle_rad = np.radians(60)
    target_dot = np.cos(target_angle_rad)
    cons.append({
        'type': 'eq',
        'fun': lambda x: np.dot(
            normalize(x[5 * 4:5 * 4 + 3]), normalize(x[6 * 4:6 * 4 + 3])
        ) - target_dot
    })

    # Line of intersection of planes 5 and 6 should be parallel to inner planes 2 & 3 (in XY only)
    def intersection_dir(x):
        n5 = normalize(x[5 * 4:5 * 4 + 3])
        n6 = normalize(x[6 * 4:6 * 4 + 3])
        direction = normalize(np.cross(n5, n6))
        return direction

    def dir_xy_parallel(x):
        d = intersection_dir(x)
        n2 = normalize(x[2 * 4:2 * 4 + 3])
        return np.dot(d[:2], n2[:2])

    cons.append({
        'type': 'eq',
        'fun': dir_xy_parallel
    })

    # Intersection line should have zero Z component
    cons.append({
        'type': 'eq',
        'fun': lambda x: intersection_dir(x)[2]
    })

    # Unit norm constraint for each plane normal
    for i in range(7):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.linalg.norm(x[i * 4:i * 4 + 3]) - 1
        })

    return cons


def optimize_all_planes(keep_planes, keep_inlier_clouds):
    """
    Optimize planes with the following constraints:
    - keep_planes[0] orthogonal to keep_planes[1:5]
    - Among keep_planes[1:5], identify outer and inner pairs along X:
       * outer two planes parallel
       * inner two planes parallel
    - keep_planes[5] and keep_planes[6] have 60 degrees separation
    - Intersection line of planes 5 & 6 is parallel to the inner pair and has zero Z component
    """

    # Identify which indices correspond to outer and inner planes in keep_planes[1:5]
    idx_outer1, idx_inner1, idx_inner2, idx_outer2 = identify_planes_along_x(
        keep_planes[1:5], keep_inlier_clouds[1:5]
    )
    # Map these local indices to global indices in keep_planes
    x_plane_indices = [1 + idx_outer1, 1 + idx_inner1, 1 + idx_inner2, 1 + idx_outer2]

    # Flatten all planes into parameters for optimization
    flat_params = np.hstack(keep_planes)

    def constraint_func(params):
        n_planes = len(params) // 4
        planes = [params[i*4:(i+1)*4] for i in range(n_planes)]

        constraints = []

        # 1) keep_planes[0] orthogonal to all planes in keep_planes[1:5]
        base_plane = planes[0]
        for idx in x_plane_indices:
            n_base = base_plane[:3] / np.linalg.norm(base_plane[:3])
            n_other = planes[idx][:3] / np.linalg.norm(planes[idx][:3])
            # Dot product should be zero for orthogonality
            constraints.append(np.dot(n_base, n_other))

        # 2) Outer pair of keep_planes[1:5] parallel
        l_support = planes[x_plane_indices[0]][:3] / np.linalg.norm(planes[x_plane_indices[0]][:3])
        r_support = planes[x_plane_indices[3]][:3] / np.linalg.norm(planes[x_plane_indices[3]][:3])
        constraints.append(np.dot(l_support, r_support) - 1)  # parallel means dot == ±1, use +1 here and consider orientation fix if needed

        # 3) Inner pair of keep_planes[1:5] parallel
        l_anvil = planes[x_plane_indices[1]][:3] / np.linalg.norm(planes[x_plane_indices[1]][:3])
        r_anvil = planes[x_plane_indices[2]][:3] / np.linalg.norm(planes[x_plane_indices[2]][:3])
        constraints.append(np.dot(l_anvil, r_anvil) - 1)

        # 4) keep_planes[5] and keep_planes[6] separated by 60 degrees
        n5 = planes[5][:3] / np.linalg.norm(planes[5][:3])
        n6 = planes[6][:3] / np.linalg.norm(planes[6][:3])
        cos_60 = np.cos(np.radians(60))
        constraints.append(np.dot(n5, n6) - cos_60)
        
        # 5) Ensure sides of anvil and angled faces of anvil are 150 degrees apart
        cos_150 = np.cos(np.radians(150))
        constraints.append(np.dot(n5, l_anvil) - np.abs(cos_150))
        constraints.append(np.dot(n6, r_anvil) - np.abs(cos_150))

        # 6) Each plane normal must be unit length
        for plane in planes:
            constraints.append(np.linalg.norm(plane[:3]) - 1)
            
        # # 7) Support planes must be separated by supports_distance
        # n5 = planes[5][:3]
        # n6 = planes[6][:3]
        # d5 = planes[5][3]
        # d6 = planes[6][3]
        
        # # Compute unit normals
        # n5_unit = normalize(n5)
        # n6_unit = normalize(n6)
        
        # # Mid-normal (should be parallel to both)
        # n_avg = normalize((n5_unit + n6_unit) / 2)
        
        # # Signed separation: project difference of d terms onto normal
        # if d6 >= d5:
        #     separation = (d6 - d5) / np.dot(n_avg, n5_unit)
        # else:
        #     separation = (d5 - d6) / np.dot(n_avg, n5_unit)
        
        # # Enforce positive direction by sign convention
        # constraints.append(separation - supports_distance)
                
        # 8) Anvil vertical planes must be separated by 25 mm
        l_anvil_plane = planes[x_plane_indices[1]]
        r_anvil_plane = planes[x_plane_indices[2]]
        
        na1 = l_anvil_plane[:3]
        na2 = r_anvil_plane[:3]
        da1 = l_anvil_plane[3]
        da2 = r_anvil_plane[3]
        
        na1_unit = normalize(na1)
        na2_unit = normalize(na2)
        na_avg = normalize((na1_unit + na2_unit) / 2)
        
        if da1 >= da2:
            anvil_sep = (da1 - da2) / np.dot(na_avg, na1_unit)
        else:
            anvil_sep = (da2 - da1) / np.dot(na_avg, na1_unit)
        constraints.append(anvil_sep - 25.0)


        return np.array(constraints)

    # Define objective function: sum of squared residuals of points to their respective planes
    def objective_func(params):
        n_planes = len(params) // 4
        planes = [params[i*4:(i+1)*4] for i in range(n_planes)]
        loss = 0.0
        for i, plane in enumerate(planes):
            n = plane[:3]
            n /= np.linalg.norm(n)
            d = plane[3]
            pts = np.asarray(keep_inlier_clouds[i].points)
            res = pts @ n + d
            loss += np.sum(res**2)
        return loss

    n_constraints = len(constraint_func(flat_params))
    cons = [{'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]} for i in range(n_constraints)]

    result = minimize(
        objective_func,
        flat_params,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 500,
                 'disp': True,
                 'ftol': 1e-10}
    )

    if not result.success:
        print("Optimization failed:", result.message)

    optimized_planes = [result.x[i*4:(i+1)*4] for i in range(len(keep_planes))]
    
    # After optimization, create plane meshes for visualization
    optimized_plane_meshes = []
    for plane, cloud in zip(optimized_planes, keep_inlier_clouds):
        mesh = create_plane_mesh(plane, cloud, plane_size=50.0)
        optimized_plane_meshes.append(mesh)

    return optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds


def separation_between_parallel_planes(plane1, plane2):
    """
    Computes the perpendicular distance between two parallel planes.
    Planes must be in the form [a, b, c, d] for ax + by + cz + d = 0.
    """
    n1 = plane1[:3]
    d1 = plane1[3]
    d2 = plane2[3]

    # Normalize the normal vector
    n1_norm = n1 / np.linalg.norm(n1)

    # Compute signed distance from origin to each plane
    # Signed distance = -d / ||n||
    dist1 = -d1 / np.linalg.norm(n1)
    dist2 = -d2 / np.linalg.norm(n1)

    # Return the absolute separation between the planes
    return abs(dist1 - dist2)

def create_cylinder_relative_to_planes(
    plane1, plane2,
    plane1_offset,
    plane2_offset,
    diameter, height
):
    """
    Create a cylinder aligned with the intersection of two orthogonal planes.

    Parameters:
    - plane1, plane2: 4-element numpy arrays [a, b, c, d] for ax + by + cz + d = 0
    - plane1_offset, plane2_offset: Scalar distances (can be positive or negative)
    - diameter: Cylinder diameter
    - height: Cylinder height

    Returns:
    - trimesh.Trimesh cylinder mesh
    """

    def flip_if_major_negative(plane):
        normal = plane[:3]
        major_idx = np.argmax(np.abs(normal))
        if normal[major_idx] < 0:
            return np.concatenate([-normal, [-plane[3]]])
        return plane

    # Ensure normals face positive along their major axis
    plane1 = flip_if_major_negative(plane1)
    plane2 = flip_if_major_negative(plane2)

    # Extract normals
    n1 = plane1[:3]
    n2 = plane2[:3]

    # Axis of the cylinder = intersection line of planes
    axis_dir = np.cross(n1, n2)
    axis_dir /= np.linalg.norm(axis_dir)

    # Find a point on both planes (intersection point)
    A = np.vstack([n1, n2, axis_dir])
    b = -np.array([plane1[3], plane2[3], 0])
    intersection_point = np.linalg.lstsq(A, b, rcond=None)[0]

    # Offset from each plane along its normal
    offset_vector = n1 * plane1_offset + n2 * plane2_offset
    base_center = intersection_point + offset_vector - axis_dir * height / 2

    # Create default cylinder aligned with +Z
    cyl = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=32)

    # Rotate to align Z axis with desired axis_dir
    z_axis = np.array([0, 0, 1])
    if np.allclose(axis_dir, z_axis):
        R = np.eye(3)
    elif np.allclose(axis_dir, -z_axis):
        R = -np.eye(3)
    else:
        rotation_axis = np.cross(z_axis, axis_dir)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, axis_dir), -1.0, 1.0))
        R = trimesh.transformations.rotation_matrix(angle, rotation_axis)[:3, :3]

    # Compose transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = base_center
    cyl.apply_transform(T)

    # Translate further along the cylinder's axis
    T2 = np.eye(4)
    T2[:3, 3] = axis_dir * (height / 2)
    cyl.apply_transform(T2)

    return cyl

def trimesh_to_open3d(tri_mesh):
    """
    Convert a trimesh.Trimesh object to open3d.geometry.TriangleMesh.
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def align_planes_to_axes_minimal_v2(aligned_pcd, optimized_planes, planeX, planeZ):
    """
    Align planes to coordinate axes using truly minimal rotation.
    This version uses a more robust approach to avoid large rotations.
    
    Args:
        aligned_pcd: Open3D point cloud to rotate
        optimized_planes: List of plane equations [a, b, c, d] to rotate
        planeX: Plane equation whose normal should align with X-axis
        planeZ: Plane equation whose normal should align with Z-axis
    
    Returns:
        rotated_pcd: Rotated point cloud
        rotated_planes: List of rotated plane equations
        rotation_matrix: 3x3 rotation matrix that was applied
        rotation_info: Dictionary with rotation details
    """
    
    # Extract and normalize plane normals
    normalX = planeX[:3] / np.linalg.norm(planeX[:3])
    normalZ = planeZ[:3] / np.linalg.norm(planeZ[:3])
    
    print(f"Original normalX: {normalX}")
    print(f"Original normalZ: {normalZ}")
    
    # Target axes - we'll determine the best direction more carefully
    target_X_pos = np.array([1, 0, 0])
    target_X_neg = np.array([-1, 0, 0])
    target_Z_pos = np.array([0, 0, 1])
    target_Z_neg = np.array([0, 0, -1])
    
    # Calculate all possible alignments and their required rotation angles
    def rotation_angle_between_vectors(v1, v2):
        """Calculate the rotation angle needed to align v1 with v2"""
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(abs(dot_product))
    
    # Test all four combinations
    options = [
        (target_X_pos, target_Z_pos, "X+, Z+"),
        (target_X_pos, target_Z_neg, "X+, Z-"),
        (target_X_neg, target_Z_pos, "X-, Z+"),
        (target_X_neg, target_Z_neg, "X-, Z-")
    ]
    
    best_option = None
    min_total_angle = float('inf')
    
    print("\nEvaluating alignment options:")
    for target_X, target_Z, label in options:
        angle_X = rotation_angle_between_vectors(normalX, target_X)
        angle_Z = rotation_angle_between_vectors(normalZ, target_Z)
        total_angle = angle_X + angle_Z
        
        print(f"{label}: X angle = {np.degrees(angle_X):.3f}°, Z angle = {np.degrees(angle_Z):.3f}°, Total = {np.degrees(total_angle):.3f}°")
        
        if total_angle < min_total_angle:
            min_total_angle = total_angle
            best_option = (target_X, target_Z, label, angle_X, angle_Z)
    
    final_X, final_Z, best_label, best_angle_X, best_angle_Z = best_option
    print(f"\nSelected option: {best_label}")
    print(f"Target X direction: {final_X}")
    print(f"Target Z direction: {final_Z}")
    
    # Check orthogonality of target directions
    orthogonality_check = abs(np.dot(final_X, final_Z))
    if orthogonality_check > 1e-10:
        print(f"Warning: Target directions not orthogonal! Dot product: {orthogonality_check}")
    
    # Now compute the actual minimal rotation
    # We'll use a different approach: find the rotation that simultaneously
    # minimizes the distance to both target orientations
    
    # Method: Use the fact that we want to solve:
    # R @ normalX ≈ final_X
    # R @ normalZ ≈ final_Z
    # 
    # This is an orthogonal Procrustes problem
    R = compute_optimal_rotation_procrustes(
        np.column_stack([normalX, normalZ]), 
        np.column_stack([final_X, final_Z])
    )
    
    # Alternative method if Procrustes doesn't work well:
    # Use two sequential minimal rotations
    if np.linalg.det(R) < 0.9 or np.linalg.norm(R @ R.T - np.eye(3)) > 1e-6:
        print("Procrustes method failed, using sequential rotations")
        R = compute_sequential_minimal_rotation(normalX, normalZ, final_X, final_Z)
    
    # Verify the rotation
    final_normalX = R @ normalX
    final_normalZ = R @ normalZ
    
    print("\nAfter rotation:")
    print(f"Final normalX: {final_normalX}")
    print(f"Final normalZ: {final_normalZ}")
    
    # Check alignment quality
    alignment_X = abs(np.dot(final_normalX, final_X))
    alignment_Z = abs(np.dot(final_normalZ, final_Z))
    
    angle_error_X = np.degrees(np.arccos(np.clip(alignment_X, 0, 1)))
    angle_error_Z = np.degrees(np.arccos(np.clip(alignment_Z, 0, 1)))
    
    print(f"X alignment: {alignment_X:.6f} (error: {angle_error_X:.6f}°)")
    print(f"Z alignment: {alignment_Z:.6f} (error: {angle_error_Z:.6f}°)")
    
    # Verify rotation matrix properties
    det_R = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    
    if abs(det_R - 1.0) > 1e-10:
        print(f"Warning: Rotation matrix determinant = {det_R}, should be 1.0")
    if orthogonality_error > 1e-10:
        print(f"Warning: Rotation matrix orthogonality error = {orthogonality_error}")
    
    # Create transformation matrix and apply
    T = np.eye(4)
    T[:3, :3] = R
    rotated_pcd = aligned_pcd.transform(T)
    
    # Apply rotation to all plane equations
    rotated_planes = []
    for plane in optimized_planes:
        old_normal = plane[:3]
        new_normal = R @ old_normal
        new_plane = np.array([new_normal[0], new_normal[1], new_normal[2], plane[3]])
        rotated_planes.append(new_plane)
    
    # Prepare rotation info
    rotation_scipy = Rotation.from_matrix(R)
    euler_angles = rotation_scipy.as_euler('xyz', degrees=True)
    rotvec = rotation_scipy.as_rotvec()
    rotation_angle = np.linalg.norm(rotvec) * 180 / np.pi
    
    rotation_info = {
        'original_normalX': normalX,
        'original_normalZ': normalZ,
        'target_X': final_X,
        'target_Z': final_Z,
        'final_normalX': final_normalX,
        'final_normalZ': final_normalZ,
        'euler_angles_deg': euler_angles,
        'rotation_axis_angle': rotvec,
        'rotation_angle_deg': rotation_angle,
        'alignment_errors_deg': (angle_error_X, angle_error_Z),
        'best_option': best_label
    }
    
    print("\nRotation summary:")
    print(f"Euler angles (XYZ): {euler_angles}")
    print(f"Total rotation angle: {rotation_angle:.3f}°")
    
    return rotated_pcd, rotated_planes, R, rotation_info


def compute_optimal_rotation_procrustes(source_vectors, target_vectors):
    """
    Compute optimal rotation using orthogonal Procrustes analysis.
    This finds the rotation R that minimizes ||R @ source_vectors - target_vectors||_F
    """
    # Compute SVD of target_vectors @ source_vectors.T
    H = target_vectors @ source_vectors.T
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = U @ Vt
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    return R


def compute_sequential_minimal_rotation(normalX, normalZ, final_X, final_Z):
    """
    Compute rotation using sequential minimal rotations.
    """
    # Step 1: Rotate normalX to final_X
    R1 = compute_rotation_between_vectors(normalX, final_X)
    
    # Step 2: See where normalZ goes after R1
    normalZ_intermediate = R1 @ normalZ
    
    # Step 3: Rotate around final_X to align normalZ_intermediate with final_Z
    R2 = compute_rotation_around_axis(final_X, normalZ_intermediate, final_Z)
    
    # Combined rotation
    R = R2 @ R1
    
    return R


def compute_rotation_between_vectors(v1, v2):
    """
    Compute rotation matrix that rotates vector v1 to vector v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Check if vectors are already aligned
    dot_product = np.dot(v1, v2)
    if abs(dot_product - 1.0) < 1e-10:
        return np.eye(3)
    
    # Check if vectors are opposite
    if abs(dot_product + 1.0) < 1e-10:
        # Find an orthogonal vector for 180° rotation
        if abs(v1[0]) < 0.9:
            orthogonal = np.array([1, 0, 0])
        else:
            orthogonal = np.array([0, 1, 0])
        
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        return rodrigues_rotation(axis, np.pi)
    
    # General case using Rodrigues' formula
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) < 1e-10:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return rodrigues_rotation(axis, angle)


def compute_rotation_around_axis(axis, v_from, v_to):
    """
    Compute rotation around given axis that rotates v_from to v_to.
    """
    axis = axis / np.linalg.norm(axis)
    
    # Project vectors onto plane perpendicular to axis
    v_from_proj = v_from - np.dot(v_from, axis) * axis
    v_to_proj = v_to - np.dot(v_to, axis) * axis
    
    # Check if vectors are already aligned in the plane
    if np.linalg.norm(v_from_proj) < 1e-10 or np.linalg.norm(v_to_proj) < 1e-10:
        return np.eye(3)
    
    # Normalize projected vectors
    v_from_proj = v_from_proj / np.linalg.norm(v_from_proj)
    v_to_proj = v_to_proj / np.linalg.norm(v_to_proj)
    
    # Compute angle between projected vectors
    dot_product = np.clip(np.dot(v_from_proj, v_to_proj), -1.0, 1.0)
    cross_product = np.cross(v_from_proj, v_to_proj)
    
    # Determine angle sign
    angle = np.arccos(dot_product)
    if np.dot(cross_product, axis) < 0:
        angle = -angle
    
    return rodrigues_rotation(axis, angle)


def rodrigues_rotation(axis, angle):
    """
    Compute rotation matrix using Rodrigues' rotation formula.
    """
    if abs(angle) < 1e-10:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Skew-symmetric matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # Rodrigues' formula
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    
    return R


# Debug function to help understand what's happening
def debug_plane_orientations(planes, labels=None):
    """
    Debug function to print plane normal orientations
    """
    print("\n=== PLANE ORIENTATIONS DEBUG ===")
    for i, plane in enumerate(planes):
        normal = plane[:3] / np.linalg.norm(plane[:3])
        label = labels[i] if labels else f"Plane {i}"
        
        # Compute angles with coordinate axes
        angle_x = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [1,0,0])), 0, 1)))
        angle_y = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [0,1,0])), 0, 1)))
        angle_z = np.degrees(np.arccos(np.clip(abs(np.dot(normal, [0,0,1])), 0, 1)))
        
        print(f"{label}: normal = {normal}")
        print(f"  Angles to axes: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°")
        
        # Show which axis is closest
        min_angle = min(angle_x, angle_y, angle_z)
        if min_angle == angle_x:
            closest = "X-axis"
        elif min_angle == angle_y:
            closest = "Y-axis"
        else:
            closest = "Z-axis"
        print(f"  Closest to: {closest} ({min_angle:.1f}°)")
    print("=" * 40)


def validate_minimal_rotation(original_planes, rotated_planes, planeX_idx, planeZ_idx, rotation_matrix):
    """
    Validate that the minimal rotation alignment worked correctly.
    """
    print("\n=== MINIMAL ROTATION VALIDATION ===")
    
    # Check the target planes
    planeX_original = original_planes[planeX_idx]
    planeZ_original = original_planes[planeZ_idx]
    planeX_rotated = rotated_planes[planeX_idx]
    planeZ_rotated = rotated_planes[planeZ_idx]
    
    # Original normals
    normalX_orig = planeX_original[:3] / np.linalg.norm(planeX_original[:3])
    normalZ_orig = planeZ_original[:3] / np.linalg.norm(planeZ_original[:3])
    
    # Rotated normals
    normalX_rot = planeX_rotated[:3] / np.linalg.norm(planeX_rotated[:3])
    normalZ_rot = planeZ_rotated[:3] / np.linalg.norm(planeZ_rotated[:3])
    
    # Target axes
    target_X = np.array([1, 0, 0])
    target_Z = np.array([0, 0, 1])
    
    # Check alignment
    alignment_X_pos = abs(np.dot(normalX_rot, target_X))
    alignment_X_neg = abs(np.dot(normalX_rot, -target_X))
    alignment_Z_pos = abs(np.dot(normalZ_rot, target_Z))
    alignment_Z_neg = abs(np.dot(normalZ_rot, -target_Z))
    
    best_X_alignment = max(alignment_X_pos, alignment_X_neg)
    best_Z_alignment = max(alignment_Z_pos, alignment_Z_neg)
    
    angle_error_X = np.arccos(np.clip(best_X_alignment, 0, 1)) * 180 / np.pi
    angle_error_Z = np.arccos(np.clip(best_Z_alignment, 0, 1)) * 180 / np.pi
    
    print(f"PlaneX alignment with X-axis: {best_X_alignment:.6f} (error: {angle_error_X:.6f}°)")
    print(f"PlaneZ alignment with Z-axis: {best_Z_alignment:.6f} (error: {angle_error_Z:.6f}°)")
    
    # Check total rotation angle
    rotation_scipy = Rotation.from_matrix(rotation_matrix)
    total_angle = np.linalg.norm(rotation_scipy.as_rotvec()) * 180 / np.pi
    print(f"Total rotation angle: {total_angle:.3f}°")
    
    success = angle_error_X < 0.1 and angle_error_Z < 0.1
    print(f"Alignment successful: {success}")
    
    return success, angle_error_X, angle_error_Z, total_angle

# def generate_candidate_orientations(mesh, visualize=False, target_pcd=None):
#     print("[INFO] Generating candidate 180° flips...")
#     identities = {
#         "Original": np.eye(3),
#         "Flip Z": Rotation.from_euler('z', 180, degrees=True).as_matrix(),
#         "Flip X": Rotation.from_euler('x', 180, degrees=True).as_matrix(),
#         "Flip Z+X": Rotation.from_euler('zx', [180, 180], degrees=True).as_matrix()
#     }

#     candidates = []
#     for name, rot in identities.items():
#         m = copy.deepcopy(mesh)
#         m.rotate(rot, center=(0, 0, 0))
#         print(f"  - Candidate: {name}")
#         if visualize and target_pcd:
#             o3d.visualization.draw_geometries([
#                 m.paint_uniform_color([1, 0, 0]),
#                 target_pcd.paint_uniform_color([0, 1, 0]),
#                 axis
#             ], window_name=f"Candidate Orientation: {name}")
#         candidates.append((m, name))
#     return candidates

def generate_candidate_orientations(mesh, visualize=False, target_pcd=None):
    print("[INFO] Generating candidate 180° flips...")

    # Define all required 180° rotations
    rotations = {
        "Original": np.eye(3),
        "Flip Z": Rotation.from_euler('z', 180, degrees=True).as_matrix(),
        "Flip X": Rotation.from_euler('x', 180, degrees=True).as_matrix(),
        "Flip Z+X": Rotation.from_euler('x', 180, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix()
    }

    candidates = []
    for name, rot in rotations.items():
        m = copy.deepcopy(mesh)
        m.rotate(rot, center=(0, 0, 0))
        print(f"  - Candidate: {name}")
        
        if visualize and target_pcd is not None:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
            m.paint_uniform_color([1, 0, 0])
            target_pcd.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries(
                [m, target_pcd, axis],
                window_name=f"Candidate Orientation: {name}"
            )
        
        candidates.append((m, name))

    return candidates


if __name__ == "__main__":
    mesh_path = "E:/Fixture Scans/scan_1_with_specimen.stl"
    # mesh_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3_point_bend_flat_surfaces.stl"
    print("Loading mesh...")
    base_pcd = load_mesh_as_point_cloud(mesh_path)
    
    target_axes = [[0, 0, 1],
                   [1, 0, 0],
                   [np.sqrt(3)/2, 0, -0.5],
                   [-np.sqrt(3)/2, 0, -0.5]]
    
    supports_distance = 127.66
    
    keep_planes, keep_inlier_clouds, aligned_pcd = detect_fixture_planes(base_pcd, target_axes)
    
    # o3d.visualization.draw_geometries([aligned_pcd] + keep_inlier_clouds, window_name='Keep Inlier Clouds')
    
    optimized_planes, x_plane_indices, optimized_plane_meshes, keep_inlier_clouds = optimize_all_planes(keep_planes, keep_inlier_clouds)
    
    # o3d.visualization.draw_geometries([aligned_pcd] + optimized_planes, window_name='Optimized_Planes')
    
    # Verification of constraints
    print("")
    for combo in combinations(range(len(optimized_planes)), 2):
        n1 = optimized_planes[combo[0]][:3]
        n2 = optimized_planes[combo[1]][:3]
        angle_diff = angular_change_between_normals(n1, n2)
        
        # Check if the current combo contains X-oriented planes
        set_combo = set(combo)
        set_x_plane_indices = set(x_plane_indices)
        
        common_elements = list(set_combo.intersection(set_x_plane_indices))
        
        labels = []
        for ele in common_elements:
            x_plane_pos = x_plane_indices.index(ele)
            if x_plane_pos == 0:
                labels.append("Left Support")
            elif x_plane_pos == 1:
                labels.append("Left Side Anvil")
            elif x_plane_pos == 2:
                labels.append("Right Side Anvil")
            elif x_plane_pos == 3:
                labels.append("Right Support")
            else:
                raise ValueError("Index out of range")
            
        if len(common_elements) == 2:
            print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} ({labels[1]}) are separated by {angle_diff:.4f} degrees')
        elif len(common_elements) == 1:
            if combo[0] in common_elements:
                print(f'Planes {combo[0]} ({labels[0]}) and {combo[1]} are separated by {angle_diff:.4f} degrees')
            else:
                print(f'Planes {combo[0]} and {combo[1]} ({labels[0]}) are separated by {angle_diff:.4f} degrees')                
        else:
            print(f'Planes {combo[0]} and {combo[1]} are separated by {angle_diff:.4f} degrees')
        
        if np.abs(np.around(angle_diff,2)) <= 0.01:
            separation_dist = separation_between_parallel_planes(optimized_planes[combo[0]], optimized_planes[combo[1]])
            print(f'\tPlanes {combo[0]} and {combo[1]} are separated by a distance of {separation_dist}')
    
    
    # Re-align everything so the support planes and base plane define the world orientation
    planeX_idx = x_plane_indices[0]
    planeZ_idx = 0
    planeX = optimized_planes[planeX_idx]
    planeZ = optimized_planes[planeZ_idx]
    
    # Perform alignment
    rotated_pcd, rotated_planes, R, info = align_planes_to_axes_minimal_v2(
        aligned_pcd, optimized_planes, planeX, planeZ
    )
    
    # Validate results
    success, error_X, error_Z, total_angle = validate_minimal_rotation(
        optimized_planes, rotated_planes, planeX_idx, planeZ_idx, R
    )    
    
    
    # # Apply the R rotation matrix to the original mesh
    # test_mesh = load_mesh(mesh_path)
    # test_mesh_rotated = test_mesh.rotate(R, center=(0,0,0))
    
    # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    # matched_specimen_mesh = load_mesh(matched_specimen_scan_path)
    
    # aligned_specimen_mesh = align_tgt_to_ref_meshes(test_mesh_rotated, matched_specimen_mesh)
    
    # Load original mesh
    original_mesh = load_mesh(mesh_path)
    
    # Get original mesh centroid for translation
    original_vertices = np.asarray(original_mesh.vertices)
    centroid = original_vertices.mean(axis=0)
    
    # Step 1: Translate to origin
    translated_vertices = original_vertices - centroid
    
    # Step 2: Apply rotation matrix
    rotated_vertices = translated_vertices @ R.T  # Open3D expects R.T for .rotate()
    
    # Step 3: Create transformed mesh
    transformed_reference_mesh = o3d.geometry.TriangleMesh()
    transformed_reference_mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    transformed_reference_mesh.triangles = original_mesh.triangles
    transformed_reference_mesh.compute_vertex_normals()
    
    # Optionally: Save or visualize it
    # o3d.io.write_triangle_mesh("transformed_reference.stl", transformed_reference_mesh)
    
    # Load and align matched specimen
    matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    matched_specimen_mesh = load_mesh(matched_specimen_scan_path)
    
    aligned_specimen_mesh = align_tgt_to_ref_meshes(transformed_reference_mesh, matched_specimen_mesh)

    
    
    
    
    
    
    # # #### Align single scanned STL to the full test setup scan
    # # # FIX: Need to scan the specific matching specimen
    # # matched_specimen_scan_path = "E:/Fixture Scans/specimen.stl"
    # # matched_specimen_mesh = load_mesh(matched_specimen_scan_path)
    # # # matched_specimen_mesh.compute_vertex_normals()
    # # matched_specimen_mesh = align_mesh_with_pca(matched_specimen_mesh)
    # # matched_specimen_mesh.orient_triangles()
    
    
    # # # Determine fit performance of initial alignment (PCA)
    # # print('Testing fit performance of initial PCA alignment')
    # # aligned_specimen_mesh, metrics = align_tgt_mesh_to_ref_pcd(rotated_pcd, matched_specimen_mesh, visualize=True)
    # # best_percentile = metrics['percentile_95']
    # # best_mesh = aligned_specimen_mesh
    
    # # candidates = generate_candidate_orientations(matched_specimen_mesh, visualize=False)
    # # for mesh, label in candidates:
    # #     print(f'Testing fit performance for {label}')
    # #     aligned_specimen_mesh, metrics = align_tgt_mesh_to_ref_pcd(rotated_pcd, matched_specimen_mesh, visualize=True)
    # #     percentile = metrics['percentile_95']
    # #     if percentile < best_percentile:
    # #         best_percentile = percentile
    # #         best_mesh = aligned_specimen_mesh
    # #         print('{label} is the current best fit')
    # #     # fitness = result.fitness
    # #     # if fitness > best_fitness:
    # #     #     best_fitness = fitness
    # #     #     best_mesh = transformed_source_mesh
    # #     #     print(f'Found better candidate: {label}, Fitness={best_fitness}')
    
    
    
    
    
    #### Create mesh models of support and anvil cylinders
    diameter = 10
    height = 40
    
    base_plane = rotated_planes[0]
    base_support_offset = 52
    
    support_offset = 25.4   # 1 inch
    
    anvil_plane1 = rotated_planes[5]
    anvil_plane2 = rotated_planes[6]
    anvil = create_cylinder_relative_to_planes(anvil_plane1,
                                            anvil_plane2,
                                            0,
                                            0,
                                            diameter,
                                            height)
    anvil_mesh = trimesh_to_open3d(anvil)
    
    l_support_plane = rotated_planes[x_plane_indices[0]]
    l_support_plane_offset = support_offset
    l_support = create_cylinder_relative_to_planes(
                    base_plane, l_support_plane,
                    base_support_offset,
                    l_support_plane_offset,
                    diameter, height)
    
    l_support_mesh = trimesh_to_open3d(l_support)
    
    r_support_plane = rotated_planes[x_plane_indices[3]]
    r_support_plane_offset = -support_offset
    r_support = create_cylinder_relative_to_planes(
                    base_plane, r_support_plane,
                    base_support_offset,
                    r_support_plane_offset,
                    diameter, height)
    
    r_support_mesh = trimesh_to_open3d(r_support)
    
    
    
    
    
    # # Perform final alignment so supports are aligned to WCS
    # aligned_pcd_rot, l_support_rot, r_support_rot, anvil_rot = align_supports_to_Y_axis_and_Z0(aligned_pcd, l_support, r_support, anvil)
    
    # Create coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])

    # Visualize optimized planes and the inlier clouds used for fitting
    # o3d.visualization.draw_geometries(keep_inlier_clouds + [axis] + optimized_plane_meshes + [l_support_mesh], window_name="Optimized Planes")
    cylinder_meshes = [anvil_mesh, l_support_mesh, r_support_mesh]
    # cylinder_meshes_aligned = [trimesh_to_open3d(mesh) for mesh in cylinder_meshes]
    o3d.visualization.draw_geometries([rotated_pcd] + [axis] + cylinder_meshes + [aligned_specimen_mesh], window_name="Aligned Cylinders")
    # o3d.visualization.draw_geometries(keep_inlier_clouds + [axis] + optimized_plane_meshes + cylinder_meshes, window_name="Optimized Planes")
    
    
    
    
    # NEXT STEPS:
    # * Use the known relative angles and positions of the found datum planes
    #   to adjust the found datum planes into perfect relative position, while
    #   optimizing for the best fit of the scan data. ******DONE******
    
    # * Use known distances between datum faces to adjust placement (25 mm
    #   between vertical faces on the anvil - let the supports be where the
    #   scan indicates) ******DONE******
    
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
    #   support or anvil meshes. Adjust positions in the Z axis if necessary,
    #   but only by the minimum amount.
    
    # * Export the aligned specimen, anvil, left support, and right support as
    #   a new mesh.
    