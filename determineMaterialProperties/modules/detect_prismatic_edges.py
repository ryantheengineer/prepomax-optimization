import open3d as o3d
import numpy as np
import trimesh
from sklearn.cluster import KMeans
from tqdm import tqdm

def fit_plane_ransac(vertices, distance_threshold=1e-4, ransac_n=3, num_iterations=1000):
    """
    Fit a plane to a set of 3D vertices using RANSAC.
    Returns: plane_model (a, b, c, d) where ax + by + cz + d = 0
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vertices)
    plane_model, _ = pc.segment_plane(distance_threshold=distance_threshold,
                                      ransac_n=ransac_n,
                                      num_iterations=num_iterations)
    return plane_model  # a, b, c, d

def project_points_onto_plane(points, plane_model):
    """
    Projects a set of 3D points onto a given plane.
    plane_model: (a, b, c, d)
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    distances = (points @ normal) + d
    projected = points - np.outer(distances, normal)
    return projected

def flatten_cluster_vertices(mesh, labels, target_label):
    """
    Use RANSAC to fit a plane to a specific cluster and project its vertices onto that plane.
    """
    # Get all face indices with the target label
    face_indices = np.where(labels == target_label)[0]
    if len(face_indices) == 0:
        return mesh.vertices  # no update

    # Extract vertices used by those faces
    face_vertex_indices = mesh.faces[face_indices].flatten()
    unique_vertex_indices = np.unique(face_vertex_indices)
    cluster_vertices = mesh.vertices[unique_vertex_indices]

    # Fit RANSAC plane
    plane_model = fit_plane_ransac(cluster_vertices)

    # Project vertices onto the plane
    projected_vertices = project_points_onto_plane(cluster_vertices, plane_model)

    # Update mesh.vertices in place
    updated_vertices = mesh.vertices.copy()
    updated_vertices[unique_vertex_indices] = projected_vertices
    return updated_vertices

def process_mesh_with_ransac(input_path):
    # Load mesh
    mesh = trimesh.load_mesh(input_path)

    # Cluster faces by normal
    face_normals = mesh.face_normals
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(face_normals)

    # Identify +X, -X, +Y, -Y direction clusters
    axis_dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    selected_clusters = []

    for axis in axis_dirs:
        # Dot product with each cluster centroid
        similarities = kmeans.cluster_centers_ @ axis
        best_cluster = np.argmax(similarities)
        selected_clusters.append(best_cluster)

    # Iteratively flatten the mesh by projecting selected clusters
    current_vertices = mesh.vertices.copy()
    for label in tqdm(selected_clusters, desc="Flattening faces"):
        mesh.vertices = current_vertices
        current_vertices = flatten_cluster_vertices(mesh, labels, label)

    # Update mesh with final flattened vertices
    mesh.vertices = current_vertices

    # Convert to Open3D for visualization
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices.copy()),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    o3d_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([o3d_mesh])

# Run
input_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl"
process_mesh_with_ransac(input_path)
