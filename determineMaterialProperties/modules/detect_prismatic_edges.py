# -*- coding: utf-8 -*-
"""
Created on Thu May  1 07:23:39 2025

@author: Ryan.Larson
"""

import numpy as np
import trimesh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import open3d as o3d
from tqdm import tqdm

input_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/MeshBody1_Edited.stl"
output_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/scans/Test1/flattened_edge_planes.stl"

# === Hardcoded path to input mesh ===
# input_path = "your_model.stl"

def load_mesh_trimesh(path):
    print(f"Loading mesh from: {path}")
    return trimesh.load_mesh(path, force='mesh')

def cluster_face_normals(mesh, n_clusters=6):
    print("Computing PCA and clustering on face normals...")

    normals = mesh.face_normals
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(normals)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reduced)

    return labels

def trimesh_to_open3d(mesh, face_labels):
    print("Converting mesh to Open3D format...")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)

    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Assign a different color to each face cluster
    n_clusters = len(set(face_labels))
    colormap = plt_colormap(n_clusters)

    face_colors = np.array([colormap[label] for label in face_labels])
    vertex_colors = np.zeros_like(vertices)

    # Paint each vertex by averaging face colors (approximate visualization)
    print("Assigning vertex colors...")
    count = np.zeros(len(vertices))
    for i, face in enumerate(triangles):
        for vi in face:
            vertex_colors[vi] += face_colors[i]
            count[vi] += 1
    vertex_colors = (vertex_colors.T / np.maximum(count, 1)).T
    triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return triangle_mesh

def plt_colormap(n):
    # Generate n distinct colors using matplotlib colormap (but return as RGB)
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N)[:3] for i in range(n)]

def main():
    mesh = load_mesh_trimesh(input_path)
    labels = cluster_face_normals(mesh, n_clusters=12)  # adjust as needed

    o3d_mesh = trimesh_to_open3d(mesh, labels)

    print("Launching Open3D viewer...")
    o3d.visualization.draw_geometries([o3d_mesh])

if __name__ == "__main__":
    main()
