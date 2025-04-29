import trimesh
import numpy as np
from sklearn.decomposition import PCA
import os
import glob

def align_mesh_with_pca(input_path, output_prefix):
    mesh = trimesh.load(input_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{input_path} is not a valid mesh.")

    # Center the mesh at the origin
    mesh.vertices -= mesh.centroid

    # PCA on vertex positions
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    axes = pca.components_

    # Sort by explained variance (descending)
    order = np.argsort(pca.explained_variance_)[::-1]
    ordered_axes = axes[order]

    # +X, +Y, +Z target alignment
    target_axes_v1 = np.eye(3)

    rotation_matrix_v1 = ordered_axes.T @ target_axes_v1
    aligned_vertices_v1 = mesh.vertices @ rotation_matrix_v1
    mesh_v1 = mesh.copy()
    mesh_v1.vertices = aligned_vertices_v1
    mesh_v1.export(f"{output_prefix}_aligned_positive.stl")

    # -X, -Y, -Z target alignment
    rotation_matrix_v2 = ordered_axes.T @ -target_axes_v1
    aligned_vertices_v2 = mesh.vertices @ rotation_matrix_v2
    mesh_v2 = mesh.copy()
    mesh_v2.vertices = aligned_vertices_v2
    mesh_v2.export(f"{output_prefix}_aligned_negative.stl")

    print(f"Processed: {os.path.basename(input_path)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch PCA alignment for STL meshes.")
    parser.add_argument("folder", type=str, help="Folder containing .stl files")
    parser.add_argument("--output", type=str, default="aligned", help="Output subfolder name")
    args = parser.parse_args()

    input_folder = args.folder
    # output_folder = os.path.join(input_folder, args.output)
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)

    stl_files = glob.glob(os.path.join(input_folder, "*.stl"))

    for stl_path in stl_files:
        base_name = os.path.splitext(os.path.basename(stl_path))[0]
        output_prefix = os.path.join(output_folder, base_name)
        try:
            align_mesh_with_pca(stl_path, output_prefix)
        except Exception as e:
            print(f"Error processing {stl_path}: {e}")


### HOW TO RUN ###
# python prepareScans.py ./stl_meshes --output ./aligned_meshes