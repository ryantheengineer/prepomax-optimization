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
    
    # 180-degree rotation about +Z axis
    rotation_180_z = np.array([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 1]
    ])

    # Rotate aligned_positive by 180 degrees about Z
    aligned_vertices_v1_rot = aligned_vertices_v1 @ rotation_180_z
    mesh_v1_rot = mesh.copy()
    mesh_v1_rot.vertices = aligned_vertices_v1_rot
    mesh_v1_rot.export(f"{output_prefix}_aligned_positive_rot180z.stl")

    # Rotate aligned_negative by 180 degrees about Z
    aligned_vertices_v2_rot = aligned_vertices_v2 @ rotation_180_z
    mesh_v2_rot = mesh.copy()
    mesh_v2_rot.vertices = aligned_vertices_v2_rot
    export_path = f"{output_prefix}_aligned_negative_rot180z.stl"
    # print(f"Exporting to {export_path}")
    mesh_v2_rot.export(export_path)

    print(f"Processed: {os.path.basename(input_path)}")
    
def process_scans(input_folder, output_folder):
    stl_files = glob.glob(os.path.join(input_folder, "*.stl"))

    for stl_path in stl_files:
        base_name = os.path.splitext(os.path.basename(stl_path))[0]
        base_name = base_name.replace("_raw", "")
        output_prefix = os.path.join(output_folder, base_name)
        # print(f'Base name:\t{base_name}')
        # print(f'Output prefix:\t{output_prefix}')
        try:
            align_mesh_with_pca(stl_path, output_prefix)
        except Exception as e:
            print(f"Error processing {stl_path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch PCA alignment for STL meshes.")
    parser.add_argument("folder", type=str, help="Folder containing .stl files")
    parser.add_argument("--output", type=str, default="aligned", help="Output subfolder name")
    args = parser.parse_args()

    input_folder = args.folder
    output_folder = os.path.join(input_folder, args.output)
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)

    # stl_files = glob.glob(os.path.join(input_folder, "*.stl"))

    # for stl_path in stl_files:
    #     base_name = os.path.splitext(os.path.basename(stl_path))[0]
    #     base_name = base_name.replace("_raw", "")
    #     output_prefix = os.path.join(output_folder, base_name)
    #     # print(f'Base name:\t{base_name}')
    #     # print(f'Output prefix:\t{output_prefix}')
    #     try:
    #         align_mesh_with_pca(stl_path, output_prefix)
    #     except Exception as e:
    #         print(f"Error processing {stl_path}: {e}")
    
    process_scans(input_folder, output_folder)

### HOW TO RUN ###
# python prepareScans.py ./stl_meshes --output ./aligned_meshes