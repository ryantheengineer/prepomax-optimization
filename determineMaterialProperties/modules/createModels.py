# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:45:07 2025

@author: Ryan.Larson

Create multibody STL files given a base STL file (the flexural specimen) and
data in a CSV file for where to place the specimen and the cylindrical
supports.
"""

import numpy as np
import pandas as pd
import trimesh
from trimesh import intersections
import os
from scipy.optimize import minimize_scalar, minimize
from trimesh.intersections import mesh_plane
from trimesh.transformations import rotation_matrix

############################## ASSUMPTIONS ####################################
# 1. Flexural sample STLs are originally oriented with their primary axis along
#    X, their secondary axis along Y, and tertiary axis along Z
# 2. 
###############################################################################

def load_flexural_specimen(stl_filepath):
    flex_mesh = trimesh.load_mesh(stl_filepath)
    return flex_mesh

def get_top_surface_bbox(flex_mesh):
    """
    Get the bounding box of the flex mesh's top surface.
    Returns the max Z coordinate (highest point of the top surface).
    """
    
    # Get the axis-aligned bounding box
    aabb_min, aabb_max = flex_mesh.bounds  # min (x, y, z) and max (x, y, z)
    
    # Compute bounding box dimensions
    aabb_size = aabb_max - aabb_min
    
    zmax = aabb_size[2]
    
    return aabb_max[2]

def get_bottom_surface_bbox(flex_mesh):
    """
    Get the bounding box of the flex mesh's bottom surface.
    Returns the min Z coordinate (lowest point of the bottom surface).
    """
    # Get the axis-aligned bounding box
    aabb_min, aabb_max = flex_mesh.bounds  # min (x, y, z) and max (x, y, z)
    
    # Compute bounding box dimensions
    aabb_size = aabb_max - aabb_min
    return aabb_min[2]

def position_flex_mesh(flex_mesh, l_pos):
    aabb_min, aabb_max = flex_mesh.bounds
    
    diff = aabb_min[0] - l_pos
    
    flex_mesh.apply_translation([-diff, 0, 0])
    
def create_anvil(flex_mesh_top):
    d_anvil = 10.0
    h_anvil = 20.0
    spacing = 0.0
    
    # Create anvil cylinder
    anvil = trimesh.creation.cylinder(
        radius=d_anvil/2,
        height=h_anvil,
        sections=32
    )
    
    # Rotate the cylinder (rotates about center?)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,  # 90-degree rotation
        direction=[1, 0, 0],  # Rotate around the Y-axis
        )
    anvil.apply_transform(rotation_matrix)
    
    # Move the anvil up so it is above the top of the flex_mesh specimen model
    anvil.apply_translation([0, 0, flex_mesh_top + d_anvil/2 + spacing])
    
    return anvil

def create_supports(flex_mesh_bottom, l_support_x, r_support_x):
    d_support = 10.0
    h_support = 20.0
    spacing = 0.0
    
    # Create generic support
    cylinder = trimesh.creation.cylinder(
        radius=d_support/2,
        height=h_support,
        sections=32
    )
    
    # Rotate the cylinder (rotates about center?)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,  # 90-degree rotation
        direction=[1, 0, 0],  # Rotate around the Y-axis
        )
    cylinder.apply_transform(rotation_matrix)
    
    # Make copies of cylinder and move them
    l_support = cylinder.copy()
    r_support = cylinder.copy()
    l_support.apply_translation([l_support_x, 0, flex_mesh_bottom - d_support/2 - spacing])
    r_support.apply_translation([r_support_x, 0, flex_mesh_bottom - d_support/2 - spacing])
    
    return l_support, r_support

def ensure_normals_outward(mesh: trimesh.Trimesh, mesh_name=""):
    if not mesh.is_watertight:
        print(f"Warning: Mesh '{mesh_name}' is not watertight. Normal orientation may be unreliable.")
    
    # Attempt to fix normals
    mesh.fix_normals()

    # Optional check
    if not mesh.is_winding_consistent:
        print(f"Warning: Mesh '{mesh_name}' has inconsistent winding after fixing normals.")

    return mesh

def rotate_and_position_flex_mesh(flex_mesh, angle_deg, target_x):
    mesh = flex_mesh.copy()
    angle_rad = np.radians(angle_deg)
    
    rot = rotation_matrix(angle_rad, [0, 1, 0], point=[0, 0, 0])
    mesh.apply_transform(rot)
    
    min_x = mesh.bounds[0][0]
    shift_x = target_x - min_x
    mesh.apply_translation([shift_x, 0, 0])
    return mesh

def slice_distance(flex_mesh, support_center, support_axis, radius, angle_deg):
    """
    Slices the flex_mesh using a plane that fully intersects the support axis.
    The plane is rotated by angle_deg around the support axis.
    
    - support_center: a point on the axis
    - support_axis: a 3D vector (does not need to be normalized)
    - angle_deg: rotation angle about the support axis
    - radius: only keep slice points within this radial distance from axis center
    """
    angle_rad = np.radians(angle_deg)
    axis = support_axis / np.linalg.norm(support_axis)

    # Define an arbitrary vector not parallel to the axis to help form the plane
    if np.allclose(axis, [1, 0, 0]):
        v = np.array([0, 0, 1])
    else:
        v = np.array([1, 0, 0])

    # Create a normal vector perpendicular to the axis (initial slicing plane)
    base_normal = np.cross(axis, v)
    base_normal /= np.linalg.norm(base_normal)

    # Rotate this normal vector around the axis
    rot = rotation_matrix(angle_rad, axis, point=support_center)
    rotated_normal = (rot[:3, :3] @ base_normal)

    # Perform mesh slicing    
    slice_result = intersections.mesh_plane(flex_mesh, plane_normal=rotated_normal, plane_origin=support_center)

    # Case 1: no result
    if slice_result is None:
        return 1e6
    
    # Case 2: returned a path or multiple
    if isinstance(slice_result, trimesh.path.Path3D):
        if len(slice_result.entities) == 0:
            return 1e6
    
    elif isinstance(slice_result, list):
        if all(isinstance(x, trimesh.path.Path3D) and len(x.entities) == 0 for x in slice_result):
            return 1e6
    
    # Case 3: raw array of points (e.g., vertices)
    elif isinstance(slice_result, np.ndarray):
        if len(slice_result) == 0:
            return 1e6

    # Collect slice points    
    points = slice_result.reshape(-1, 3)
    
    # Distance from support axis (use vector projection to axis)
    vecs = points - support_center    
    valid = np.abs(vecs[:,0]) <= radius
    if not np.any(valid):
        return 1e5  # Constraint not satisfied
    
    # Calculate the distance in the XZ plane from the support center to the points,
    # minus the support radius
    valid_points = points[valid]
    xz_points = valid_points[:, [0,2]]
    distances = np.linalg.norm(xz_points - support_center[[0, 2]], axis=1)
    # distances -= radius

    return np.min(np.abs(distances))

def optimize_support_angle(flex_mesh, support_center, support_axis, radius):
    result = minimize_scalar(lambda angle: slice_distance(flex_mesh, support_center, support_axis, radius, angle),
                             bounds=(0, 180), method='bounded')
    # print(f'Minimum distance found:\t{result.fun}')
    return result.fun  # Minimum distance

def optimize_flex_and_supports(flex_mesh, l_x, r_x, radius, target_x):
    def objective(vars):
        angle_deg, support_z = vars
        mesh = rotate_and_position_flex_mesh(flex_mesh, angle_deg, target_x)

        # Translate supports up to support_z
        l_center = np.array([l_x, 0, support_z])
        r_center = np.array([r_x, 0, support_z])
        
        support_axis = np.array([0, 1, 0])

        d_l = optimize_support_angle(mesh, l_center, support_axis, radius)
        d_r = optimize_support_angle(mesh, r_center, support_axis, radius)
        return d_l + d_r  # Total error

    result = minimize(objective,
                      x0=[0.0, flex_mesh.bounds[0][2] - 10],
                      bounds=[(-10, 10), (flex_mesh.bounds[0][2] - 20, flex_mesh.bounds[0][2] + 20)],
                      method='L-BFGS-B')
    print(f'\nAngle:\t{result.x[0]}')
    print(f'Supports Z height:\t{result.x[1]}')
    print(f'Total spacing:\t{result.fun}')
    return result



def create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder):
    df_test_data = pd.read_excel(test_data_filepath)
    # scanned_meshes_folder = os.path.join(baseFolder, "scanned_meshes")
    # prepared_meshes_folder = os.path.join(baseFolder, "prepared_meshes")
    
    
    os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # Parameters
    support_offset = 0.01
    overlap_allow = 1e-5
    step_size = 1e-4
    max_steps = int(5e4)
    params = {"support_offset": support_offset,
              "overlap_allow": overlap_allow,
              "step_size": step_size,
              "max_steps": max_steps}
    
    # Iterate through df_test_data and create the necessary multibody STL files for simulation
    for index, row in df_test_data.iterrows():
        filename = row["filename"]
        print(f'\nBuilding model for {filename} Test {row["Test_Num"]}')
        stl_filepath = os.path.join(aligned_meshes_folder, filename)
        # stl_filepath = df_test_data.loc[i, "filename"]
        flex_mesh = load_flexural_specimen(stl_filepath)
        # position_flex_mesh(flex_mesh, row["L_Edge_Specimen_X"])
        # flex_mesh = ensure_normals_outward(flex_mesh, mesh_name="flex_mesh")
        
        flex_mesh = ensure_normals_outward(flex_mesh, mesh_name="flex_mesh")
        
        # Run optimization
        result = optimize_flex_and_supports(
            flex_mesh,
            row["L_Support_X"],
            row["R_Support_X"],
            radius=5.0,  # Support radius
            target_x=row["L_Edge_Specimen_X"]
        )
        
        # Apply best transform to flex_mesh
        angle_opt, support_z_opt = result.x
        flex_mesh = rotate_and_position_flex_mesh(flex_mesh, angle_opt, row["L_Edge_Specimen_X"])
        
        flex_mesh_top = get_top_surface_bbox(flex_mesh)
        flex_mesh_bottom = get_bottom_surface_bbox(flex_mesh)
        anvil = create_anvil(flex_mesh_top)
        l_support_x = row["L_Support_X"]
        r_support_x = row["R_Support_X"]
        # l_support, r_support = create_supports(flex_mesh_bottom, l_support_x, r_support_x)
        l_support, r_support = create_supports(support_z_opt, l_support_x, r_support_x)
        
        # Ensure outward normals for cylinder bodies
        anvil = ensure_normals_outward(anvil, mesh_name="anvil")
        l_support = ensure_normals_outward(l_support, mesh_name="left_support")
        r_support = ensure_normals_outward(r_support, mesh_name="right_support")
        
        # Combine meshes
        all_meshes = [flex_mesh, anvil, l_support, r_support]
        merged_mesh = trimesh.util.concatenate(all_meshes)
        
        output_filename = f"{os.path.splitext(filename)[0]}_Test{row['Test_Num']}_prepared{os.path.splitext(filename)[1]}"
        output_filepath = os.path.join(prepared_meshes_folder, output_filename)
        merged_mesh.export(output_filepath)
        print(f'File exported to {output_filepath}')
    

if __name__ == "__main__":
    # test_data_filepath = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/test_data.csv"
    # baseFolder = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/"
    test_data_filepath = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    aligned_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder)
    
    
    
    # test_data_filepath = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/data/test_data.csv"
    # df_test_data = pd.read_csv(test_data_filepath)
    
    # baseFolder = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/"
    # scanned_meshes_folder = os.path.join(baseFolder, "scanned_meshes")
    # prepared_meshes_folder = os.path.join(baseFolder, "prepared_meshes")
    
    # os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # # Iterate through df_test_data and create the necessary multibody STL files for simulation
    # for index, row in df_test_data.iterrows():
    #     filename = row["filename"]
    #     stl_filepath = os.path.join(scanned_meshes_folder, filename)
    #     # stl_filepath = df_test_data.loc[i, "filename"]
    #     flex_mesh = load_flexural_specimen(stl_filepath)
    #     position_flex_mesh(flex_mesh, row["L_Edge_Specimen_X"])
    #     flex_mesh_top = get_top_surface_bbox(flex_mesh)
    #     flex_mesh_bottom = get_bottom_surface_bbox(flex_mesh)
    #     anvil = create_anvil(flex_mesh_top)
    #     l_support_x = row["L_Support_X"]
    #     r_support_x = row["R_Support_X"]
    #     # l_support_x = df_test_data.loc[i, "L_Support_X"]
    #     # r_support_x = df_test_data.loc[i, "R_Support_X"]
    #     l_support, r_support = create_supports(flex_mesh_bottom, l_support_x, r_support_x)
        
    #     # scene = trimesh.Scene()
        
    #     # scene.add_geometry(flex_mesh)
    #     # scene.add_geometry(anvil)
    #     # scene.add_geometry(l_support)
    #     # scene.add_geometry(r_support)
        
    #     # # axes = trimesh.creation.axis(origin_size=0.2, axis_length=20)
    #     # # scene.add_geometry(axes)
        
    #     # scene.show()
        
    #     # Combine meshes
    #     all_meshes = [flex_mesh, anvil, l_support, r_support]
    #     merged_mesh = trimesh.util.concatenate(all_meshes)
        
    #     output_filename = f"{os.path.splitext(filename)[0]}_Test{row['Test_Num']}_prepared{os.path.splitext(filename)[1]}"
    #     output_filepath = os.path.join(prepared_meshes_folder, output_filename)
    #     merged_mesh.export(output_filepath)