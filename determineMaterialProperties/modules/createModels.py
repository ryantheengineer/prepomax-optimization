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
import os

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
    h_anvil = 30.0
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
    h_support = 30.0
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


def create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder):
    df_test_data = pd.read_excel(test_data_filepath)
    # scanned_meshes_folder = os.path.join(baseFolder, "scanned_meshes")
    # prepared_meshes_folder = os.path.join(baseFolder, "prepared_meshes")
    
    
    os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # Iterate through df_test_data and create the necessary multibody STL files for simulation
    for index, row in df_test_data.iterrows():
        filename = row["Quad Mesh File"]
        stl_filepath = os.path.join(aligned_meshes_folder, filename)
        # stl_filepath = df_test_data.loc[i, "filename"]
        flex_mesh = load_flexural_specimen(stl_filepath)
        position_flex_mesh(flex_mesh, row["L_Edge_Specimen_X"])
        flex_mesh = ensure_normals_outward(flex_mesh, mesh_name="flex_mesh")
        
        flex_mesh_top = get_top_surface_bbox(flex_mesh)
        flex_mesh_bottom = get_bottom_surface_bbox(flex_mesh)
        anvil = create_anvil(flex_mesh_top)
        l_support_x = row["L_Support_X"]
        r_support_x = row["R_Support_X"]
        l_support, r_support = create_supports(flex_mesh_bottom, l_support_x, r_support_x)
        
        # Ensure outward normals for cylinder bodies
        anvil = ensure_normals_outward(anvil, mesh_name="anvil")
        l_support = ensure_normals_outward(l_support, mesh_name="left_support")
        r_support = ensure_normals_outward(r_support, mesh_name="right_support")

        
        # Combine meshes
        all_meshes = [flex_mesh, anvil, l_support, r_support]
        merged_mesh = trimesh.util.concatenate(all_meshes)
        
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace("_positive","").replace("_negative","").replace("_quad","")
        output_filename = f"{base_name}_Test{row['Test_Num']}{os.path.splitext(filename)[1]}"
        output_filepath = os.path.join(prepared_meshes_folder, output_filename)
        merged_mesh.export(output_filepath)
        
        # Save the job name and add it to test_data.xlsx
        job_name = base_name.replace("_quad","") + f"_Test{row['Test_Num']}"
        df_test_data.loc[index, "Job Name"] = job_name
        
        # Save the test specific mesh file name and add it to test_data.xlsx
        df_test_data.loc[index, "Test Specific Mesh File"] = output_filepath
        
        print(f"{output_filename} complete")
        
    # Export changed dataframe
    df_test_data.to_excel(test_data_filepath, index=False)
    

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