def get_anvil_reference_height(flex_mesh, anvil_x=0.0):
    """
    Get the specimen top surface height at the anvil contact location.
    Anvil is typically centered at X=0 (specimen center).
    """
    return get_specimen_height_at_x(flex_mesh, anvil_x, surface='top')

def get_support_reference_height(flex_mesh, l_support_x, r_support_x):
    """
    Get the appropriate Z coordinate for positioning both supports.
    Since supports must be at the same Z coordinate, we need to find
    the height that works for both contact points.
    
    Returns the Z coordinate where both supports should be placed.
    """
    # Get specimen bottom height at each support location
    l_specimen_bottom = get_specimen_height_at_x(flex_mesh, l_support_x, surface='bottom')
    r_specimen_bottom = get_specimen_height_at_x(flex_mesh, r_support_x, surface='bottom')
    
    # For supports to be at same Z and maintain contact, use the lower of the two
    # This ensures both supports can contact the specimen
    support_reference_z = min(l_specimen_bottom, r_specimen_bottom)
    
    print(f"Support reference heights - Left: {l_specimen_bottom:.3f}, Right: {r_specimen_bottom:.3f}")
    print(f"Using support reference Z: {support_reference_z:.3f}")
    
    return support_reference_z# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:45:07 2025

@author: Ryan.Larson

Create multibody STL files given a base STL file (the flexural specimen) and
data in a CSV file for where to place the specimen and the cylindrical
supports.

IMPROVED VERSION: Includes realistic specimen placement considering natural
resting position on supports.
"""

import numpy as np
import pandas as pd
import trimesh
import os

############################## ASSUMPTIONS ####################################
# 1. Flexural sample STLs are originally oriented with their primary axis along
#    X, their secondary axis along Y, and tertiary axis along Z
# 2. Specimens will naturally rest on two support points with some rotation
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

def get_specimen_height_at_x(flex_mesh, target_x, search_width=2.0, surface='bottom'):
    """
    Get the surface height of the specimen at a specific X coordinate.
    Uses a small search width to account for mesh discretization.
    
    Parameters:
    - surface: 'bottom' for minimum Z, 'top' for maximum Z
    """
    vertices = flex_mesh.vertices
    
    # Find vertices within the search window
    x_mask = np.abs(vertices[:, 0] - target_x) <= search_width
    if not np.any(x_mask):
        # If no vertices found in search window, expand search
        search_width *= 2
        x_mask = np.abs(vertices[:, 0] - target_x) <= search_width
    
    if not np.any(x_mask):
        # Still no vertices, return specimen bound
        if surface == 'bottom':
            return flex_mesh.bounds[0][2]
        else:
            return flex_mesh.bounds[1][2]
    
    # Get the appropriate Z coordinate in this region
    region_vertices = vertices[x_mask]
    if surface == 'bottom':
        return np.min(region_vertices[:, 2])
    else:
        return np.max(region_vertices[:, 2])

def find_natural_resting_orientation(flex_mesh, l_support_x, r_support_x, support_radius):
    """
    Calculate how the specimen would naturally rest on two support points.
    Returns the rotation angle needed and contact heights.
    """
    # Get bottom surface vertices (bottom 10% of specimen height)
    vertices = flex_mesh.vertices
    z_min, z_max = flex_mesh.bounds[0][2], flex_mesh.bounds[1][2]
    bottom_threshold = z_min + 0.1 * (z_max - z_min)
    bottom_vertices = vertices[vertices[:, 2] <= bottom_threshold]
    
    # Find potential contact points for each support
    l_contact_candidates = bottom_vertices[
        np.abs(bottom_vertices[:, 0] - l_support_x) <= support_radius * 1.5
    ]
    r_contact_candidates = bottom_vertices[
        np.abs(bottom_vertices[:, 0] - r_support_x) <= support_radius * 1.5
    ]
    
    # Get the lowest points for each support contact
    if len(l_contact_candidates) > 0:
        l_contact_z = np.min(l_contact_candidates[:, 2])
    else:
        l_contact_z = get_specimen_height_at_x(flex_mesh, l_support_x, surface='bottom')
    
    if len(r_contact_candidates) > 0:
        r_contact_z = np.min(r_contact_candidates[:, 2])
    else:
        r_contact_z = get_specimen_height_at_x(flex_mesh, r_support_x, surface='bottom')
    
    # Calculate required rotation angle
    z_diff = r_contact_z - l_contact_z
    x_span = r_support_x - l_support_x
    
    if abs(x_span) < 1e-6:  # Avoid division by zero
        rotation_angle = 0.0
    else:
        rotation_angle = np.arctan(z_diff / x_span)
    
    # Limit rotation to reasonable range (±15 degrees)
    max_rotation = np.pi / 12  # 15 degrees
    rotation_angle = np.clip(rotation_angle, -max_rotation, max_rotation)
    
    return rotation_angle, l_contact_z, r_contact_z

def improve_specimen_placement(flex_mesh, l_pos, l_support_x, r_support_x, 
                             support_radius, gap_offset=0.5):
    """
    Improved specimen placement that accounts for natural resting position.
    
    Parameters:
    - flex_mesh: The specimen mesh
    - l_pos: Left edge position (maintains specimen's left edge at this X coordinate)
    - l_support_x, r_support_x: X coordinates of support cylinders
    - support_radius: Radius of support cylinders
    - gap_offset: Small gap to maintain between specimen and supports (mm)
    """
    
    # Step 1: Initial positioning (maintain left edge constraint)
    aabb_min, aabb_max = flex_mesh.bounds
    diff = aabb_min[0] - l_pos
    flex_mesh.apply_translation([-diff, 0, 0])
    
    # Step 2: Find natural resting orientation
    rotation_angle, l_contact_z, r_contact_z = find_natural_resting_orientation(
        flex_mesh, l_support_x, r_support_x, support_radius
    )
    
    # Step 3: Apply rotation about Y-axis (only if significant)
    if abs(rotation_angle) > 0.001:  # ~0.06 degrees threshold
        # Get specimen centroid for rotation point
        centroid = flex_mesh.centroid
        
        # Create rotation matrix
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=rotation_angle,
            direction=[0, 1, 0],
            point=centroid
        )
        flex_mesh.apply_transform(rotation_matrix)
        
        # Re-adjust X position after rotation to maintain left edge constraint
        aabb_min_new, _ = flex_mesh.bounds
        x_drift = aabb_min_new[0] - l_pos
        flex_mesh.apply_translation([-x_drift, 0, 0])
        
        # Verify final position after rotation and re-adjustment
        final_aabb_min, _ = flex_mesh.bounds
        print(f"Applied rotation: {np.degrees(rotation_angle):.2f} degrees")
        print(f"Specimen min X after rotation: {final_aabb_min[0]:.3f} (target: {l_pos:.3f})")
    
    # Step 4: Final height adjustment
    # Get current specimen bottom after rotation
    current_bottom = flex_mesh.bounds[0][2]
    
    # Calculate where supports will be positioned (they'll be positioned later)
    # For now, assume supports are at Z=0 and we want specimen bottom at gap_offset above
    target_bottom_z = gap_offset
    z_adjustment = target_bottom_z - current_bottom
    
    flex_mesh.apply_translation([0, 0, z_adjustment])
    
    # Final verification of specimen position
    final_bounds = flex_mesh.bounds
    print(f"Final specimen min X: {final_bounds[0][0]:.3f} (target: {l_pos:.3f})")
    print(f"Final specimen bottom Z: {final_bounds[0][2]:.3f}")
    
    return flex_mesh

def position_flex_mesh(flex_mesh, l_pos):
    """
    Original positioning function - now replaced by improve_specimen_placement
    but kept for backward compatibility if needed.
    """
    aabb_min, aabb_max = flex_mesh.bounds
    diff = aabb_min[0] - l_pos
    flex_mesh.apply_translation([-diff, 0, 0])

def create_anvil(flex_mesh, anvil_x=0.0, anvil_gap=1.0):
    """
    Create anvil with specified gap above specimen top surface at anvil location.
    """
    d_anvil = 10.0
    h_anvil = 30.0
    
    # Get specimen top surface height at anvil contact location
    specimen_top_at_anvil = get_anvil_reference_height(flex_mesh, anvil_x)
    
    # Create anvil cylinder
    anvil = trimesh.creation.cylinder(
        radius=d_anvil/2,
        height=h_anvil,
        sections=32
    )
    
    # Rotate the cylinder to be horizontal (along Y-axis)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,  # 90-degree rotation
        direction=[1, 0, 0],  # Rotate around the X-axis
    )
    anvil.apply_transform(rotation_matrix)
    
    # Position anvil with specified gap above specimen top surface at contact location
    # Anvil bottom should be at specimen_top_at_anvil + anvil_gap
    anvil_bottom_z = specimen_top_at_anvil + anvil_gap
    anvil_center_z = anvil_bottom_z + d_anvil/2
    anvil.apply_translation([anvil_x, 0, anvil_center_z])
    
    print(f"Anvil positioned - Specimen top at X={anvil_x:.1f}: {specimen_top_at_anvil:.3f}, "
          f"Gap: {anvil_gap:.1f}, Anvil center Z: {anvil_center_z:.3f}")
    
    return anvil

def create_supports(flex_mesh, l_support_x, r_support_x, support_gap=0.5):
    """
    Create supports with specified gap below specimen bottom surface.
    Both supports are positioned at the same Z coordinate.
    """
    d_support = 10.0
    h_support = 30.0
    
    # Get the reference Z coordinate for support positioning
    support_reference_z = get_support_reference_height(flex_mesh, l_support_x, r_support_x)
    
    # Create generic support
    cylinder = trimesh.creation.cylinder(
        radius=d_support/2,
        height=h_support,
        sections=32
    )
    
    # Rotate the cylinder to be horizontal (along Y-axis)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,  # 90-degree rotation
        direction=[1, 0, 0],  # Rotate around the X-axis
    )
    cylinder.apply_transform(rotation_matrix)
    
    # Make copies of cylinder and move them
    l_support = cylinder.copy()
    r_support = cylinder.copy()
    
    # Position supports with specified gap below specimen bottom surface
    # Support top should be at support_reference_z - support_gap
    support_top_z = support_reference_z - support_gap
    support_center_z = support_top_z - d_support/2
    
    l_support.apply_translation([l_support_x, 0, support_center_z])
    r_support.apply_translation([r_support_x, 0, support_center_z])
    
    print(f"Supports positioned - Top Z: {support_top_z:.3f}, Center Z: {support_center_z:.3f}")
    print(f"Left support at X={l_support_x:.1f}, Right support at X={r_support_x:.1f}")
    
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
    
    os.makedirs(prepared_meshes_folder, exist_ok=True)
    
    # Configuration parameters
    support_gap = 0.01  # mm gap between specimen and supports
    anvil_gap = 0.01    # mm gap between specimen and anvil
    support_radius = 5.0  # mm - radius of support cylinders
    
    # Iterate through df_test_data and create the necessary multibody STL files for simulation
    for index, row in df_test_data.iterrows():
        filename = row["Quad Mesh File"]
        stl_filepath = os.path.join(aligned_meshes_folder, filename)
        
        print(f"\nProcessing {filename}...")
        
        # Load specimen
        flex_mesh = load_flexural_specimen(stl_filepath)
        
        # IMPROVED PLACEMENT: Use new function instead of simple positioning
        flex_mesh = improve_specimen_placement(
            flex_mesh, 
            l_pos=row["L_Edge_Specimen_X"],
            l_support_x=row["L_Support_X"],
            r_support_x=row["R_Support_X"],
            support_radius=support_radius,
            gap_offset=support_gap
        )
        
        flex_mesh = ensure_normals_outward(flex_mesh, mesh_name="flex_mesh")
        
        # Create anvil and supports using local surface heights (not bounding box extremes)
        # Anvil positioned at specimen center (X=0) by default
        anvil_x = 0.0  # Could be made configurable if needed
        anvil = create_anvil(flex_mesh, anvil_x, anvil_gap)
        
        l_support_x = row["L_Support_X"]
        r_support_x = row["R_Support_X"]
        l_support, r_support = create_supports(flex_mesh, l_support_x, r_support_x, support_gap)
        
        # Ensure outward normals for cylinder bodies
        anvil = ensure_normals_outward(anvil, mesh_name="anvil")
        l_support = ensure_normals_outward(l_support, mesh_name="left_support")
        r_support = ensure_normals_outward(r_support, mesh_name="right_support")

        # Combine meshes
        all_meshes = [flex_mesh, anvil, l_support, r_support]
        merged_mesh = trimesh.util.concatenate(all_meshes)
        
        # Generate output filename
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
    print(f"\n\nProcessed {len(df_test_data)} specimens with improved placement.")

if __name__ == "__main__":
    test_data_filepath = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    aligned_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder)