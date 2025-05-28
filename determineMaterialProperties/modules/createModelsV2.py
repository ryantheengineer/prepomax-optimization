"""
Fixed optimization with proper distance calculation and rotation sensitivity
Now with debug visualization improvements
"""

import pybullet as p
import pybullet_data
import time
import pandas as pd
import os
import math
import trimesh
import numpy as np
import matplotlib.pyplot as plt

def fix_normals(input_path, output_path):
    mesh = trimesh.load_mesh(input_path)
    if not mesh.is_watertight:
        print(f"⚠️ {os.path.basename(input_path)} is not watertight.")
    mesh.rezero()
    mesh.fix_normals()
    mesh.export(output_path)


def create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder):
    df_test_data = pd.read_excel(test_data_filepath)
    os.makedirs(prepared_meshes_folder, exist_ok=True)

    # Start the physics engine in GUI mode
    p.connect(p.DIRECT)
    # p.connect(p.GUI)
    
    # Let it settle
    hz = 240.0
    tstep = 1.0 / hz
    tsim = 2.0
    nsteps = int(tsim / tstep)
    p.setRealTimeSimulation(0)          # Turn off real-time mode
    p.setTimeStep(tstep)
    
    # Configure PyBullet visualization
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=0.3,
    #     cameraYaw=0,
    #     cameraPitch=-40,
    #     cameraTargetPosition=[0, 0, 0]
    # )
    p.setPhysicsEngineParameter(
        numSolverIterations=100,
        numSubSteps=4
    )

    for index, row in df_test_data.iterrows():
        if index > 0:
            continue
        filename = row["filename"]
        stl_filepath = os.path.join(aligned_meshes_folder, filename)
        
        # Fix normals and export to a temporary file
        fixed_stl_filepath = os.path.join(aligned_meshes_folder, f"fixed_{filename}")
        fix_normals(stl_filepath, fixed_stl_filepath)
        
        stl_filepath = fixed_stl_filepath
        
        mesh_scale = 1
        
        # Load STL with corrected normals
        specimen_col = p.createCollisionShape(p.GEOM_MESH, fileName=fixed_stl_filepath, meshScale=mesh_scale)
        specimen_vis = p.createVisualShape(p.GEOM_MESH, fileName=fixed_stl_filepath, meshScale=mesh_scale)
        
        # Create support cylinders
        cylinder_shape = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=5, height=50, meshScale=0.001)
        cylinder_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=5, length=50, meshScale=0.001)
        
        l_support_x = row["L_Support_X"]
        r_support_x = row["R_Support_X"]
        rot_x_90 = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        
        l_support = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cylinder_shape,
            baseVisualShapeIndex=cylinder_visual,
            basePosition=[l_support_x, 0, -5],
            baseOrientation=rot_x_90
        )
        
        r_support = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cylinder_shape,
            baseVisualShapeIndex=cylinder_visual,
            basePosition=[r_support_x, 0, -5],
            baseOrientation=rot_x_90
        )
        
        # Load STL mesh
        scale = 1
        mesh_scale = [scale, scale, scale]  # Adjust if mesh too large/small or invisible
        # specimen_col = p.createCollisionShape(p.GEOM_MESH, fileName=stl_filepath, meshScale=mesh_scale)
        specimen_col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=stl_filepath,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        specimen_vis = p.createVisualShape(p.GEOM_MESH, fileName=stl_filepath, meshScale=mesh_scale)
        print(f"\nProcessing: {filename}")
        print("Collision Shape ID:", specimen_col)
        print("Visual Shape ID:", specimen_vis)

        if specimen_col < 0 or specimen_vis < 0:
            print("⚠️ Failed to load mesh:", stl_filepath)
            continue

        # Temporarily create to get dimensions
        temp_id = p.createMultiBody(
            baseMass=100.0,
            baseCollisionShapeIndex=specimen_col,
            baseVisualShapeIndex=specimen_vis,
            basePosition=[0, 0, 5]
        )
        
        # Get AABB and calculate height in Y direction
        aabb_min, aabb_max = p.getAABB(temp_id)
        height_y = aabb_max[1] - aabb_min[1]
        shift_y = height_y / 2.0
        
        # Remove temp body
        p.removeBody(temp_id)
        
        # Now create the actual specimen, shifted in +Y
        specimen = p.createMultiBody(
            baseMass=100000.0,
            baseCollisionShapeIndex=specimen_col,
            baseVisualShapeIndex=specimen_vis,
            basePosition=[row["L_Edge_Specimen_X"], shift_y, 0.5],
        )
        
        # # Create a dummy anchor
        # anchor = p.createMultiBody(
        #     baseMass=0,
        #     baseCollisionShapeIndex=-1,
        #     baseVisualShapeIndex=-1,
        #     basePosition=[desired_x_position, 0, 10],
        #     useMaximalCoordinates=True
        # )
        
        for body_id in [specimen, l_support, r_support]:
            p.changeDynamics(body_id, -1,
                restitution=0.0,
                linearDamping=0.05,
                angularDamping=0.05,
                lateralFriction=0.3,
                contactStiffness=1e10,      # Optional: more compliant contact
                contactDamping=10000         # Optional: adds energy loss
            )
        
        yaw = 0
        p.resetDebugVisualizerCamera(
            cameraDistance=50,
            cameraYaw=yaw,
            cameraPitch=-2,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Optional: Add constraint logic here if you want the model aligned
        
        min_L_vals = []
        min_R_vals = []
        specimen_L_x = []
        
        # Before your simulation loop
        initial_pos, _ = p.getBasePositionAndOrientation(specimen)
        desired_x = initial_pos[0]
        k = 10000000  # spring constant, adjust for your simulation stiffness
        
        for t in range(nsteps):
            p.stepSimulation()
            
            # Get current specimen position
            pos, _ = p.getBasePositionAndOrientation(specimen)
            error_x = desired_x - pos[0]
            
            # Calculate corrective force along X
            force_x = k * error_x
            
            # Apply external force at specimen's center of mass (linkIndex = -1)
            p.applyExternalForce(objectUniqueId=specimen,
                                 linkIndex=-1,
                                 forceObj=[force_x, 0, 0],
                                 posObj=pos,
                                 flags=p.WORLD_FRAME)
            
            aabb_min, aabb_max = p.getAABB(specimen)
            min_x = aabb_min[0]
            specimen_L_x.append(min_x)
            
            # Support contacts
            L_contacts = p.getClosestPoints(bodyA=l_support, bodyB=specimen, distance=0.5)
            R_contacts = p.getClosestPoints(bodyA=r_support, bodyB=specimen, distance=0.5)
            if L_contacts:
                L_contactDistances = [L_contacts[i][8] for i in range(len(L_contacts))]
                min_L = min(L_contactDistances)
            else:
                min_L = np.nan
                
            if R_contacts:
                R_contactDistances = [R_contacts[i][8] for i in range(len(R_contacts))]
                min_R = min(R_contactDistances)
            else:
                min_R = np.nan
                
            min_L_vals.append(min_L)
            min_R_vals.append(min_R)
                
            # if (not np.isnan(min_L)) or (not np.isnan(min_R)):
            #     # Print with fixed column widths:
            #     # Left column width: 12 chars, right column width: 12 chars, right-aligned
            #     # Show 'N/A' if None
            #     print(f"L: {min_L if min_L is not None else 'N/A':>8}    R: {min_R if min_R is not None else 'N/A':>8}")
                
            
                
            time.sleep(tstep)
        
        # Print final pose
        final_pos, final_orn = p.getBasePositionAndOrientation(specimen)
        print("Final position:", final_pos)
        
        print(f"\nClosest L contact: {np.nanmin(min_L_vals)}")
        print(f"Closest R contact: {np.nanmin(min_R_vals)}")
        
        plt.figure(dpi=300)
        plt.plot(np.asarray(specimen_L_x)-desired_x)


if __name__ == "__main__":
    test_data_filepath = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    aligned_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    create_models(test_data_filepath, aligned_meshes_folder, prepared_meshes_folder)
