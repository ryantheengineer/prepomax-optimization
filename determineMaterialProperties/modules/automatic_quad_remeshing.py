# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:35:02 2025

@author: Ryan.Larson
"""

import bpy
import os
import sys

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def remesh_with_quad_remesher(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.qremesher.remesh()
#    # You may need to explore the exact operator name in your version
#    bpy.ops.quad_remesher.remesh(
#        use_target_quad_count=True,
#        target_quad_count=quad_remesher_params["target_quad_count"],
#        adaptive_size=quad_remesher_params["adaptive_size"],
#        preserve_sharp=quad_remesher_params["preserve_sharp"],
#        symmetry=quad_remesher_params["symmetry_axis"]
#    )

def quad_remesh_aligned_meshes(input_folder, output_folder):
    file_list = [f for f in os.listdir(input_folder) if f.endswith(".stl")]  # or .fbx, .stl, etc.
    
    # Quad Remesher parameters (these may differ depending on addon version)
    bpy.context.scene.qremesher.adaptive_size = 50
    bpy.context.scene.qremesher.target_count = 40000
    
    for filename in file_list:
        suffix = "_quad"
        name, ext = os.path.splitext(filename)
        name = name.replace("_aligned","")
        new_filename = name + suffix + ext
        
        clear_scene()
    
        # Import the model
        filepath = os.path.join(input_folder, filename)
        bpy.ops.wm.stl_import(filepath=filepath)
    
        # Get imported object (assuming one object per file)
        obj = bpy.context.selected_objects[0]
    
        # Apply Quad Remesher
        remesh_with_quad_remesher(obj)
    
        # Export the result
        output_path = os.path.join(output_folder, new_filename)
        bpy.ops.wm.stl_export(filepath=output_path)  # use correct format
        
def quad_remesh_single_specimen(input_path, output_path):
    
    clear_scene()
    
    # Import the model
    bpy.ops.wm.stl_import(filepath=input_path)
    
    # Get imported object
    obj = bpy.context.selected_objects[0]
    
    # Apply Quad Remesher
    remesh_with_quad_remesher(obj)
    
    # Export the result
    bpy.ops.wm.stl_export(filepath=output_path)
    
if __name__ == "__main__":
    # input_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/2 - Aligned Scans"
    # output_folder = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    
    # quad_remesh_aligned_meshes(input_folder, output_folder)
    
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] # Get all args after '--'
    
    if len(argv) != 2:
        print("Usage: blender --background --python automatic_quad_remeshing.py -- <input_path> <output_path>")
        sys.exit(1)
        
    input_path, output_path = argv
    quad_remesh_single_specimen(input_path, output_path)
