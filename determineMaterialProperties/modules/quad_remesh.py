# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:08:33 2025

@author: Ryan.Larson
"""

import bpy
import sys
from pathlib import Path

def main():
    args = sys.argv
    script_args = args[args.index("--") + 1:]  # Everything after '--'
    input_path = str(Path(script_args[0]).resolve())
    output_path = str(Path(script_args[1]).resolve())
    print("\nRunning quad_remesh.py", flush=True)
    
    print(f"input_path:\t{input_path}", flush=True)
    print(f"output_path:\t{output_path}", flush=True)
    
    # # Enable the add-on
    # bpy.ops.preferences.addon_enable(module="QuadRemesher")
    
    # Clear scene
    print("\nClearing scene...", flush=True)
    bpy.ops.wm.read_homefile(use_empty=True)

    # Import model
    print("\nImporting model...", flush=True)
    bpy.ops.wm.stl_import(filepath=input_path)
    obj = bpy.context.selected_objects[0]
    
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Apply Quad Remesher (example)
    print("\nRunning Quad Remesher...", flush=True)
    bpy.ops.qremesher.remesh()
    
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    # Find the new remeshed object
    print("\nFinding new remeshed object...", flush=True)
    visible_objects = [obj for obj in bpy.data.objects if obj.visible_get()]
    
    if len(visible_objects) != 1:
        raise RuntimeError(f"Expected 1 visible remeshed object, found {len(visible_objects)}.")
        
    remeshed_obj = visible_objects[0]
    remeshed_obj.select_set(True)
    bpy.context.view_layer.objects.active = remeshed_obj
    
    # Export
    print("\nExporting remeshed object...", flush=True)
    bpy.ops.wm.stl_export(filepath=output_path)
    
    print("\nCOMPLETE", flush=True)

if __name__ == "__main__":
    main()
