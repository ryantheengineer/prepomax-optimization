# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:08:33 2025

@author: Ryan.Larson
"""

import bpy
import sys
import time
from pathlib import Path

def main():
    args = sys.argv
    script_args = args[args.index("--") + 1:]  # Everything after '--'
    input_path = str(Path(script_args[0]).resolve())
    output_path = str(Path(script_args[1]).resolve())
    print("\nRunning quad_remesh.py", flush=True)
    
    print(f"input_path:\t{input_path}", flush=True)
    print(f"output_path:\t{output_path}", flush=True)
    
    # # Enable the Quad Remesher add-on
    # print("\nEnabling Quad Remesher add-on...", flush=True)
    # try:
    #     bpy.ops.preferences.addon_enable(module="qremesher")
    #     print("Quad Remesher add-on enabled successfully", flush=True)
    # except Exception as e:
    #     print(f"Warning: Could not enable Quad Remesher add-on: {e}", flush=True)
    #     # Try alternative module name
    #     try:
    #         bpy.ops.preferences.addon_enable(module="QuadRemesher")
    #         print("Quad Remesher add-on enabled with alternative name", flush=True)
    #     except Exception as e2:
    #         print(f"Error: Failed to enable Quad Remesher add-on: {e2}", flush=True)
    #         return
    
    # Clear scene
    print("\nClearing scene...", flush=True)
    bpy.ops.wm.read_homefile(use_empty=True)

    # Import model
    print("\nImporting model...", flush=True)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    bpy.ops.wm.stl_import(filepath=input_path)
    
    # Ensure we have an object selected
    if not bpy.context.selected_objects:
        raise RuntimeError("No objects were imported or selected")
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    print(f"Imported object: {obj.name}", flush=True)
    print(f"Object has {len(obj.data.vertices)} vertices", flush=True)

    # Enter Edit mode briefly to ensure mesh data is accessible
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply Quad Remesher
    print("\nRunning Quad Remesher...", flush=True)
    
    # Store original object count to detect when remeshing is complete
    original_objects = set(bpy.data.objects.keys())
    
    try:
        # Ensure we're in the right context
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Call the remesher
        result = bpy.ops.qremesher.remesh()
        print(f"Remesh operation result: {result}", flush=True)
        
        # Wait for the remeshing to complete
        print("Waiting for remeshing to complete...", flush=True)
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Update the scene
            bpy.context.view_layer.update()
            
            # Check if new objects have been created
            current_objects = set(bpy.data.objects.keys())
            new_objects = current_objects - original_objects
            
            if new_objects:
                print(f"New objects detected: {new_objects}", flush=True)
                break
                
            # Also check if the original object has been modified significantly
            if hasattr(obj.data, 'vertices') and len(obj.data.vertices) > 0:
                # If we still have the original object, wait a bit more
                time.sleep(1)
            else:
                break
        else:
            print("Warning: Remeshing timeout - proceeding anyway", flush=True)
            
    except Exception as e:
        print(f"Error during remeshing: {e}", flush=True)
        # Try to continue anyway in case the remeshing partially worked
    
    # Give additional time for any background processes
    time.sleep(2)
    
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    # Find the remeshed object
    print("\nFinding remeshed object...", flush=True)
    current_objects = list(bpy.data.objects)
    visible_objects = [obj for obj in current_objects if obj.visible_get() and obj.type == 'MESH']
    
    print(f"Found {len(visible_objects)} visible mesh objects", flush=True)
    for i, obj in enumerate(visible_objects):
        print(f"  Object {i}: {obj.name} with {len(obj.data.vertices)} vertices", flush=True)
    
    if len(visible_objects) == 0:
        raise RuntimeError("No visible mesh objects found after remeshing")
    
    # Use the object with the most vertices (likely the remeshed one)
    # or if there's only one, use that
    if len(visible_objects) == 1:
        remeshed_obj = visible_objects[0]
    else:
        # Find the object that's most likely the remeshed result
        # This could be the one with "remesh" in the name or the largest one
        remeshed_candidates = [obj for obj in visible_objects if "remesh" in obj.name.lower()]
        if remeshed_candidates:
            remeshed_obj = remeshed_candidates[0]
        else:
            # Fall back to the object with the most vertices
            remeshed_obj = max(visible_objects, key=lambda x: len(x.data.vertices))
    
    print(f"Selected remeshed object: {remeshed_obj.name}", flush=True)
    
    remeshed_obj.select_set(True)
    bpy.context.view_layer.objects.active = remeshed_obj
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export
    print("\nExporting remeshed object...", flush=True)
    bpy.ops.wm.stl_export(filepath=output_path, check_existing=False)
    
    # Verify export
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        print(f"Export successful: {output_path} ({file_size} bytes)", flush=True)
    else:
        raise RuntimeError(f"Export failed: {output_path} not found")
    
    print("\nCOMPLETE", flush=True)

if __name__ == "__main__":
    main()