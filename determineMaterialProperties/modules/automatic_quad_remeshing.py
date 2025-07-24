import bpy
import os
import sys
import time

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def remesh_with_quad_remesher(obj, timeout=300):
    """
    Remesh with proper waiting for the modal operation to complete
    timeout: maximum time to wait in seconds (default 5 minutes)
    """
    print(f"Starting remesh on object: {obj.name}")
    print(f"Original vertex count: {len(obj.data.vertices)}")
    print(f"Original face count: {len(obj.data.polygons)}")
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Store original mesh data for comparison
    original_vertex_count = len(obj.data.vertices)
    original_face_count = len(obj.data.polygons)
    
    # Start the remesh operation
    print("Starting Quad Remesher operation...")
    result = bpy.ops.qremesher.remesh()
    print(f"Remesh operator result: {result}")
    
    # Now we need to wait for the operation to complete
    # Quad Remesher works by exporting the mesh, processing it externally, then importing back
    print("Waiting for Quad Remesher to complete...")
    
    start_time = time.time()
    check_interval = 0.5  # Check every 0.5 seconds
    last_vertex_count = original_vertex_count
    stable_readings = 0
    required_stable_readings = 5  # Need 5 consecutive stable readings
    
    # Look for the temp file that Quad Remesher creates
    temp_fbx_path = "C:/Users/RYANLA~1.ROC/AppData/Local/Temp/Exoside/QuadRemesher/Blender/inputMesh.fbx"
    
    while time.time() - start_time < timeout:
        # Force Blender to process events and update
        bpy.app.handlers.depsgraph_update_post.clear()
        bpy.context.view_layer.update()
        
        # Check current mesh state
        current_vertex_count = len(obj.data.vertices)
        current_face_count = len(obj.data.polygons)
        elapsed = time.time() - start_time
        
        print(f"[{elapsed:.1f}s] Vertices: {current_vertex_count}, Faces: {current_face_count}")
        
        # Check if mesh has changed significantly (remesh completed)
        if current_vertex_count != original_vertex_count:
            print("Mesh has changed - checking stability...")
            
            if current_vertex_count == last_vertex_count:
                stable_readings += 1
                print(f"Stable reading {stable_readings}/{required_stable_readings}")
                
                if stable_readings >= required_stable_readings:
                    print("Mesh appears stable - remesh completed!")
                    break
            else:
                stable_readings = 0  # Reset if still changing
                print("Mesh still changing...")
            
            last_vertex_count = current_vertex_count
        
        # Also check if the temp file exists (indicates processing)
        if os.path.exists(temp_fbx_path):
            print(f"[{elapsed:.1f}s] Temp file exists - Quad Remesher is working...")
        
        time.sleep(check_interval)
    else:
        print(f"WARNING: Timeout after {timeout} seconds!")
        print("Remesh may not have completed properly")
    
    # Final verification
    final_vertex_count = len(obj.data.vertices)
    final_face_count = len(obj.data.polygons)
    
    print(f"\nFinal Results:")
    print(f"Original: {original_vertex_count} vertices, {original_face_count} faces")
    print(f"Final: {final_vertex_count} vertices, {final_face_count} faces")
    
    if final_vertex_count == original_vertex_count:
        print("WARNING: No change in vertex count - remesh may have failed!")
        return False
    else:
        change_percent = ((final_vertex_count - original_vertex_count) / original_vertex_count) * 100
        print(f"SUCCESS: Vertex count changed by {change_percent:.1f}%")
        return True

def quad_remesh_aligned_meshes(input_folder, output_folder):
    print(f"Processing folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return
    
    file_list = [f for f in os.listdir(input_folder) if f.endswith(".stl")]
    print(f"Found {len(file_list)} STL files to process")
    
    if not file_list:
        print("No STL files found!")
        return
    
    # Quad Remesher parameters
    print("Setting Quad Remesher parameters...")
    bpy.context.scene.qremesher.adaptive_size = 50
    bpy.context.scene.qremesher.target_count = 40000
    print(f"Adaptive size: {bpy.context.scene.qremesher.adaptive_size}")
    print(f"Target count: {bpy.context.scene.qremesher.target_count}")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    successful_count = 0
    
    for i, filename in enumerate(file_list, 1):
        print(f"\n{'='*50}")
        print(f"Processing file {i}/{len(file_list)}: {filename}")
        print(f"{'='*50}")
        
        suffix = "_quad"
        name, ext = os.path.splitext(filename)
        name = name.replace("_aligned", "")
        new_filename = name + suffix + ext
        
        clear_scene()
    
        # Import the model
        filepath = os.path.join(input_folder, filename)
        print(f"Importing: {filepath}")
        
        try:
            result = bpy.ops.wm.stl_import(filepath=filepath)
            print(f"Import result: {result}")
        except Exception as e:
            print(f"ERROR importing {filename}: {e}")
            continue
    
        # Get imported object
        if not bpy.context.selected_objects:
            print(f"ERROR: No objects imported from {filename}")
            continue
            
        obj = bpy.context.selected_objects[0]
        print(f"Processing object: {obj.name}")
    
        # Apply Quad Remesher with waiting
        success = remesh_with_quad_remesher(obj, timeout=300)  # 5 minute timeout per file
        
        if success:
            # Export the result
            output_path = os.path.join(output_folder, new_filename)
            print(f"Exporting to: {output_path}")
            
            try:
                result = bpy.ops.wm.stl_export(filepath=output_path)
                print(f"Export result: {result}")
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"SUCCESS: Exported {new_filename} ({file_size} bytes)")
                    successful_count += 1
                else:
                    print(f"ERROR: Output file not created: {output_path}")
            except Exception as e:
                print(f"ERROR exporting {new_filename}: {e}")
        else:
            print(f"SKIPPED: {filename} (remesh failed)")
    
    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE: {successful_count}/{len(file_list)} files processed successfully")
    print(f"{'='*50}")

def quad_remesh_single_specimen(input_path, output_path):
    print(f"Single file processing:")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file does not exist: {input_path}")
        return False
    
    # Set Quad Remesher parameters
    bpy.context.scene.qremesher.adaptive_size = 30
    bpy.context.scene.qremesher.target_count = 60000
    
    clear_scene()
    
    # Import the model
    print(f"Importing: {input_path}")
    result = bpy.ops.wm.stl_import(filepath=input_path)
    print(f"Import result: {result}")
    
    # Get imported object
    if not bpy.context.selected_objects:
        print("ERROR: No objects imported")
        return False
        
    obj = bpy.context.selected_objects[0]
    
    # Apply Quad Remesher with waiting
    success = remesh_with_quad_remesher(obj, timeout=300)
    
    if success:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export the result
        print(f"Exporting to: {output_path}")
        result = bpy.ops.wm.stl_export(filepath=output_path)
        print(f"Export result: {result}")
        
        if os.path.exists(output_path):
            print(f"SUCCESS: File exported ({os.path.getsize(output_path)} bytes)")
            return True
        else:
            print("ERROR: Output file not created")
            return False
    else:
        print("ERROR: Remesh failed")
        return False

if __name__ == "__main__":
    print("Starting Quad Remesher script")
    print(f"Blender version: {bpy.app.version_string}")
    
    # Check if we have command line arguments
    argv = sys.argv
    if "--" in argv:
        # Single file mode
        argv = argv[argv.index("--") + 1:]
        if len(argv) == 2:
            input_path, output_path = argv
            print("Running in single-file mode")
            success = quad_remesh_single_specimen(input_path, output_path)
            if not success:
                sys.exit(1)
        else:
            print("Usage: blender --background --python script.py -- <input_path> <output_path>")
            sys.exit(1)
    else:
        # Batch mode with hardcoded paths
        print("Running in batch mode")
        input_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/2 - Aligned Scans"
        output_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
        
        quad_remesh_aligned_meshes(input_folder, output_folder)
    
    print("Script completed successfully")