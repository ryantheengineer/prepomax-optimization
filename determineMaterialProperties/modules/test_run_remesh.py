# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:13:39 2025

@author: Ryan.Larson
"""

import subprocess
import time
from pathlib import Path

def run_remesh(input_path, output_path, timeout=600):
    """
    Run the quad remesh operation with improved error handling and timeout.
    
    Args:
        input_path: Path to input STL file
        output_path: Path for output STL file
        timeout: Maximum time to wait in seconds (default 10 minutes)
    """
    blender_exe = Path("C:/Program Files/Blender Foundation/Blender 4.3/blender.exe").resolve()
    remesh_script_path = Path("C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/modules/quad_remesh.py")
    
    # Verify paths exist
    if not blender_exe.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_exe}")
    
    if not remesh_script_path.exists():
        raise FileNotFoundError(f"Remesh script not found: {remesh_script_path}")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Starting remesh operation...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Timeout: {timeout} seconds")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            str(blender_exe),
            "--background",
            "--python", str(remesh_script_path),
            "--", str(input_path), str(output_path)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        encoding="utf-8",
        timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nProcess completed in {elapsed_time:.1f} seconds")
        
        print("\n" + "="*50)
        print("STDOUT:")
        print("="*50)
        print(result.stdout)
        
        print("\n" + "="*50)
        print("STDERR:")
        print("="*50)
        print(result.stderr)
        
        # Check if output file was created
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"\nOutput file created successfully: {output_path}")
            print(f"File size: {file_size} bytes")
        else:
            print(f"\nWarning: Output file not found: {output_path}")
        
        if result.returncode != 0:
            print(f"\nBlender process returned non-zero exit code: {result.returncode}")
            # Don't raise error immediately - check if output was still created
            if not Path(output_path).exists():
                raise RuntimeError(f"Blender subprocess failed with return code {result.returncode}")
            else:
                print("But output file exists, so operation may have succeeded despite error")
        
        return result
        
    except subprocess.TimeoutExpired:
        print(f"\nProcess timed out after {timeout} seconds")
        raise RuntimeError(f"Blender process timed out after {timeout} seconds")
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nProcess failed after {elapsed_time:.1f} seconds")
        raise e

if __name__ == "__main__":
    input_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/2 - Aligned Scans/X4_aligned.stl"
    output_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes/X4_quad.stl"
    
    try:
        run_remesh(input_path, output_path)
        print("\nRemesh operation completed successfully!")
    except Exception as e:
        print(f"\nRemesh operation failed: {e}")
        raise