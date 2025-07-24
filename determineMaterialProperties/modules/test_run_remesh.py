# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:13:39 2025

@author: Ryan.Larson
"""

import subprocess
from pathlib import Path

def run_remesh(input_path, output_path):
    blender_exe = Path("C:/Program Files/Blender Foundation/Blender 4.3/blender.exe").resolve()
    remesh_script_path = Path("C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/modules/quad_remesh.py")
    
    result = subprocess.run([
        blender_exe,
        "--background",
        "--python", remesh_script_path,
        "--", input_path, output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError("Blender subprocess failed.")
        
if __name__ == "__main__":
    input_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/2 - Aligned Scans/X4_aligned.stl"
    output_path = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes/X4_quad.stl"
    
    run_remesh(input_path, output_path)