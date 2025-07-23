# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:00:03 2025

@author: Ryan.Larson

Takes test data and prepared specimen meshes, and creates the necessary job
folders and file structure to be uploaded to S3.
"""

from prepareScans import process_scans
# from automatic_quad_remeshing import quad_remesh_aligned_meshes
# from createModelsV2 import create_models
from define_test_setup_from_scan import create_models
import subprocess
import pandas as pd
import os
import shutil
import yaml

def create_jobs(poisson):
    # # PCA align the raw scan meshes
    # raw_scans_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans"
    # aligned_scans_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/2 - Aligned Scans"
    
    # print("\nProcessing raw scans - aligning them with PCA")
    # process_scans(raw_scans_folder, aligned_scans_folder)
    
    # # Quad remesh
    # quad_meshes_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/3 - Quad Meshes"
    # blender_exe = "C:/Program Files/Blender Foundation/Blender 4.3/blender.exe"
    # quad_remesh_script_path = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/modules/automatic_quad_remeshing.py"
    # # quad_remesh_aligned_meshes(aligned_scans_folder, quad_meshes_folder)
    # print("\nUsing quad remesher in Blender to prepare aligned meshes")
    # subprocess.run([blender_exe, "--background", "--python", quad_remesh_script_path])
    # print("Quad remesher processing complete\n")
    
    # # NOTE: Need to decide which of the quad mesh files are aligned correctly to match the test orientations
    # # Visualize the mesh files and accept them?
    # print("#"*60)
    # print("#"*60)
    # print("PAUSING HERE TO MANUALLY UPDATE test_data.xlsx with the files that give the correct orientation.")
    # print("#"*60)
    # print("#"*60)
    # input("\nPress ENTER when test_data.xlsx has been updated...")
    
    # # Create test-specific mesh files based on real test data
    # test_data_filepath = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    # prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    # print("\nCreating test-specific mesh files based on real test data")
    # create_models(test_data_filepath, quad_meshes_folder, prepared_meshes_folder)
    
    
    test_data_filepath = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/4 - Flexural Test Data/test_data.xlsx"
    scanned_fixtures_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Fixtures"
    scanned_specimens_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/1 - Raw Scans/Specimens"
    prepared_meshes_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes"
    
    create_models(test_data_filepath, scanned_fixtures_folder, scanned_specimens_folder, prepared_meshes_folder)
    
    
    # Create job folders with YAML config files and the test-specific meshes
    print("\nCreating job folders and necessary files")
    jobs_folder = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/7 - Jobs"
    df_test_data = pd.read_excel(test_data_filepath)
    os.makedirs(jobs_folder, exist_ok=True)
    
    for index, row in df_test_data.iterrows():
        job_name = row["Job Name"]
        
        # Create a folder in jobs_folder
        job_folder = jobs_folder + "/" + job_name
        # job_folder = os.path.join(jobs_folder, job_name)
        os.makedirs(job_folder, exist_ok=True)
        
        # Place the correct test mesh in the folder
        test_mesh = row["Test Specific Mesh File"]
        print(f"Copying {os.path.basename(test_mesh)} into {job_folder}")
        destination_test_mesh = os.path.join(job_folder, os.path.basename(test_mesh))
        shutil.copy2(test_mesh, destination_test_mesh)
        
        target_stiffness = row["Flexural Stiffness (N/mm)"]
        
        # Create the YAML parameters
        params = {'poisson': poisson,
                  'displacement': row["Displacement (mm)"],
                  'results_directory': 'C:/Users/Administrator/prepomax-optimization/determineMaterialProperties/output',
                  'ccx_executable': 'C:/Users/Administrator/prepomax-optimization/determineMaterialProperties/PrePoMax v2.2.0/PrePoMax.com',
                  'disp_pmx_file': 'C:/Users/Administrator/prepomax-optimization/determineMaterialProperties/pmx_files/v2/displacement_v2.pmx',
                  'geo_source_file': 'C:/Users/Administrator/prepomax-optimization/determineMaterialProperties/pmx_files/v2/base_geometry.stl',
                  'geo_target_file': f'C:/tmp/job/{os.path.basename(test_mesh)}',
                  'target_stiffness': target_stiffness,
                  'log_file': f'{job_name}.log',
                  'job_name': job_name,
                  'opt_working_directory': 'C:/Users/Administrator/prepomax-optimization/determineMaterialProperties/modules',
            }
        
        # Create the YAML file
        yaml_filename = f"{job_name}.yaml"
        yaml_filepath = os.path.join(job_folder, yaml_filename)
        with open(yaml_filepath, 'w') as file:
            yaml.dump(params, file, default_flow_style=False)
            
    print("\nALL PROCESSING COMPLETE")
        

if __name__ == "__main__":
    poisson = 0.3
    
    create_jobs(poisson)
    