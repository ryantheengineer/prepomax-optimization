# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:49:09 2025

@author: Ryan.Larson
"""

import subprocess
import time
import shutil
import os

class CalculiXRunner:
    def __init__(self, working_directory, ccx_executable, pmx_file=None, geo_source_file=None, geo_target_file=None, params=None):
        self.working_directory = os.path.normpath(working_directory)
        self.ccx_executable = os.path.normpath(ccx_executable)
        # self.inp_files = inp_files
        self.pmx_file = os.path.normpath(pmx_file)
        self.geo_source_file = os.path.normpath(geo_source_file)  # Geometry source file (original geometry file the .pmx file was created with)
        self.geo_target_file = os.path.normpath(geo_target_file)  # Geometry target file (new geometry file the .pmx file will be regenerated with)
        self.params = params # NOTE: params must be in the form ["a=1.2; b=31.4"] where a and b are named parameters to be changed

    def run(self):
        # tStartTotal = time.time()
        # print("\n\n>> Running Operations Starting...")
        
        tStart = time.time()
        
        env = os.environ.copy()
        env["TMP"] = self.working_directory
        env["TEMP"] = self.working_directory
        
        if self.params:
            command = [self.ccx_executable,
                       "-r", self.pmx_file,
                       "-p", self.params,
                       "-g", "No",
                       "-w", self.working_directory,
                       "-x", "Yes",
                       ]
        else:
            command = [self.ccx_executable,
                       "-r", self.pmx_file,
                       "-g", "No",
                       "-w", self.working_directory,
                       "-x", "Yes",
                       ]
        
        try:
            result = subprocess.run(command, cwd=self.working_directory, check=True, capture_output=True, text=True)
            # result = subprocess.run(command, cwd=self.working_directory, check=True, capture_output=True, text=True)
            print(f">> CalculiX ran successfully for {self.pmx_file}!")
            # print(">> Output:")
            # print(result.stdout)  # CalculiX standard output
            # print(">> Errors (if any):")
            # print(result.stderr)  # CalculiX error output
        except subprocess.CalledProcessError as e:
            print(f">> Error while running CalculiX for {self.pmx_file}:")
            print(f"Result:\t{e.stdout}")
            # print(e.stderr)
            
        deltaTime = time.time() - tStart
        print(f"Elapsed time:\t{deltaTime} [s]")
    
    # def rename_basename(self, file_path, new_basename):
    #     """Renames the basename of a file path.
        
    #     Args:
    #       file_path: The original file path.
    #       new_basename: The new basename for the file.
    #     """
    #     dirname = os.path.dirname(file_path)
    #     new_path = os.path.join(dirname, new_basename)
    #     os.rename(file_path, new_path)
    
    # def copy_and_rename(self, selected_geometry_file, base_geometry_file):        
    #     ### Replace base_geometry.stl with the chosen file
    #     # Copy the selected_geometry_file
    #     shutil.copy(selected_geometry_)
        
    #     # Copy the file to the new location (overwrites the existing file)
    #     shutil.copy(src_file, dest_file)
    
    # 	# Rename the copied file
    #     self.rename_basename(dest_file, new_name)
    
    def regenerate_run(self):
        if self.pmx_file is None or self.geo_source_file is None or self.geo_target_file is None:
            raise Exception("Missing .pmx file or geometry source or target file")
        
        # Copy replacement geometry STL file to the base geometry file's
        # location and rename to match the base geometry's name (overwrites
        # the original)
        shutil.copy(self.geo_target_file, self.geo_source_file)
        
        # # Copy and rename the geo_source_file so it replaces geo_target_file
        # self.copy_and_rename(src_file=self.geo_source_file,
        #                 dest_file=self.geo_target_file,
        #                 new_name=os.path.basename(self.geo_source_file))
        
        self.run()