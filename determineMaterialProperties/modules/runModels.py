# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:49:09 2025

@author: Ryan.Larson
"""

import subprocess
import time
import shutil

class CalculiXRunner:
    def __init__(self, working_directory, ccx_executable, inp_files, pmx_file=None, geo_source_file=None, geo_target_file=None, number_of_cores=4):
        self.working_directory = working_directory
        self.ccx_executable = ccx_executable
        self.inp_files = inp_files
        self.geo_source_file = geo_source_file
        self.geo_target_file = geo_target_file
        self.number_of_cores = number_of_cores

    def run(self):
        tStartTotal = time.time()
        # print("\n\n>> Running Operations Starting...")

        for input_file in self.inp_files:
            # print(f"\n>> Running CalculiX for input file: {input_file}")
            tStartCycle = time.time()

            command = [self.ccx_executable, "-f", input_file, "-t", str(self.number_of_cores)]
            try:
                result = subprocess.run(command, cwd=self.working_directory, check=True, capture_output=True, text=True)
                # print(f">> CalculiX ran successfully for {input_file}!")
                # print(">> Output:")
                # print(result.stdout)  # CalculiX standard output
                # print(">> Errors (if any):")
                # print(result.stderr)  # CalculiX error output
            except subprocess.CalledProcessError as e:
                print(f">> Error while running CalculiX for {input_file}:")
                print(e.stderr)

            deltaTimeCycle = time.time() - tStartCycle
            print(f'\n>> Elapsed time for this run = {deltaTimeCycle} [s]')
            # print("-------------------------------------------------------------------------------------------------------------------\n")

        deltaTimeTotal = time.time() - tStartTotal
        # print("\n****************************************\n>> All CalculiX run's have completed! <<\n****************************************")
        # print(f"\n>> Total elapsed time for runs = {deltaTimeTotal} [s] <<\n")
    
    def copy_and_rename(src_path, dest_path, new_name):
    	# Copy the file
    	shutil.copy(src_path, dest_path)
    
    	# Rename the copied file
    	new_path = f"{dest_path}/{new_name}"
    	shutil.move(f"{dest_path}/{src_path}", new_path)
    
    def regenerate_run(self):
        if self.pmx_file is None or self.geo_source_file is None or self.geo_target_file is None:
            raise Exception("Missing .pmx file or geometry source or target file")
            
        # Copy and rename the geo_source_file so it replaces geo_target_file
        os.path.splitext(os.path.basename(new_file_path))[0]
        copy_and_rename()
        
        command = [self.ccx_executable, "-f", self.pmx_file]