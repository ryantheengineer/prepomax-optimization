# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:15:10 2025

@author: Ryan.Larson
"""

from runModels import CalculiXRunner
from getResults import get_node_deformations as getNodeDef
import random
import os
import shutil



results_directory = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/output"
ccx_executable = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/PrePoMax v2.2.0/PrePoMax.com"
pmx_file = "E:/Rockwell Scans/Rockwell Material Sensitivity.pmx"

geo_source_file = "E:/Rockwell Scans/Elite 664460 Surface Model v2.step"

min_modulus = 77500.0
max_modulus = 12000.0

n_iterations = 10
for i in range(n_iterations):
    # Randomly generate material values
    modulus1 = random.uniform(min_modulus, max_modulus)
    modulus2 = random.uniform(min_modulus, max_modulus)
    modulus3 = random.uniform(min_modulus, max_modulus)

    params = f"modulus1={modulus1}; modulus2={modulus2}; modulus3={modulus3}"
    calculix_runner = CalculiXRunner(results_directory,
                                                  ccx_executable,
                                                  pmx_file=pmx_file,
                                                  geo_source_file=geo_source_file,
                                                  params=params,)
    calculix_runner.regenerate_run_no_geo_change()
    
    displacement_file = os.path.splitext(os.path.basename(pmx_file))[0]
    dat_path = results_directory + '/' + displacement_file + '.dat'    # Try this with a path library or os
    # df_results = get_contact_force(dat_path)
    
    
    
    # Log the results
    

# Based on the results, run a multiple regression on material properties and
# the resulting displacement

# Could also compare the results to the cases where all the material values
# are identical. This might give an idea of a rough error factor that is
# possible if using a single material simulation as opposed to a multi material
# simulation.