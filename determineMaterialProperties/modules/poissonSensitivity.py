# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:26:32 2025

@author: Ryan.Larson

Perform sensitivity studies on mesh size, Poisson ratio, etc.
"""

import yaml
import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import optimizeStiffnessAgainstData as opt
# from scipy.optimize import minimize_scalar
# import logging
import matplotlib.pyplot as plt
# import time
# import argparse
# from logger_config import setup_logger
# import logging
# from datetime import datetime
# from pathlib import Path
import numpy as np
import subprocess
from runModels import CalculiXRunner
import trimesh
from getResults import get_contact_force
from scipy.stats import linregress

def load_yaml(yaml_file):
    """Load YAML file."""
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file) or {}
    else:
        return None
    
def save_yaml(data, yaml_file):
    """Save YAML file."""
    with open(yaml_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

def estimate_preload_displacement(target_stl):
    mesh = trimesh.load_mesh(target_stl)

    # Ensure a list of disconnected components
    if isinstance(mesh, trimesh.Trimesh):
        parts = mesh.split(only_watertight=False)  # Returns a list of meshes
    elif isinstance(mesh, trimesh.Scene):
        parts = list(mesh.geometry.values())
    else:
        raise TypeError("Unsupported mesh type")

    # print(f"Loaded {len(parts)} parts")

    # for i, mesh in enumerate(parts):
    #     print(f"Part {i} center: {mesh.bounding_box.centroid}")
    
    specimen = parts[0]  # top body
    anvil = parts[1]
    l_support = parts[2]
    r_support = parts[3]
        
    def find_local_max_z(mesh, x_target, window=1.0):
        """
        Get the max Z value of the mesh within a narrow slab centered at x_target.
        """
        verts = mesh.vertices
        x = verts[:, 0]
        z = verts[:, 2]
        
        # Keep only vertices within the X window
        mask = (x > x_target - window/2) & (x < x_target + window/2)
        if mask.sum() == 0:
            print("No vertices found in this X-slice window.")
            return None
        local_max_z = z[mask].max()
        local_min_z = z[mask].min()
        return local_max_z, local_min_z
            
    
    # Find the local distance adjustment for the anvil
    x_anvil = anvil.bounding_box.centroid[0]
    z_anvil = anvil.bounds[0][2]            # min Z value of the anvil
    x_l_support = l_support.bounding_box.centroid[0]
    z_l_support = l_support.bounds[1][2]    # max Z value of l_support
    x_r_support = r_support.bounding_box.centroid[0]
    z_r_support = r_support.bounds[1][2]    # max Z value of r_support
    
    max_z_specimen = specimen.bounds[1][2]
    min_z_specimen = specimen.bounds[0][2]
    
    # max_specimen_anvil, _ = find_slice_min_max(x_anvil, specimen)
    # _, min_specimen_l_support = find_slice_min_max(x_l_support, specimen)
    # _, min_specimen_r_support = find_slice_min_max(x_r_support, specimen)
    
    max_specimen_anvil, _ = find_local_max_z(specimen, x_anvil)
    _, min_specimen_l_support = find_local_max_z(specimen, x_l_support)
    _, min_specimen_r_support = find_local_max_z(specimen, x_r_support)

    
    diff_anvil = z_anvil - max_specimen_anvil
    diff_l_support = np.abs(z_l_support - min_specimen_l_support)
    diff_r_support = np.abs(z_r_support - min_specimen_r_support)
    
    avg_diff_support = (diff_l_support + diff_r_support) / 2
    
    preload_step_displacement = 0.1
    preload_displacement = diff_anvil + avg_diff_support + preload_step_displacement
    print(f"Estimated preload_displacement:\t{preload_displacement}")
    
    return preload_displacement

def perform_poisson_sensitivity_study(poisson_values, yaml_file):
    moduli = []
    for poisson in list(poisson_values):
        
        print(f"\nRunning test with Poisson ratio {poisson}")
        # Load yaml config file
        data = load_yaml(yaml_file)
        data['poisson'] = float(poisson)
        save_yaml(data, yaml_file)
        
        results_directory = data['results_directory']
        ccx_executable = data['ccx_executable']
        disp_pmx_file = data['disp_pmx_file']
        geo_source_file = data['geo_source_file']
        geo_target_file = data['geo_target_file']
        
        modulus = 6000.0
        preload_displacement = estimate_preload_displacement(geo_target_file)
        total_displacement = preload_displacement + data['displacement']
        displacement_params = f"modulus={modulus}; poisson={poisson}; displacement={total_displacement}; preload_displacement={preload_displacement}"
        
        
        calculix_runner_displacement = CalculiXRunner(results_directory,
                                                      ccx_executable,
                                                      pmx_file=disp_pmx_file,
                                                      geo_source_file=geo_source_file,
                                                      geo_target_file=geo_target_file,
                                                      params=displacement_params,)
        calculix_runner_displacement.regenerate_run()
        
        
        # Get the contact force from the .dat file
        displacement_file = os.path.splitext(os.path.basename(disp_pmx_file))[0]
        dat_path = results_directory + '/' + displacement_file + \
            '.dat'    # Try this with a path library or os
        df_results = get_contact_force(dat_path)
        
        try:
            if df_results.empty:
                raise ValueError("FEA resulted in no data in .dat file")
            # logger.info(".dat file is valid. Proceeding with processing.")
            print(".dat file is valid. Proceeding with processing")
        except Exception as e:
            # logger.exception(f"Script failed: {e}")
            print(f"Script failed: {e}")
            raise

        # Use df_results and displacement to run regression and get the stiffness
        # X = df_results['UZ'].abs() - np.abs(df_results.loc[0, 'UZ'])
        reject_tol = 1e-5
        filtered_df_results = df_results[df_results['RZ'].notna() & (df_results['RZ'].abs() > reject_tol)]
        print(f"Results:\n{filtered_df_results}")
        X = filtered_df_results['UZ'].abs() - filtered_df_results['UZ'].abs().iloc[0]
        Y = filtered_df_results['RZ'].abs()
        # Y = df_results['FZ'].abs()
        regression_modulus, intercept, r_value, p_value, std_err = linregress(X, Y)
        # print(f"\nFEA regression modulus:\t{regression_modulus}")
        # print(f"R-squared:\t{r_value}")
        print(f"FEA regression modulus:\t{regression_modulus}")
        print(f"R-squared:\t{r_value}")
        
        
        # Save the regression modulus to a list of modulus values
        moduli.append([poisson, regression_modulus])
        
        print(f"\nCurrent sensitivity results:\n{moduli}")
        
        # Plot the sensitivity
        plt.figure(dpi=300)
        plt.plot(moduli[:,0], moduli[:,1])
        plt.xlabel('Poisson ratio')
        plt.ylabel('Regression Modulus (MPa)')
        plt.title(f'Poisson Sensitivity with Modulus {modulus} MPa')
        
if __name__ == "__main__":
    min_poisson = 0.1
    max_poisson = 0.6
    step = 0.1
    poisson_values = np.arange(start=min_poisson, stop=max_poisson, step=step)
    
    yaml_file = 'config.yaml'
    
    perform_poisson_sensitivity_study(poisson_values, yaml_file)
    