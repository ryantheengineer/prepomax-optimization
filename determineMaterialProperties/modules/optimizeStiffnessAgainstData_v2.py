# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:20:02 2025

@author: Ryan.Larson
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:20:07 2025

@author: Ryan.Larson

Script for finding necessary material properties to match physical tests, for
multiple specimens and multiple tests per specimen. Each specimen/test combo
is optimized to find the material property that most closely matches the
corresponding physical test, then it moves on to another specimen/test combo.
"""

# libs
import numpy as np
# import matplotlib.pyplot as plt
import os
from generatePreloadDisplacement import INPFileGenerator
from runINPFiles import CalculiXRunner
from getResults import get_node_deformations as getNodeDef
from getResults import get_contact_force
# from scipy.optimize import minimize_scalar
# import yaml
from scipy.stats import linregress
import time

optHistory = np.empty((0, 3))


def objective_fun(modulus, params):
    global optHistory

    tStart = time.time()

    print('\n\n')
    print('#'*60)
    print(f'Trying modulus: {modulus}')

    # Unpack params dictionary
    # FEA model parameters
    poisson = params['poisson']
    displacement = params['displacement']

    # Filepath parameters
    results_directory = params['results_directory']
    ccx_executable = params['ccx_executable']
    preload_pmx_file = params['preload_pmx_file']
    disp_pmx_file = params['disp_pmx_file']
    # Geometry source file (original geometry file the .pmx file was created with)
    geo_source_file = params['geo_source_file']
    # Geometry target file (new geometry file the .pmx file will be regenerated with)
    geo_target_file = params['geo_target_file']

    # Simulation parameters
    target_stiffness = params['target_stiffness']
    
    preload_file_path = params['preload_file_path']
    displacement_file_path = params['displacement_file_path']
    inp_directory = params['inp_directory']
    
    # global stiffnessHistory
    # global defHistory
    
    inpgen = INPFileGenerator(preload_file_path, displacement_file_path, inp_directory, ccx_executable)
    
    # Generate the preload inp file
    file_index = 0
    preload_file_name = inpgen.generate_new_preload_inp(modulus, file_index)
    
    # Set up and run first .pmx file (preload only)
    print("Running preload simulation...\n")
    calculix_runner_preload = CalculiXRunner(results_directory,
                                             ccx_executable,
                                             [preload_file_name],
                                             number_of_cores=4
                                             )
    calculix_runner_preload.run()

    print("\nFinished running preload simulation...")

    # Get the Z displacement from the preload and solve for the total required
    # displacement. This will be the minimum value since the load is in the
    # negative Z direction.
    frd_file = os.path.splitext(os.path.basename(preload_file_path))[0]
    frd_path = results_directory + '/' + frd_file + ".frd"
    # frd_path = os.path.join(results_directory, frd_file + ".frd")
    minDef = getNodeDef(frd_path)['dz'].min()

    # Assumes the displacement is defined as a positive value but used as a negative value in the simulation
    total_displacement = np.abs(minDef) + np.abs(displacement)
    total_displacement = -total_displacement
    
    displacement_file_name = inpgen.generate_new_displacement_inp(modulus, total_displacement, file_index)
    

    print("\nRunning displacement simulation...")
    calculix_runner_displacement = CalculiXRunner(results_directory,
                                                 ccx_executable,
                                                 [displacement_file_name],
                                                 number_of_cores=4
                                                 )
    calculix_runner_displacement.run()

    print("\nFinished running displacement simulation")

    # Get the contact force from the .dat file
    displacement_file = os.path.splitext(os.path.basename(displacement_file_name))[0]
    dat_path = results_directory + '/' + displacement_file + \
        '.dat'    # Try this with a path library or os
    df_results = get_contact_force(dat_path)

    # Use df_results and displacement to run regression and get the stiffness
    X = df_results['UZ'].abs() - np.abs(df_results.loc[0, 'UZ'])
    Y = df_results['FZ'].abs()
    regression_modulus, intercept, r_value, p_value, std_err = linregress(X, Y)

    deltaTime = time.time() - tStart
    if deltaTime < 60.0:
        print(f"Elapsed time:\t{deltaTime:.2f} [s]")
    else:
        print(f"Elapsed time:\t{deltaTime/60:.2f} [min]")

    diff = np.abs(regression_modulus - target_stiffness)
    histrow = np.asarray([[modulus, regression_modulus, diff]])
    optHistory = np.vstack((optHistory, histrow))
    # optHistory = np.append(
    #     optHistory, np.array([modulus, regression_modulus, diff]), axis=0)
    print(f'Absolute Difference:\t{diff}')
    print(f'Optimization History:\n{optHistory}')

    return diff