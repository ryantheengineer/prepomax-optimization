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
import matplotlib.pyplot as plt
import os
# from generateInputFiles import INPFileGenerator
from runModels import CalculiXRunner
from getResults import get_node_deformations as getNodeDef
from getResults import get_contact_force
# from scipy.optimize import minimize_scalar
# import yaml
from scipy.stats import linregress
import time
import logging

logger = logging.getLogger(__name__)

optHistory = np.empty((0, 3))


def objective_fun(modulus, params):
    global optHistory

    tStart = time.time()

    # print('\n\n')
    # print('#'*60)
    # print(f'Trying modulus:\t{modulus}')
    
    logger.info("\n")
    logger.info(f"Trying modulus:\t{modulus}")

    # Unpack params dictionary
    # FEA model parameters
    poisson = params['poisson']
    preload = params['preload']
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

    # global stiffnessHistory
    # global defHistory

    # # Set up and run first .pmx file (preload only)
    # print("Running preload simulation...\n")
    # preload_params = f"modulus={modulus}; poisson={poisson}; preload={preload}"
    # calculix_runner_preload = CalculiXRunner(results_directory,
    #                                          ccx_executable,
    #                                          pmx_file=preload_pmx_file,
    #                                          geo_source_file=geo_source_file,
    #                                          geo_target_file=geo_target_file,
    #                                          params=preload_params,)
    # calculix_runner_preload.regenerate_run()

    # print(f"\nFinished running preload simulation in {time.time() - tStart:.2f} [s]")

    # # Get the Z displacement from the preload and solve for the total required
    # # displacement. This will be the minimum value since the load is in the
    # # negative Z direction.
    # # frd_file = os.path.splitext(os.path.basename(preload_pmx_file))[0]
    # # frd_path = results_directory + '/' + frd_file + ".frd"
    # # # frd_path = os.path.join(results_directory, frd_file + ".frd")
    # # minDef = getNodeDef(frd_path)['dz'].min()
    # preload_file = os.path.splitext(os.path.basename(preload_pmx_file))[0]
    # dat_path = results_directory + '/' + preload_file + '.dat'    # Try this with a path library or os
    # df_results = get_contact_force(dat_path)
    
    # minDef = df_results['UZ'].min()

    # Assumes the displacement is defined as a positive value but used as a negative value in the simulation
    # preload_displacement = np.abs(minDef)
    preload_displacement = 0.8
    total_displacement = preload_displacement + np.abs(displacement)
    
    # print(f'Preload displacement: {preload_displacement}')
    # print(f'Preload displacement: {np.abs(minDef)}')
    # print(f'Total displacement: {total_displacement}')

    # print("\nRunning displacement simulation...")
    displacement_params = f"modulus={modulus}; poisson={poisson}; displacement={total_displacement}; preload_displacement={preload_displacement}"
    calculix_runner_displacement = CalculiXRunner(results_directory,
                                                  ccx_executable,
                                                  pmx_file=disp_pmx_file,
                                                  geo_source_file=geo_source_file,
                                                  geo_target_file=geo_target_file,
                                                  params=displacement_params,)
    calculix_runner_displacement.regenerate_run()

    # print("\nFinished running displacement simulation")
    
    # Get the contact force from the .dat file
    displacement_file = os.path.splitext(os.path.basename(disp_pmx_file))[0]
    dat_path = results_directory + '/' + displacement_file + \
        '.dat'    # Try this with a path library or os
    df_results = get_contact_force(dat_path)
    
    try:
        if df_results.empty:
            raise ValueError("FEA resulted in no data in .dat file")
        logger.info(".dat file is valid. Proceeding with processing.")
    except Exception as e:
        logger.exception(f"Script failed: {e}")
        raise

    # Use df_results and displacement to run regression and get the stiffness
    # X = df_results['UZ'].abs() - np.abs(df_results.loc[0, 'UZ'])
    reject_tol = 1e-5
    filtered_df_results = df_results[df_results['RZ'].notna() & (df_results['RZ'].abs() > reject_tol)]
    logger.info(f"Results:\n{filtered_df_results}")
    X = filtered_df_results['UZ'].abs() - filtered_df_results['UZ'].abs().iloc[0]
    Y = filtered_df_results['RZ'].abs()
    # Y = df_results['FZ'].abs()
    regression_modulus, intercept, r_value, p_value, std_err = linregress(X, Y)
    # print(f"\nFEA regression modulus:\t{regression_modulus}")
    # print(f"R-squared:\t{r_value}")
    logger.info(f"FEA regression modulus:\t{regression_modulus}")
    logger.info(f"R-squared:\t{r_value}")

    deltaTime = time.time() - tStart
    if deltaTime < 60.0:
        # print(f"Total elapsed time for this run:\t{deltaTime:.2f} [s]")
        logger.info(f"Total elapsed time for this run:\t{deltaTime:.2f} [s]")
    else:
        # print(f"Total elapsed time for this run:\t{deltaTime/60:.2f} [min]")
        logger.info(f"Total elapsed time for this run:\t{deltaTime/60:.2f} [min]")

    diff = np.abs(regression_modulus - target_stiffness)
    histrow = np.asarray([[modulus, regression_modulus, diff]])
    optHistory = np.vstack((optHistory, histrow))
    # optHistory = np.append(
    #     optHistory, np.array([modulus, regression_modulus, diff]), axis=0)
    # print(f'Absolute Difference:\t{diff}')
    # print(f'Optimization History:\n{optHistory}')
    logger.info(f'Absolute Difference:\t{diff}')
    logger.info(f'Optimization History:\n{optHistory}')
    
    # PLOT
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), dpi=300)
    # Plot data on each subplot
    axes[0].plot(optHistory[:, 0])
    axes[0].set_title('Modulus of Elasticity (MPa)')
    axes[0].set_xlabel('Iteration')
    
    axes[1].plot(optHistory[:, 1])
    axes[1].set_title('Regression Stiffness (N/mm)')
    axes[1].set_xlabel('Iteration')
    
    axes[2].plot(optHistory[:, 2])
    axes[2].set_title(
        f'Difference in Regression Stiffness from Target {params["target_stiffness"]} (N/mm)')
    axes[2].set_xlabel('Iteration')
    
    plt.suptitle('Optimization Performance')
    
    fig.tight_layout()  # to ensure the right y-label is not slightly clipped
    
    plt.show()

    return diff