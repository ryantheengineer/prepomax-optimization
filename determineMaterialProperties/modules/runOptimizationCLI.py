# -*- coding: utf-8 -*-
"""
Created on Wed May 28 07:15:15 2025

@author: Ryan.Larson
"""

import yaml
import os
import optimizeStiffnessAgainstData as opt
from scipy.optimize import minimize_scalar
import logging
import matplotlib.pyplot as plt
import time
import argparse
from logger_config import setup_logger
from datetime import datetime
from pathlib import Path

# Get the base directory (parent of this script's folder)
base_dir = Path(__file__).resolve().parent.parent

# Configure logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = base_dir / "logs"
# log_dir = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run_{timestamp}.log")
setup_logger(log_file)

# Get a module-specific logger
logger = logging.getLogger(__name__)
logger.info("runOptimizationCLI.py started")


def load_yaml(yaml_file):
    """Load YAML file."""
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file) or {}
    else:
        return None


def find_necessary_stiffness(params, min_modulus, max_modulus, xatol):
    result = minimize_scalar(opt.objective_fun,
                             bounds=(min_modulus, max_modulus),
                             method='bounded',
                             args=(params),
                             options={'maxiter': 100,
                                      'xatol': xatol}
                             )
    return result


if __name__ == "__main__":
    ### Parse arguments    
    parser = argparse.ArgumentParser(description="Flexural stiffness calculation using optimization methods.")
    parser.add_argument('yaml', type=str, help="Input yaml file")
    args = parser.parse_args()
    
    yaml_file = args.yaml
    # yaml_file = "G:/Shared drives/RockWell Shared/Projects/Rockwell Redesign/Strength + Performance/Flexural Stiffness Characterization/7 - Test Specific YAML Files/W1L1V1_Test1.yaml"
    
    params = load_yaml(yaml_file)
    # params = load_yaml(YAML_FILE)
    
    # # Get the base directory (parent of this script's folder)
    # base_dir = Path(__file__).resolve().parent.parent
    results_dir = params['results_directory']
    job_name = params['job_name']

    # Configure logging
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_dir = base_dir / "logs"
    # # log_dir = "C:/Users/Ryan.Larson.ROCKWELLINC/github/prepomax-optimization/determineMaterialProperties/logs"
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    log_file = os.path.join(results_dir, params['log_file'])
    setup_logger(log_file)

    # Get a module-specific logger
    logger = logging.getLogger(__name__)
    logger.info("runOptimizationCLI.py started")
    
    tstart = time.time()
    
    # Optimization bounds and tolerance
    min_modulus = 500.0
    max_modulus = 20000.0
    xatol = 1.0           # Tolerance (in MPa) of the optimization loop. This is how narrow the search window must be to exit the optimization.
    
    ### Calculate the necessary modulus to get the target stiffness
    result = find_necessary_stiffness(params, min_modulus, max_modulus, xatol)
    
    ### Save the results in a .result file
    modulus_opt = result.x
    stiffness_opt = result.fun
    # print(f"Modulus calculated at {modulus_opt} MPa")
    logger.info(f"Modulus calculated at {modulus_opt} MPa")
    
    tend = time.time()
    t_calculation = tend - tstart
    
    # Define the filename with your desired custom extension
    result_filename = f"{job_name}.result"
    result_filename = os.path.join(params['results_directory'], result_filename)
    
    # The content you want to write to the file
    result_content = f"Job:\t{job_name}\n" \
                    f"Modulus:\t{modulus_opt} MPa\n" \
                    f"Target Stiffness:\t{params['target_stiffness']} N/mm\n" \
                    f"Abs Stiffness Error:\t{stiffness_opt} N/mm\n" \
                    f"Calculation Time:\t{t_calculation / 60.0} min"
    
    # Open the file in write mode ('w')
    # 'w' mode will create the file if it doesn't exist, or overwrite it if it does.
    try:
        with open(result_filename, 'w') as file:
            file.write(result_content)
        # print(f"File '{result_filename}' created successfully.")
        logger.info(f"File '{result_filename}' created successfully.")
    except IOError as e:
        # print(f"Error writing to file: {e}")
        logger.error(f"Error writing to file: {e}")
    
    ### Plot and save the optimization progress
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), dpi=300)
    # Plot data on each subplot
    axes[0].plot(opt.optHistory[:, 0])
    axes[0].set_title('Modulus of Elasticity (MPa)')
    axes[0].set_xlabel('Iteration')
    
    axes[1].plot(opt.optHistory[:, 1])
    axes[1].set_title('Regression Stiffness (N/mm)')
    axes[1].set_xlabel('Iteration')
    
    axes[2].plot(opt.optHistory[:, 2])
    axes[2].set_title(
        f'Difference in Regression Stiffness from Target {params["target_stiffness"]} (N/mm)')
    axes[2].set_xlabel('Iteration')
    
    plt.suptitle('Optimization Performance')
    
    fig.tight_layout()  # to ensure the right y-label is not slightly clipped
    plot_path = os.path.join(
        params['results_directory'], f'{params["job_name"]}_optimization_plot.png')
    fig.savefig(plot_path)
    # print(f"\n>> Plot saved to {plot_path}")
    logger.info(f"Plot saved to {plot_path}")
    
    # plt.show()

    if t_calculation < 60.0:
        # print(f"\n>> Calculation time:\t{t_calculation:.2f} sec")
        logger.info(f"Calculation time:\t{t_calculation:.2f} sec")
    elif t_calculation >= 60.0 and t_calculation < 360.0:
        # print(f"\n>> Calculation time:\t{t_calculation/60.0:.2f} min")
        logger.info(f"Calculation time:\t{t_calculation/60.0:.2f} min")
    else:
        # print(f"\n>> Calculation time:\t{t_calculation/3600.0:.2f} hrs")
        logger.info(f"Calculation time:\t{t_calculation/3600.0:.2f} hrs")