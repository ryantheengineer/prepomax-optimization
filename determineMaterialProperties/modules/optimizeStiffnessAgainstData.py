# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:20:07 2025

@author: Ryan.Larson

Script for finding necessary material properties to match physical tests, for
multiple specimens and multiple tests per specimen. Each specimen/test combo
is optimized to find the material property that most closely matches the
corresponding physical test, then it moves on to another specimen/test combo.
"""

#libs
import numpy as np
import matplotlib.pyplot as plt
import os
from generateInputFiles import INPFileGenerator
from runModels import CalculiXRunner
from getResults import get_node_deformations as getNodeDef
from getResults import get_contact_force
from scipy.optimize import minimize_scalar
import yaml


def objective_fun(modulus, params):
    ### Unpack params dictionary
    # FEA model parameters
    poisson = params['poisson']
    displacement = params['displacement']
    
    # Filepath parameters
    results_directory = params['results_directory']
    ccx_executable = params['ccx_executable']
    preload_pmx_file = params['preload_pmx_file']
    disp_pmx_file = params['disp_pmx_file']
    geo_source_file = params['geo_source_file'] # Geometry source file (original geometry file the .pmx file was created with)
    geo_target_file = params['geo_target_file'] # Geometry target file (new geometry file the .pmx file will be regenerated with)
    
    # Simulation parameters
    number_of_cores = params['number_of_cores']
    
    
    
    # global stiffnessHistory
    # global defHistory
    
    
    # Set up and run first .pmx file (preload only)
    print("Running preload simulation...\n")
    calculix_runner_preload = CalculiXRunner(results_directory,
                                             ccx_executable,
                                             pmx_file=preload_pmx_file,
                                             geo_source_file=geo_source_file,
                                             geo_target_file=geo_target_file,
                                             params=f'["modulus={modulus}; poisson={poisson}"]',
                                             number_of_cores=number_of_cores)
    calculix_runner_preload.regenerate_run()
    
    print("\nFinished running preload simulation...")
    
    # Get the Z displacement from the preload and solve for the total required
    # displacement. This will be the minimum value since the load is in the
    # negative Z direction.
    frd_file = os.path.splitext(os.path.basename(preload_pmx_file))[0]
    frd_path = results_directory + '/' + frd_file + ".frd"
    # frd_path = os.path.join(results_directory, frd_file + ".frd")
    minDef = getNodeDef(frd_path)['dz'].min()
    
    total_displacement = np.abs(minDef) + np.abs(displacement) # Assumes the displacement is defined as a positive value but used as a negative value in the simulation
    
    print("\nRunning displacement simulation...")
    calculix_runner_displacement = CalculiXRunner(results_directory,
                                                  ccx_executable,
                                                  pmx_file=disp_pmx_file,
                                                  geo_source_file=geo_source_file,
                                                  geo_target_file=geo_target_file,
                                                  params=f'["modulus={modulus}; poisson={poisson}; displacement={total_displacement}"]',
                                                  number_of_cores=number_of_cores)
    calculix_runner_displacement.regenerate_run()
    
    print("\nFinished running preload simulation")
    
    # Get the contact force from the .dat file
    displacement_file = os.path.splitext(os.path.basename(disp_pmx_file))[0]
    dat_path = results_directory + '/' + displacement_file + '.dat'    # Try this with a path library or os
    df_results = get_contact_force(dat_path)
    
    return df_results
    
    # Use CFZ and displacement to run regression and get the stiffness
    
    
    
    # Optimize to minimize the difference between the FEA stiffness value and
    # the test stiffness
    
    # return maxCFZ

#init parameters
base_file_path = r'C:\Users\Ryan.Larson.ROCKWELLINC\github\prepomax-optimization\materialOpt\02_baseModelFile\shell_7_parts_base_static.inp'
# base_file_path = r'simpleShellThicknessOpt\02_baseModelFile\shell_7_parts_base_static.inp'
resultsDirectory = r'C:\Users\Ryan.Larson.ROCKWELLINC\github\prepomax-optimization\materialOpt\03_results'
# resultsDirectory = r'materialOpt\03_results'
ccx_executable = r'E:\Downloads\PrePoMax v2.2.0\PrePoMax v2.2.0\Solver\ccx_dynamic.exe'
# ccx_executable = r'C:\\Users\\CalculiX\\bin\\ccx\\ccx213.exe'
number_of_cores = 4

defHistory = []
stiffnessHistory = []
deformationLimit = 5

# params = {'poisson': 0.25,
#         'displacement': 3,
#         'results_directory': r'C:\Users\Ryan.Larson.ROCKWELLINC\github\prepomax-optimization\materialOpt\03_results',
#         'ccx_executable': r'E:\Downloads\PrePoMax v2.2.0\PrePoMax v2.2.0\Solver\ccx_dynamic.exe',
#         'preload_pmx_file': ,
#         'disp_pmx_file': ,
#         'geo_source_file': ,
#         'geo_target_file': ,
#         'number_of_cores': 4}

# result = minimize_scalar(objective_fun,
#                          bounds=(1e-10, 1000000),
#                          method='bounded',
#                          args=(params,)
#                          )

# #RESULTING THICKNESS AND DEFORMATION
# print('\n\n>> All CalculiX runs completed successfully!\n' + 
#       f">> Final Stiffness:\t{stiffnessHistory[-1]}\n" + 
#       f">> Final Deformation:\t{defHistory[-1]}\n")


# #PLOT
# fig, ax1 = plt.subplots()
# final_stiffness = stiffnessHistory[-1]
# final_deformation = defHistory[-1]

# ax1.annotate(f'Final Deformation: {final_deformation:.2f}', xy=(len(defHistory)-1, final_deformation), 
#              xytext=(len(defHistory)-1, final_deformation + 10),
#              arrowprops=dict(facecolor='blue', shrink=0.05),
#              fontsize=10, color='blue')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.annotate(f'Final Stiffness: {final_stiffness:.2f}', xy=(len(stiffnessHistory)-1, final_stiffness), 
#              xytext=(len(stiffnessHistory)-1, final_stiffness + 0.5),
#              arrowprops=dict(facecolor='red', shrink=0.05),
#              fontsize=10, color='red')
# color = 'blue'
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('Deformation', color=color)
# ax1.plot(defHistory, color=color)
# ax1.set_ylim(0, 25)
# ax1.tick_params(axis='y', labelcolor=color)

# color = 'red'
# ax2.set_ylabel('Thickness', color=color)
# ax2.plot(stiffnessHistory, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylabel('Thickness', color=color)
# ax1.axhline(y=deformationLimit, color='lightgreen', linestyle='--', label=f'Deformation Limit: {deformationLimit}')
# ax1.text(len(defHistory)*.3, deformationLimit*2, f'Deformation Limit: {deformationLimit} [mm]', fontsize=10, color='lightgreen')
# ax1.legend()
# ax2.plot(stiffnessHistory, color=color)

# fig.tight_layout()  # to ensure the right y-label is not slightly clipped
# plot_path = os.path.join(resultsDirectory, 'AAA_thickness_optimization_plot.png')
# fig.savefig(plot_path)
# print(f"\n>> Plot saved to {plot_path}")

# plt.show()

