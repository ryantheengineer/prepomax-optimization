#libs
import numpy as np
import matplotlib.pyplot as plt
import os
from A_INP_FILE_GENERATOR_MODULE import INPFileGenerator
from B_AUTOMATED_MULTI_RUN_MODULE import CalculiXRunner
from C_getResultsFromFRD import get_node_deformations as getNodeDef
from scipy.optimize import minimize_scalar


def objective_fun(stiffness, base_file_path, resultsDirectory, ccx_executable, number_of_cores, deformationLimit):
    global stiffnessHistory
    global defHistory
    
    #INP Generator Initialization
    inpGeneratorClass = INPFileGenerator(base_file_path, resultsDirectory, ccx_executable, number_of_cores)
    
    currentStiffness = [str(float(stiffness))]
    stiffnessHistory.append(float(currentStiffness[0]))
    
    file_name = inpGeneratorClass.generate_new_inp_file(new_value = currentStiffness, file_index = len(defHistory))
    print(f">> File Generated: {file_name}")
        
    calculix_runner = CalculiXRunner(resultsDirectory, ccx_executable, [file_name], number_of_cores)
    calculix_runner.run()
    print(f">> CalculiX Run Completed for {file_name}")

    frd_path = resultsDirectory + '\\' + file_name + '.frd'
    maxDef = getNodeDef(frd_path)['dy'].max()
    defHistory.append(maxDef)
    
    return np.abs(maxDef - deformationLimit)

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

result = minimize_scalar(objective_fun,
                         bounds=(1e-10, 1000000),
                         method='bounded',
                         args=(base_file_path,
                               resultsDirectory,
                               ccx_executable,
                               number_of_cores,
                               deformationLimit)
                         )

#RESULTING THICKNESS AND DEFORMATION
print('\n\n>> All CalculiX runs completed successfully!\n' + 
      f">> Final Stiffness:\t{stiffnessHistory[-1]}\n" + 
      f">> Final Deformation:\t{defHistory[-1]}\n")


#PLOT
fig, ax1 = plt.subplots()
final_stiffness = stiffnessHistory[-1]
final_deformation = defHistory[-1]

ax1.annotate(f'Final Deformation: {final_deformation:.2f}', xy=(len(defHistory)-1, final_deformation), 
             xytext=(len(defHistory)-1, final_deformation + 10),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=10, color='blue')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.annotate(f'Final Stiffness: {final_stiffness:.2f}', xy=(len(stiffnessHistory)-1, final_stiffness), 
             xytext=(len(stiffnessHistory)-1, final_stiffness + 0.5),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red')
color = 'blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Deformation', color=color)
ax1.plot(defHistory, color=color)
ax1.set_ylim(0, 25)
ax1.tick_params(axis='y', labelcolor=color)

color = 'red'
ax2.set_ylabel('Thickness', color=color)
ax2.plot(stiffnessHistory, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Thickness', color=color)
ax1.axhline(y=deformationLimit, color='lightgreen', linestyle='--', label=f'Deformation Limit: {deformationLimit}')
ax1.text(len(defHistory)*.3, deformationLimit*2, f'Deformation Limit: {deformationLimit} [mm]', fontsize=10, color='lightgreen')
ax1.legend()
ax2.plot(stiffnessHistory, color=color)

fig.tight_layout()  # to ensure the right y-label is not slightly clipped
plot_path = os.path.join(resultsDirectory, 'AAA_thickness_optimization_plot.png')
fig.savefig(plot_path)
print(f"\n>> Plot saved to {plot_path}")

plt.show()

