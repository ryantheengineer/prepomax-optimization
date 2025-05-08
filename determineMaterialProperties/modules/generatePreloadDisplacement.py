# -*- coding: utf-8 -*-
"""
Created on Thu May  8 07:33:59 2025

@author: Ryan.Larson
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:24:30 2025

@author: Ryan.Larson
"""

import os

class INPFileGenerator:
    def __init__(self, preload_file_path, displacement_file_path, new_directory, ccx_executable, number_of_cores=4):
        self.preload_file_path = preload_file_path
        self.displacement_file_path = displacement_file_path
        self.new_directory = new_directory
        self.ccx_executable = ccx_executable
        self.number_of_cores = number_of_cores
        self.generated_files = []
    
    def generate_new_preload_inp(self, new_stiffness, file_index):
        # Create new file path
        base_name = os.path.basename(self.preload_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_file_path = os.path.join(self.new_directory, f'{name_without_ext}_{file_index}.inp')
        
        with open(self.preload_file_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if "*Material, Name=Fiberglass" in line:
                    target_char = ","
                    index = lines[i+2].find(target_char)
                    if index != -1:
                        lines[i + 2] = str(new_stiffness) + lines[i+2][index:]
                    
        # Save new file
        os.makedirs(self.new_directory, exist_ok=True)
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(lines)

        # Print file name and directory
        print(f"GENERATED! New preload .inp file ---> {new_file_path}")

        # Add the created file name to the list
        self.generated_files.append(os.path.splitext(os.path.basename(new_file_path))[0])
        
        return new_file_path
    
    def generate_new_displacement_inp(self, new_stiffness, new_displacement, file_index):
        # Create new file path
        base_name = os.path.basename(self.displacement_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_file_path = os.path.join(self.new_directory, f'{name_without_ext}_{file_index}.inp')
        
        with open(self.displacement_file_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if "*Material, Name=Fiberglass" in line:
                    target_char = ","
                    index = lines[i+2].find(target_char)
                    if index != -1:
                        lines[i + 2] = str(new_stiffness) + lines[i+2][index:]
                elif line == "** Name: Displacement_Rotation-Anvil\n":
                    target_line = lines[i + 4]
                    vals = target_line.split(", ")
                    vals[3] = str(new_displacement)
                    lines[i + 4] = ", ".join(vals) + "\n"
                    
        # Save new file
        os.makedirs(self.new_directory, exist_ok=True)
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(lines)

        # Print file name and directory
        print(f"GENERATED! New displacement .inp file ---> {new_file_path}")

        # Add the created file name to the list
        self.generated_files.append(os.path.splitext(os.path.basename(new_file_path))[0])
        
        return new_file_path
                        
    
    # def generate_new_inp_file(self, new_value, file_index):
    #     # # Create new file path
    #     # new_file_path = os.path.join(self.new_directory, f'shell_7_parts_static_{file_index}.inp')

    #     # Read main file to determine whether it is a Preload or Displacement simulation
    #     with open(self.base_file_path, 'r') as file:
    #         lines = file.readlines()
            
    #         for i, line in enumerate(lines):
    #             if "** Name: Preload: Deactivated" in line:
    #                 mode = "DISPLACEMENT"
    #                 break
    #             elif "** Name: Prescribed_Displacement: Deactivated" in line:
    #                 mode = "PRELOAD"
    #                 break
    #             else:
    #                 mode = None
                
    #     return mode

        # # Change values
        # for i, line in enumerate(lines):
        #     # if "*Shell section, Elset=set_1, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[0] + '\n'
        #     # elif "*Shell section, Elset=set_2, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[1] + '\n'
        #     # elif "*Shell section, Elset=set_3, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[2] + '\n'
        #     # elif "*Shell section, Elset=set_4, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[3] + '\n'
        #     # elif "*Shell section, Elset=set_5, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[4] + '\n'
        #     # elif "*Shell section, Elset=set_6, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[5] + '\n'
        #     # elif "*Shell section, Elset=set_7, Material=ABS, Offset=0" in line:
        #     #     lines[i + 1] = new_values[6] + '\n'
        #     if "Elastic" in line:
        #         target_char = ","
        #         index = lines[i+1].find(target_char)
        #         if index != -1:
        #             lines[i + 1] = new_value[0] + lines[i+1][index:]

        # # Save new file
        # os.makedirs(self.new_directory, exist_ok=True)
        # with open(new_file_path, 'w') as new_file:
        #     new_file.writelines(lines)

        # # Print file name and directory
        # print(f"GENERATED! New .inp file ---> {new_file_path}")

        # # Add the created file name to the list
        # self.generated_files.append(os.path.splitext(os.path.basename(new_file_path))[0])
        
        # return self.generated_files[-1]

    # def generate_all_files(self, new_values_lists):
    #     for index, new_values in enumerate(new_values_lists, start=1):
    #         self.generate_new_inp_file(new_values, index)

    #     # Print all operations completed
    #     print("\n**************************************************************\nAll new .inp file generation processes completed.\n**************************************************************")
    #     return self.generated_files
