# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:24:19 2025

@author: Ryan.Larson
"""

import yaml
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import optimizeStiffnessAgainstData as opt
from scipy.optimize import minimize_scalar
import logging
import matplotlib.pyplot as plt
import time

# Configure logging
logging.basicConfig(filename="output.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# YAML file path
YAML_FILE = "config.yaml"

# Define parameters and their types
PARAMETERS = {
    'poisson': 'float',
    'displacement': 'float',
    'results_directory': 'directory',
    'ccx_executable': 'filepath',
    'preload_pmx_file': 'filepath',
    'disp_pmx_file': 'filepath',
    'geo_source_file': 'filepath',
    'geo_target_file': 'filepath',
    'target_stiffness': 'float',
}

PARAMETER_TOOLTIPS = {
    'poisson': "Poisson's ratio to use for simulation",
    'displacement': 'Float displacement value for the 3-point-bend test (non-negative)',
    'results_directory': 'Directory where results files should be generated',
    'ccx_executable': 'CCX executable file (ccx_dynamic.exe, in the Solver folder of PrePoMax)',
    'preload_pmx_file': 'Preload-only .pmx file',
    'disp_pmx_file': 'Displacement-only .pmx file',
    'geo_source_file': 'Original geometry file the .pmx file was created with',
    'geo_target_file': 'New geometry file the .pmx file will be regenerated with',
    'target_stiffness': 'Float stiffness value taken from real test (N/mm). Optimization will run against this target.',
}


def load_yaml():
    """Load existing YAML file or create a new one with empty values."""
    if os.path.exists(YAML_FILE):
        with open(YAML_FILE, "r") as file:
            return yaml.safe_load(file) or {}
    else:
        return {param: "" for param in PARAMETERS}


def save_yaml(data):
    """Save YAML file."""
    with open(YAML_FILE, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        """ Display the tooltip """
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip.wm_geometry(f"+{x}+{y}")  # Position it near the widget

        label = tk.Label(self.tooltip, text=self.text,
                         background="lightyellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        """ Hide the tooltip """
        if self.tooltip:
            self.tooltip.destroy()


class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Select Parameters")

        # Load parameters from YAML file
        self.data = load_yaml()

        # Store widgets for later access
        self.entries = {}

        # Create GUI layout
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Parameter", font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=10, pady=5)
        tk.Label(self.root, text="Current Value", font=(
            "Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=5)
        tk.Label(self.root, text="New Value", font=("Arial", 12, "bold")).grid(
            row=0, column=2, padx=10, pady=5)
        tk.Label(self.root, text="Entry", font=("Arial", 12, "bold")).grid(
            row=0, column=3, padx=10, pady=5)

        for i, (param, param_type) in enumerate(PARAMETERS.items(), start=1):
            label = tk.Label(self.root, text=param, font=("Arial", 10))
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            # Create tooltip for each parameter label with context info
            # Customize this text as needed
            tooltip_text = f"{PARAMETER_TOOLTIPS[param]}"
            Tooltip(label, tooltip_text)

            # Current value from YAML (read-only)
            current_value = self.data.get(param, "")
            current_label = tk.Label(self.root, text=current_value, font=(
                "Arial", 10), relief="sunken", width=50, wraplength=300)
            current_label.grid(row=i, column=1, padx=10, pady=5)

            # New value label (updates from entry)
            new_label = tk.Label(self.root, text=current_value, font=(
                "Arial", 10), relief="sunken", width=50, wraplength=300)
            new_label.grid(row=i, column=2, padx=10, pady=5)

            if param_type in ["integer", "float"]:
                entry = tk.Entry(self.root)
                entry.grid(row=i, column=3, padx=10, pady=5)
                self.entries[param] = (entry, new_label)

            elif param_type == "filepath":
                btn = tk.Button(
                    self.root, text="Browse", command=lambda p=param, l=new_label: self.select_file(p, l))
                btn.grid(row=i, column=3, padx=10, pady=5)
                self.entries[param] = (None, new_label)

            elif param_type == "directory":
                btn = tk.Button(self.root, text="Browse", command=lambda p=param,
                                l=new_label: self.select_directory(p, l))
                btn.grid(row=i, column=3, padx=10, pady=5)
                self.entries[param] = (None, new_label)

        # Bottom Buttons
        update_btn = tk.Button(
            self.root, text="Update Values", command=self.update_values, width=15)
        update_btn.grid(row=len(PARAMETERS) + 1, column=0, pady=10, padx=10)

        save_btn = tk.Button(self.root, text="Accept & Run",
                             command=self.validate_and_save, width=15, bg="green", fg="white")
        save_btn.grid(row=len(PARAMETERS) + 1, column=3, pady=10, padx=10)

    def select_file(self, param, label):
        """ Open file dialog and update the label """
        file_path = filedialog.askopenfilename(title=f"Select {param}")
        if file_path:
            label.config(text=file_path)
            self.data[param] = file_path  # Update data immediately

    def select_directory(self, param, label):
        """ Open file dialog and update the label """
        directory = filedialog.askdirectory(title=f"Select {param}")
        if directory:
            label.config(text=directory)
            self.data[param] = directory  # Update data immediately

    def update_values(self):
        """ Update the 'New Value' column with entered values """
        for param, (entry, label) in self.entries.items():
            if entry:
                new_value = entry.get().strip()
                if new_value:
                    try:
                        # Convert input if necessary
                        if PARAMETERS[param] == "integer":
                            new_value = int(new_value)
                        elif PARAMETERS[param] == "float":
                            new_value = float(new_value)
                        label.config(text=str(new_value))
                        self.data[param] = new_value
                    except ValueError:
                        messagebox.showwarning(
                            "Invalid Input", f"{param} must be a valid {PARAMETERS[param]}.")

    def validate_and_save(self):
        """ Ensure no empty values in 'New Value' column before saving """
        for param in self.entries.keys():
            value = self.entries[param][1].cget("text")
            if param == 'poisson':
                if float(value) < 0.0 or float(value) > 0.5:
                    messagebox.showerror(
                        "Error", "Poisson's ratio must be between 0.0 and 0.5.")
                    return
            # if param == 'number_of_cores':
            #     if int(value) < 1 or int(value) > os.cpu_count()-1:
            #         messagebox.showerror(
            #             "Error", f"CPU cores must be between 1 and {os.cpu_count()-1}")
            if not value:
                messagebox.showerror("Error", f"{param} cannot be empty.")
                return

        save_yaml(self.data)
        # messagebox.showinfo("Success", "Configuration saved!")
        self.root.destroy()  # Close the GUI


def find_necessary_stiffness(params):
    result = minimize_scalar(opt.objective_fun,
                             bounds=(1e-10, 20000.0),
                             method='bounded',
                             args=(params),
                             options={'maxiter': 100}
                             )
    return result


# Run GUI
root = tk.Tk()
app = ParameterGUI(root)
root.mainloop()

params = load_yaml()

# modulus = 2000.0
# df_results = opt.objective_fun(modulus, params)

tstart = time.time()

result = find_necessary_stiffness(params)

tend = time.time()
t_calculation = tend - tstart
if t_calculation < 60.0:
    print(f"\n>> Calculation time:\t{t_calculation:.2f} sec")
elif t_calculation >= 60.0 and t_calculation < 360.0:
    print(f"\n>> Calculation time:\t{t_calculation/60.0:.2f} min")
else:
    print(f"\n>> Calculation time:\t{t_calculation/360.0:.2f} hrs")

# PLOT
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
    params['results_directory'], 'optimization_plot.png')
fig.savefig(plot_path)
print(f"\n>> Plot saved to {plot_path}")

plt.show()
