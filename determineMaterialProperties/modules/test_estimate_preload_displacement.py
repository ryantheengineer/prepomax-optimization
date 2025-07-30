# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:45:54 2025

@author: Ryan.Larson
"""

import optimizeStiffnessAgainstData as opt

target_stl = "G:/Shared drives/RockWell Shared/Rockwell Redesign Project/Strength + Performance/Flexural Stiffness Characterization/5 - Flexural Test Meshes/X5_Test1.stl"

preload_displacement = opt.estimate_preload_displacement(target_stl)