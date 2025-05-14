# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:16:00 2025

@author: Ryan.Larson
"""

# logger_config.py
import logging

def setup_logger(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),  # 'w' to overwrite each run
            logging.StreamHandler()     # Logs to console
        ]
    )
