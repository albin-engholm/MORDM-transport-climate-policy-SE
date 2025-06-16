# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 13:48:56 2025

@author: aengholm
"""

import math
import pandas as pd
from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                     EpsilonProgress,
                                                     ArchiveLogger, epsilon_nondominated)
# Load the data
file_str = "1000000_All levers_2025-03-07"
archives = ArchiveLogger.load_archives("./archives/"+file_str+".tar.gz")

# Filter out empty dataframes
archives = {k: v for k, v in archives.items() if not v.empty}

# Sort the dataframes and remove the first which might be empty
dataframes = [df for _, df in sorted(archives.items())]
final_df = dataframes[-1]
def are_all_solutions_in_unique_boxes(df, epsilons, objective_cols):
    """
    Check whether all solutions in a DataFrame fall into unique epsilon boxes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where each row represents a solution.
    epsilons : list of float
        A list of epsilon thresholds. If there are fewer epsilons than objectives,
        they are applied cyclically.
    objective_cols : list of str
        Column names in the DataFrame corresponding to the objectives.
    
    Returns
    -------
    tuple(bool, dict)
        A tuple containing:
          - A boolean indicating whether every solution is in a unique epsilon box.
          - A dictionary mapping each epsilon box (as a tuple) to the count of solutions in that box.
    """
    
    def compute_box(row):
        # For each objective, compute the epsilon box index using floor division.
        box = tuple(math.floor(row[col] / epsilons[i % len(epsilons)]) 
                    for i, col in enumerate(objective_cols))
        return box

    # Compute the epsilon box for every solution.
    boxes = df.apply(compute_box, axis=1)
    
    # Count occurrences of each box.
    box_counts = boxes.value_counts().to_dict()
    
    # All solutions are unique if the number of boxes equals the number of solutions.
    all_unique = len(box_counts) == len(df)
    
    return all_unique, box_counts

# Example usage:
objective_columns = ['M2_driving_cost_car', 'M3_driving_cost_truck',
                     'M4_energy_use_bio', 'M5_energy_use_electricity']
epsilons = [1.5, 8, 1.1, 0.3]  # Example epsilon thresholds for each objective

# Assuming final_df is your DataFrame with one solution per row:
all_unique, box_counts = are_all_solutions_in_unique_boxes(final_df, epsilons, objective_columns)
print("All solutions in unique epsilon boxes (i.e. epsilon-dominant):", all_unique)
print("Epsilon box counts:", box_counts)