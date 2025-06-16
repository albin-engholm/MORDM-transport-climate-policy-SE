# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:23:09 2023

@author: aengholm
"""

from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                     EpsilonProgress,
                                                     ArchiveLogger, epsilon_nondominated)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data
file_str = "1000000_All levers_2025-03-07"
archives = ArchiveLogger.load_archives("./archives/"+file_str+".tar.gz")

dpi = 300  # default dpi 300
# Filter out empty dataframes
archives = {k: v for k, v in archives.items() if not v.empty}

# Sort the dataframes and remove the first which might be empty
dataframes = [df for _, df in sorted(archives.items())]
x_var = "M5_energy_use_electricity"
y_var = "M4_energy_use_bio"
c_var = "M2_driving_cost_car"


# Calculate global min and max for the axes and the color variable
x_min = min(df[x_var].min() for df in dataframes)-0.5
x_max = max(df[x_var].max() for df in dataframes)+0.5
#x_max = 1
y_min = min(df[y_var].min() for df in dataframes)-0.5
y_max = max(df[y_var].max() for df in dataframes)+0.5
y_max = 15
c_min = min(df[c_var].min() for df in dataframes)
c_max = max(df[c_var].max() for df in dataframes)

fig, ax = plt.subplots()

# Prepare a scatter plot that will be updated at each frame.
# Initialize with empty data, and map colors to M2 outcome
scatter = ax.scatter(dataframes[0][x_var], dataframes[0][y_var],
                     c=dataframes[0][c_var], s=15,
                     cmap='viridis', vmin=c_min, vmax=c_max)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax, label=c_var)

# Set the labels for your plot
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)

# Set the limits for the x and y axes
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


def animate(i):
    # Update the data for the scatter plot.
    scatter.set_offsets(dataframes[i][[x_var, y_var]].values)
    # Set color according to M2 outcome
    scatter.set_array(dataframes[i][c_var].values)
    ax.set_title(f'Number of function evaluations: {sorted(archives.keys())[i]}')


ani = animation.FuncAnimation(fig, animate, frames=len(dataframes), interval=1, repeat=True)  # default interval 200

# Save the animation
ani.save('./figs/animation'+file_str+'.gif', writer='Pillow', fps=4, dpi=dpi)  # default fps 50

# Create a new figure and axes for the static plot
fig_static, ax_static = plt.subplots()

# Access the last dataframe
final_df = dataframes[-1]

# Create a scatter plot
scatter_static = ax_static.scatter(final_df[x_var], final_df[y_var], c=final_df[c_var], cmap='viridis',)

# Optionally, add a color bar
plt.colorbar(scatter_static, ax=ax_static, label=c_var)

# Set labels and title
ax_static.set_xlabel(x_var)
ax_static.set_ylabel(y_var)


# Save the figure
plt.savefig(f'./figs/MOEA_final_solutions {file_str}.png', dpi=dpi)

# Optionally, show the plot
plt.show()
