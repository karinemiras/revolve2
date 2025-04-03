import numpy as np

# Define tick positions
red_ticks = [0, 20, 40, 60, 80, 100]          # Red bar starts at 0, then percentages
green_ticks = [100, 150, 200, 250, 300, 338]  # Green bar starts at 100%, ends at ∞

# Define corresponding labels
red_labels = ["0"] + [f"{tick}%" for tick in red_ticks[1:]]  # First value is "0", rest are percentages
green_labels = [f"{tick}%" for tick in green_ticks[:-1]] + [r"$\infty$"]  # Last value is ∞

# Get colorbars
cbar_green = ax.collections[0].colorbar
cbar_red = ax.collections[-1].colorbar

# Apply tick positions and labels
cbar_green.set_ticks(green_ticks)
cbar_green.set_ticklabels(green_labels)

cbar_red.set_ticks(red_ticks)
cbar_red.set_ticklabels(red_labels)
