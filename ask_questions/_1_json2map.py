import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from utils import *



def visualize_grid(landmarks, grid_size=(10, 10)):
    """Visualizes a 20x20 grid with landmark names at specified (r, c) locations."""
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_yticklabels([])  # Remove y-axis tick labels

    # Place landmarks
    for name, [r, c] in landmarks:
        # Adjust text size to avoid overflow
        ax.text(c + 0.5, rows - r - 1.5, name, ha='center', va='center',
                fontsize=8, wrap=True, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                zorder=5)

    # Set limits and aspect ratio
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.invert_yaxis()  # Flip y-axis so (0,0) is at the top-left

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


def visualize_city_map(data, grid_size=(10, 10), cmap_name='tab10'):
    """Visualizes a city grid with landmarks and color-coded streets using a colormap."""
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get a colormap from matplotlib
    colormap = plt.get_cmap(cmap_name)  # Use a colormap like 'tab10', 'viridis', etc.

    # Create a list of colors for the streets
    street_colors = {}
    unique_streets = list(data["street"].keys())

    # Ensure there are enough colors in the colormap
    num_colors = len(unique_streets)
    if num_colors > colormap.N:
        raise ValueError(f"Colormap '{cmap_name}' has insufficient colors for the number of streets ({num_colors})")

    # Assign colors from the colormap to each street
    for i, street in enumerate(unique_streets):
        street_colors[street] = colormap(i / (num_colors - 1))  # Normalize the color index

    # Draw grid
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Draw streets with different colors
    legend_patches = []  # For legend
    for street, info in data["street"].items():
        color = street_colors.get(street, "gray")  # Default to gray if unknown
        legend_patches.append(mpatches.Patch(color=color, label=street))  # Add to legend
        for r, c in info:
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, color=color, alpha=0.6))

    # Place landmarks
    for each in data["landmark"]:
        name = list(each.keys())[0]
        [r, c] = each[name]
        ax.text(c + 0.5, rows - r - 0.5, name, ha='center', va='center',
                fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Set limits and aspect ratio
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.invert_yaxis()  # Flip y-axis so (0,0) is at the top-left

    # Add legend outside the figure
    ax.legend(handles=legend_patches, loc='best', bbox_to_anchor=(1.02, 1), title="Streets")

    plt.show()


txt_path = "../data/long_route_random_single/map2_soho/bev_map/Q&A.json"
full_response_txt = json.load(open(txt_path))
# print(full_response_txt[0])
json_data = parse_json_from_response(full_response_txt[0])
print(json_data)
# Visualize the grid map
visualize_city_map(json_data)

