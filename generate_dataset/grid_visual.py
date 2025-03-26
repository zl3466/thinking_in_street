import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROUTE_DATA_DIR = "644_washington_st"


def load_grid_data(file_path):
    """Load grid data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_grid_from_path(path, grid_size=100):
    """Create a grid layout with the path marked."""
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Mark path in grid
    for x, y in path:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[int(y), int(x)] = 1

    return grid


def visualize_grid(grid_data, output_file):
    """Create a visualization of the grid layout."""
    path = np.array(grid_data["path"])
    grid_size = grid_data.get("grid_size", 100)

    # Create grid from path
    grid = create_grid_from_path(path, grid_size)

    # Create figure and axis
    plt.figure(figsize=(12, 12))

    # Plot the grid
    plt.imshow(grid, cmap='binary', interpolation='nearest')

    # Plot the path with a different color to make it more visible
    path = np.array(grid_data["path"])
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, alpha=0.7)

    # Mark start and end points
    plt.plot(path[0, 0], path[0, 1], 'go', label='Start', markersize=10)  # Green dot for start
    plt.plot(path[-1, 0], path[-1, 1], 'ro', label='End', markersize=10)  # Red dot for end

    # Customize the plot
    plt.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    plt.title('Street Layout Visualization')
    plt.legend()

    # Set axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Make the plot square
    plt.axis('equal')

    # Invert y-axis to match grid coordinates
    plt.gca().invert_yaxis()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

    # Show the plot
    plt.show()


def main():
    # Set up paths
    base_data_dir = Path("/Users/edwinhuang/Desktop/nav/data")
    data_dir = base_data_dir / ROUTE_DATA_DIR

    # Load grid data
    grid_file = data_dir / f"{ROUTE_DATA_DIR}_layout.json"
    if not grid_file.exists():
        print("Grid layout file not found. Please run street_layout.py first.")
        return

    grid_data = load_grid_data(grid_file)
    output_file = data_dir / f"{ROUTE_DATA_DIR}_visualization.png"
    visualize_grid(grid_data, output_file)


if __name__ == "__main__":
    main()
