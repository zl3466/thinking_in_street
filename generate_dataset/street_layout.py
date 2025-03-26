import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_route_data(route_data_file):
    """Load route data from JSON file."""
    with open(route_data_file, 'r') as f:
        data = json.load(f)
    
    # Extract movements and initial heading
    movements = [frame["movement"] for frame in data["frames"]]
    initial_heading = data["frames"][0]["heading"]
    return movements, initial_heading

def create_initial_path(movements, initial_heading):
    """Create path coordinates based on movement information."""
    # Convert initial heading to closest cardinal direction
    # North: 0/360, East: 90, South: 180, West: 270
    cardinal_directions = {
        0: 0,      # North
        90: 90,    # East
        180: 180,  # South
        270: 270   # West
    }
    
    # Find the closest cardinal direction to the initial heading
    initial_direction = min(cardinal_directions.values(), 
                          key=lambda x: min(abs(initial_heading - x), 
                                          abs(initial_heading - (x + 360))))
    
    # Start at (0, 0) with the initial direction
    current_pos = [0, 0]
    current_direction = initial_direction
    path = [current_pos.copy()]
    
    # Movement vectors for each direction
    direction_vectors = {
        0: [0, 1],    # north
        90: [1, 0],   # east
        180: [0, -1], # south
        270: [-1, 0]  # west
    }
    
    # Track turn sequence
    turning = False
    turn_type = None
    
    for i, movement in enumerate(movements):
        if movement in ["left", "right"]:
            # Start of a new turn sequence
            if not turning:
                turning = True
                turn_type = movement
            # Skip additional turn frames of the same type
            continue
        else:  # movement == "forward"
            # If we were turning, complete the turn before moving forward
            if turning:
                if turn_type == "right":
                    current_direction = (current_direction + 90) % 360
                else:  # left turn
                    current_direction = (current_direction - 90) % 360
                turning = False
                turn_type = None
        
        # Move forward in current direction
        move_vector = direction_vectors[current_direction]
        current_pos[0] += move_vector[0]
        current_pos[1] += move_vector[1]
        path.append(current_pos.copy())
    
    return path

def center_path(path, grid_size=100):
    """Center the path in the grid."""
    path = np.array(path)
    
    # Find current bounds
    min_x, min_y = np.min(path, axis=0)
    max_x, max_y = np.max(path, axis=0)
    
    # Calculate required shifts to center
    path_width = max_x - min_x
    path_height = max_y - min_y
    
    x_shift = (grid_size - path_width) // 2 - min_x
    y_shift = (grid_size - path_height) // 2 - min_y
    
    # Apply shift
    centered_path = path + np.array([x_shift, y_shift])
    
    return centered_path.tolist()


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
    data_dir = "../data/long_route_random_single"
    # map_list = os.listdir(data_dir)
    map_name = "map5_chinatown"

    route_file_dir = f"{data_dir}/{map_name}/route_data.json"
    # Load route data directly from the directory
    movements, initial_heading = load_route_data(route_file_dir)
    
    # Create initial path with proper orientation
    path = create_initial_path(movements, initial_heading)
    
    # Center the path
    centered_path = center_path(path)
    
    # Save grid layout to JSON
    output_file = f"{data_dir}/{map_name}/route_grid_layout.json"
    with open(output_file, 'w') as f:
        json.dump({
            "path": centered_path,
            "grid_size": 100,
            "initial_heading": initial_heading
        }, f, indent=2)
    
    print(f"Grid layout saved to {output_file}")

    grid_data = load_grid_data(f"{data_dir}/{map_name}/route_grid_layout.json")
    viz_output_file = f"{data_dir}/{map_name}/grid_layout_viz.png"
    visualize_grid(grid_data, viz_output_file)

if __name__ == "__main__":
    main()

