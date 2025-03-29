import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from utils import *


def plot_coord(location_names, pred_coords, gt_coords, out_dir):
    plt.figure(figsize=(10, 8))

    # Plot predicted coordinates
    for name, (px, py), (gx, gy) in zip(location_names, pred_coords, gt_coords):
        plt.scatter(px, py, color='blue', marker='o', label='Predicted' if name == location_names[0] else "")
        plt.text(px, py, f' {name}', fontsize=9, verticalalignment='bottom', color='blue')

        # Plot ground truth coordinates
        plt.scatter(gx, gy, color='red', marker='x', label='Ground Truth' if name == location_names[0] else "")
        plt.text(gx, gy, f' {name}', fontsize=9, verticalalignment='top', color='red')

    plt.title("Predicted vs Ground Truth Coordinates")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{out_dir}/grid_map_meters.png")

    # plt.show()


def calculate_errors(pred_coords, gt_coords):
    # Convert to NumPy arrays for vectorized operations
    pred_coords = np.array(pred_coords)
    gt_coords = np.array(gt_coords)

    # Calculate Euclidean distances between predicted and ground truth coordinates
    distances = np.linalg.norm(pred_coords - gt_coords, axis=1)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(distances ** 2))

    # Mean Euclidean Distance (MED)
    med = np.mean(distances)

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} meters")
    print(f"Mean Euclidean Distance (MED): {med:.2f} meters")

    return rmse, med


def comprehensive_eval(pred_coords, gt_coords, location_names=None):
    """
    Comprehensive evaluation of coordinate predictions.

    Parameters:
    -----------
    pred_coords : array-like
        Predicted coordinates as (x, y) pairs
    gt_coords : array-like
        Ground truth coordinates as (x, y) pairs
    location_names : list, optional
        Names of the locations corresponding to the coordinates

    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    import numpy as np
    from scipy.stats import pearsonr

    # Convert to NumPy arrays
    pred_coords = np.array(pred_coords)
    gt_coords = np.array(gt_coords)

    # Calculate errors in x and y directions
    errors_x = pred_coords[:, 0] - gt_coords[:, 0]
    errors_y = pred_coords[:, 1] - gt_coords[:, 1]

    # Calculate Euclidean distances
    distances = np.linalg.norm(pred_coords - gt_coords, axis=1)

    # Basic statistics
    metrics = {
        # Overall metrics
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'med': np.mean(distances),
        'max_error': np.max(distances),
        'min_error': np.min(distances),
        'std_error': np.std(distances),
        'median_error': np.median(distances),

        # Directional metrics
        'mean_error_x': np.mean(np.abs(errors_x)),
        'mean_error_y': np.mean(np.abs(errors_y)),
        'rmse_x': np.sqrt(np.mean(errors_x ** 2)),
        'rmse_y': np.sqrt(np.mean(errors_y ** 2)),

        # Bias analysis
        'mean_bias_x': np.mean(errors_x),  # Positive means predictions are right of truth
        'mean_bias_y': np.mean(errors_y),  # Positive means predictions are above truth

        # Percentile errors
        'error_95th_percentile': np.percentile(distances, 95),
        'error_75th_percentile': np.percentile(distances, 75),
        'error_50th_percentile': np.percentile(distances, 50),

        # Count of predictions within thresholds
        'within_5m': np.mean(distances < 5.0) * 100,
        'within_20m': np.mean(distances < 20.0) * 100,
        'within_50m': np.mean(distances < 50.0) * 100,
        'within_200m': np.mean(distances < 200.0) * 100,
        'within_500m': np.mean(distances < 500.0) * 100,
        'within_500+m': np.mean(distances >= 500.0) * 100,
    }

    # Calculate correlation between predicted and ground truth
    if len(pred_coords) > 1:
        corr_x, _ = pearsonr(pred_coords[:, 0], gt_coords[:, 0])
        corr_y, _ = pearsonr(pred_coords[:, 1], gt_coords[:, 1])
        metrics['correlation_x'] = corr_x
        metrics['correlation_y'] = corr_y

    # Per-point analysis if location names are provided
    if location_names is not None:
        point_metrics = {}
        for i, name in enumerate(location_names):
            point_metrics[name] = {
                'distance': distances[i],
                'error_x': abs(errors_x[i]),
                'error_y': abs(errors_y[i]),
                'bias_x': errors_x[i],
                'bias_y': errors_y[i]
            }
        metrics['per_point'] = point_metrics

    return metrics


# def map_coordinates_to_grid(route_coords, coordinates, grid_size=10, padding=0):
#     """
#     Map a list of (x, y) coordinates to grid indices in a grid_size x grid_size grid.
#
#     Args:
#         coordinates: List of (x, y) coordinate tuples
#         grid_size: Size of the grid (default: 10x10)
#         padding: Padding percentage around min/max values (default: 5%)
#
#     Returns:
#         grid_indices: List of (row, col) grid indices
#         bounds: Dictionary with min_x, max_x, min_y, max_y values
#         cell_dimensions: Dictionary with cell_width and cell_height in meters
#     """
#     if not coordinates:
#         return [], {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1}, {"cell_width": 0, "cell_height": 0}
#
#     # Use route frame coordinates to find grid map boundary
#     # Convert to numpy array for easier operations
#     # TODO: return grid size and boundary in meters
#     coords_array = np.array(route_coords)
#
#     # Find the boundaries of the area
#     min_x, min_y = np.min(coords_array[:, 0]), np.min(coords_array[:, 1])
#     max_x, max_y = np.max(coords_array[:, 0]), np.max(coords_array[:, 1])
#
#     # Add padding to ensure all points are within the grid
#     x_range = max_x - min_x
#     y_range = max_y - min_y
#
#     min_x -= x_range * padding
#     max_x += x_range * padding
#     min_y -= y_range * padding
#     max_y += y_range * padding
#
#     bounds = {
#         "min_x": min_x,
#         "max_x": max_x,
#         "min_y": min_y,
#         "max_y": max_y
#     }
#
#     # Calculate grid cell size in meters
#     cell_width = (max_x - min_x) / grid_size
#     cell_height = (max_y - min_y) / grid_size
#
#     cell_dimensions = {
#         "cell_width": cell_width,
#         "cell_height": cell_height
#     }
#
#     # Map each coordinate to grid index
#     grid_indices = []
#     for x, y in coordinates:
#         # Calculate grid indices
#         # Note: we use grid_size-1 to ensure we don't exceed the maximum index
#         col = min(int((x - min_x) / cell_width), grid_size - 1)
#         # For row, we invert the y-axis so that higher y values are at the top of the grid
#         row = min(grid_size - 1 - int((y - min_y) / cell_height), grid_size - 1)
#         grid_indices.append([row, col])
#
#     return grid_indices, bounds, cell_dimensions
def map_coordinates_to_grid(route_coords, coordinates, grid_size=10, padding=0):
    """
    Map a list of (lat, lng) coordinates to grid indices in a grid_size x grid_size grid.

    Args:
        route_coords: List of (lat, lng) route coordinates to determine map bounds
        coordinates: List of (lat, lng) coordinates to map to the grid
        grid_size: Size of the grid (default: 10x10)
        padding: Padding percentage around min/max values (default: 5%)

    Returns:
        grid_indices: List of (row, col) grid indices
        bounds: Dictionary with min_x, max_x, min_y, max_y values in meters
        cell_dimensions: Dictionary with cell_width and cell_height in meters
    """
    if not coordinates:
        return [], {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1}, {"cell_width": 0, "cell_height": 0}

    # Use route frame coordinates to find grid map boundary
    coords_array = np.array(route_coords)

    # Find the boundaries of the area (latitude and longitude)
    min_lat, min_lng = np.min(coords_array[:, 0]), np.min(coords_array[:, 1])
    max_lat, max_lng = np.max(coords_array[:, 0]), np.max(coords_array[:, 1])

    # Add padding to ensure all points are within the grid
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng

    min_lat -= lat_range * padding
    max_lat += lat_range * padding
    min_lng -= lng_range * padding
    max_lng += lng_range * padding

    # Convert lat/lng bounds to meters using Geopy
    width_m = geodesic((min_lat, min_lng), (min_lat, max_lng)).meters
    height_m = geodesic((min_lat, min_lng), (max_lat, min_lng)).meters

    # Calculate grid cell size in meters
    cell_width = width_m / grid_size
    cell_height = height_m / grid_size

    bounds = {
        "min_x": 0,
        "max_x": width_m,
        "min_y": 0,
        "max_y": height_m
    }

    cell_dimensions = {
        "cell_width": cell_width,
        "cell_height": cell_height
    }

    # Normalize coordinates to meters and map each to a grid index
    grid_indices = []
    for coord in coordinates:
        x = geodesic((min_lat, min_lng), (min_lat, coord[1])).meters
        if coord[1] < min_lng:
            x = -x

        y = geodesic((min_lat, min_lng), (coord[0], min_lng)).meters
        if coord[0] < min_lat:
            y = -y

        # Calculate grid indices (invert y-axis to have top-to-bottom)
        col = min(int(x / cell_width), grid_size - 1)
        row = min(grid_size - 1 - int(y / cell_height), grid_size - 1)
        grid_indices.append([row, col])

    return grid_indices, bounds, cell_dimensions

def create_grid_occupation_map(grid_indices, grid_size=10):
    """
    Create a grid occupation map (binary matrix) based on grid indices.

    Args:
        grid_indices: List of (row, col) grid indices
        grid_size: Size of the grid

    Returns:
        occupation_map: 2D numpy array where 1 indicates occupied cells
    """
    occupation_map = np.zeros((grid_size, grid_size), dtype=int)
    for row, col in grid_indices:
        occupation_map[row, col] = 1
    return occupation_map

def visualize_grid_mapping(coordinates, grid_indices, bounds, cell_dimensions, grid_size=10):
    """
    Visualize the original coordinates and their grid mapping.

    Args:
        coordinates: List of (x, y) coordinate tuples
        grid_indices: List of (row, col) grid indices
        bounds: Dictionary with min_x, max_x, min_y, max_y values
        cell_dimensions: Dictionary with cell_width and cell_height
        grid_size: Size of the grid
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original coordinates
    coords_array = np.array(coordinates)
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], c='blue', marker='o')
    ax1.set_title('Original Coordinates')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True)

    # Add actual bounds
    min_x, max_x = bounds["min_x"], bounds["max_x"]
    min_y, max_y = bounds["min_y"], bounds["max_y"]
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                     fill=False, edgecolor='red', linestyle='--')
    ax1.add_patch(rect)

    # Add grid lines to show actual grid in the original coordinate space
    cell_width, cell_height = cell_dimensions["cell_width"], cell_dimensions["cell_height"]

    # Vertical grid lines
    for i in range(grid_size + 1):
        x = min_x + i * cell_width
        ax1.axvline(x, color='gray', linestyle='-', alpha=0.3)

    # Horizontal grid lines
    for i in range(grid_size + 1):
        y = min_y + i * cell_height
        ax1.axhline(y, color='gray', linestyle='-', alpha=0.3)

    # Plot grid mapping
    occupation_map = create_grid_occupation_map(grid_indices, grid_size)
    ax2.imshow(occupation_map, cmap='Blues', origin='upper')
    ax2.set_title(f'Grid Representation ({grid_size}x{grid_size})')

    # Add grid lines
    for i in range(grid_size + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=0.5)
        ax2.axvline(i - 0.5, color='black', linewidth=0.5)

    # Add coordinates as text
    for i, (row, col) in enumerate(grid_indices):
        ax2.text(col, row, f'{i}', ha='center', va='center', color='red')

    # Add grid indices
    for i in range(grid_size):
        for j in range(grid_size):
            ax2.text(j, i, f'({i},{j})', ha='center', va='center', color='gray',
                     fontsize=8, alpha=0.7)

    # Add a text box with cell dimensions
    cell_info = (f"Cell Width: {cell_width:.2f} meters\n"
                 f"Cell Height: {cell_height:.2f} meters\n"
                 f"Total Area Width: {max_x - min_x:.2f} meters\n"
                 f"Total Area Height: {max_y - min_y:.2f} meters")

    fig.text(0.02, 0.02, cell_info, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    return fig


# def get_coordinate_from_grid_index(row, col, bounds, cell_dimensions, grid_size=10):
#     """
#     Convert a grid index back to the center of the corresponding cell in coordinate space.
#
#     Args:
#         row, col: Grid indices
#         bounds: Dictionary with min_x, max_x, min_y, max_y values
#         cell_dimensions: Dictionary with cell_width and cell_height
#         grid_size: Size of the grid
#
#     Returns:
#         (x, y): Coordinate pair representing the center of the grid cell
#     """
#     cell_width = cell_dimensions["cell_width"]
#     cell_height = cell_dimensions["cell_height"]
#     min_x = bounds["min_x"]
#     min_y = bounds["min_y"]
#
#     # Calculate center of the cell
#     # Note the inverted row due to our grid representation
#     x = min_x + (col + 0.5) * cell_width
#     y = min_y + (grid_size - 1 - row + 0.5) * cell_height
#
#     return [x, y]
def grid_to_meters(grid_indices, cell_dimensions, grid_size=10):
    """
    Convert grid indices back to meter coordinates.

    Args:
        grid_indices: List of (row, col) grid indices
        bounds: Dictionary with min_x, max_x, min_y, max_y values in meters
        cell_dimensions: Dictionary with cell_width and cell_height in meters
        grid_size: Size of the grid (default: 10x10)

    Returns:
        meter_coords: List of (x, y) coordinates in meters, with (min_x, min_y) being origin (0, 0)
    """
    meter_coords = []

    for row, col in grid_indices:
        # Convert row and col to x, y meters
        x = col * cell_dimensions["cell_width"] + cell_dimensions["cell_width"] / 2
        y = (grid_size - 1 - row) * cell_dimensions["cell_height"] + cell_dimensions["cell_height"] / 2

        meter_coords.append([x, y])

    return meter_coords

def visualize_city_map(location_name_list, pred_grid_indices, gt_grid_indices, out_dir, grid_size=(10, 10)):
    """Visualizes a city grid with landmarks and color-coded streets using a colormap."""
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)

    # Place landmarks
    for i in range(len(location_name_list)):
        name = location_name_list[i]
        pred = pred_grid_indices[i]
        gt = gt_grid_indices[i]

        # ax.text(pred[1] + 0.5, rows - pred[0] - 0.5, name, ha='center', va='center',
        #         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        #
        # ax.text(gt[1] + 0.5, rows - gt[0] - 0.5, name, ha='center', va='center',
        #         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # Predicted landmark
        ax.text( pred[1] - 0.5, pred[0] + 0.5,f"{name}\n{pred[0] + 0.5, pred[1] - 0.5}", ha='center', va='center',
                fontsize=8, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

        # Ground truth landmark
        ax.text(gt[1] - 0.5, gt[0] + 0.5, f"{name}\n{gt[0], gt[1]}", ha='center', va='center',
                fontsize=8, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    # Set limits and aspect ratio
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    ax.invert_yaxis()  # Flip y-axis so (0,0) is at the top-left
    plt.savefig(f"{out_dir}/grid_map_indices.png")
    # plt.show()

def main():
    data_dir = "../data/long_route_random_single"
    result_dir = "../result/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop_auto"

    map_list = os.listdir(data_dir)
    pred_grid_list_all = []
    gt_grid_list_all = []

    pred_coord_list_all = []
    gt_coord_list_all = []

    # map_name = "map2_soho"
    for map_name in map_list:
        if map_name == "map1_downtown_bk" or map_name == "map1_downtown_bk_new":
            continue
        print(f"map {map_name}")
        json_path = f"../data/long_route_random_single/{map_name}/bev_grid_map/Q&A.json"
        out_dir = f"../data/long_route_random_single/{map_name}/bev_grid_map"
        route_data = json.load(open(f"../data/long_route_random_single/{map_name}/route_data.json"))
        route_coords = get_coord_from_route_data(route_data)

        ''' load response & parse returned grid indices '''
        json_data = json.load(open(json_path))
        grid_size = json_data["grid_size"]
        pred_coord_dict = parse_json_from_response(json_data["answers"][0])
        location_name_list = list(pred_coord_dict.keys())
        pred_grid_indices = list(pred_coord_dict.values())
        # print(pred_coord_dict)

        ''' load gt generated during prompting from same json file '''
        gt_coord_list = json_data["gt"]
        gt_grid_indices, bounds, cell_dimensions = map_coordinates_to_grid(route_coords, gt_coord_list, grid_size=grid_size)

        ''' plot the locations on grid map '''
        visualize_city_map(location_name_list, pred_grid_indices, gt_grid_indices, out_dir=out_dir, grid_size=(grid_size, grid_size))

        ''' convert the grid idxes back to meters '''
        ''' if we have a large number of grids, this essentially simulates the 3_coord_map estimation '''
        pred_coord_meters = grid_to_meters(pred_grid_indices, cell_dimensions)
        gt_coord_meters = grid_to_meters(gt_grid_indices, cell_dimensions)
        # pred_coord_meters = []
        # gt_coord_meters = []
        # for i in range(len(pred_grid_indices)):
        #     pred_idx = pred_grid_indices[i]
        #     gt_idx = gt_grid_indices[i]
        #
        #     pred_coord = grid_to_meters(pred_idx[0], pred_idx[1], bounds, cell_dimensions, grid_size=10)
        #     pred_coord_meters.append(pred_coord)
        #
        #     gt_coord = get_coordinate_from_grid_index(gt_idx[0], gt_idx[1], bounds, cell_dimensions, grid_size=10)
        #     gt_coord_meters.append(gt_coord)
        ''' plot the locations in meters on grid map '''
        plot_coord(location_name_list, pred_coord_meters, gt_coord_meters, out_dir=out_dir)

        pred_coord_list_all += pred_coord_meters
        gt_coord_list_all += gt_coord_meters

    # print(pred_coord_list_all)
    # print(gt_coord_list_all)
    # comprehensive_grid = comprehensive_eval(pred_grid_list_all, gt_grid_list_all)
    comprehensive_grid = comprehensive_eval(pred_coord_list_all, gt_coord_list_all)
    # print(comprehensive_grid)

    os.makedirs(f"../ask_questions/bev_grid_map", exist_ok=True)
    with open(f"../ask_questions/bev_grid_map/result_{grid_size}.json", 'w') as f:
        json.dump(comprehensive_grid, f, indent=4)
#     Do some sort of evaluation for all coords in the pred_coord_list_all and gt_coord_list_all

if __name__ == "__main__":
    main()
