import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

    plt.savefig(f"{out_dir}/coord_plot.png")

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


def main():
    data_dir = "../data/long_route_random_single"
    result_dir = "../result/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop_auto"

    map_list = os.listdir(data_dir)
    pred_coord_list_all = []
    gt_coord_list_all = []

    # map_name = "map2_soho"
    for map_name in map_list:
        print(f"map {map_name}")
        json_path = f"../data/long_route_random_single/{map_name}/coord_map/Q&A.json"
        out_dir = f"../data/long_route_random_single/{map_name}/coord_map"

        full_response_txt = json.load(open(json_path))
        # print(full_response_txt[0])
        pred_coord_dict = parse_json_from_response(full_response_txt["answers"][0])

        location_name_list = list(pred_coord_dict.keys())
        pred_coord_list_normalized = list(pred_coord_dict.values())

        gt_coord_list_normalized = full_response_txt["gt"]


        plot_coord(location_name_list, pred_coord_list_normalized, gt_coord_list_normalized, out_dir=out_dir)
        rmse, med = calculate_errors(pred_coord_list_normalized, gt_coord_list_normalized)

        pred_coord_list_all += pred_coord_list_normalized[1:]
        gt_coord_list_all += gt_coord_list_normalized[1:]

    # print(pred_coord_list_all)
    # print(gt_coord_list_all)
    comprehensive_mtr = comprehensive_eval(pred_coord_list_all, gt_coord_list_all)
    print(comprehensive_mtr)

    os.makedirs(f"../ask_questions/coord_map", exist_ok=True)
    with open(f"../ask_questions/coord_map/result.json", 'w') as f:
        json.dump(comprehensive_mtr, f, indent=4)
#     Do some sort of evaluation for all coords in the pred_coord_list_all and gt_coord_list_all

if __name__ == "__main__":
    main()
