import ast
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *


def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculate the heading direction angle between two coordinate points.

    Parameters:
    lat1, lon1 (float): Latitude and Longitude of the first point in degrees.
    lat2, lon2 (float): Latitude and Longitude of the second point in degrees.

    Returns:
    float: Heading angle in degrees from North (0-360).
    """
    delta_lon = math.radians(lon2 - lon1)
    lat1, lat2 = map(math.radians, [lat1, lat2])

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    heading = math.atan2(x, y)

    return (math.degrees(heading) + 360) % 360


def eval_heading_displacement(response_path, frame_data, out_dir):
    full_response_txt = json.load(open(response_path))
    json_data = parse_json_from_response(list(full_response_txt.values())[0])
    inferred_heading = json_data["delta_heading"]
    inferred_displacement = json_data["displacement"]

    # heading_offset = frame_data[0]["heading"]
    prev_heading_gt = frame_data[0]["heading"]
    prev_lat = frame_data[0]["coordinates"]["lat"]
    prev_lng = frame_data[0]["coordinates"]["lng"]
    filename_prev = frame_data[0]["filename"]

    gt_delta_heading_list = []
    gt_displacement_list = []
    results = {"RMSE_delta_heading": 0, "RMSE_displacement": 0, "frames": []}
    for i in range(len(inferred_heading)):
        frame = frame_data[i + 1]
        filename = frame["filename"]
        lat = frame["coordinates"]["lat"]
        lng = frame["coordinates"]["lng"]
        heading_gt = frame_data[i + 1]["heading"]
        movement = frame["movement"]

        # print(filename)

        delta_heading_gt = (heading_gt - prev_heading_gt) if (heading_gt - prev_heading_gt) <= 180 \
            else (heading_gt - prev_heading_gt) - 360
        displacement_gt = geodesic((prev_lat, prev_lng), (lat, lng)).meters
        # print(prev_heading_gt, heading_gt, delta_heading_gt)

        results["frames"].append({"prev": filename_prev,
                              "curr": filename,
                              "inferred_delta_heading": inferred_heading[i],
                              "delta_heading_gt": delta_heading_gt,
                              "inferred_displacement": inferred_displacement[i],
                              "displacement_gt": displacement_gt,
                              "movement": movement})
        gt_delta_heading_list.append(delta_heading_gt)
        gt_displacement_list.append(displacement_gt)

        prev_heading_gt = heading_gt
        prev_lat = lat
        prev_lng = lng
        filename_prev = filename

    mse_delta_heading = np.mean((np.array(inferred_heading) - np.array(gt_delta_heading_list)) ** 2)
    rmse_delta_heading = np.sqrt(mse_delta_heading)
    mse_displacement = np.mean((np.array(inferred_displacement) - np.array(gt_displacement_list)) ** 2)
    rmse_displacement = np.sqrt(mse_displacement)

    print(f"RMSE delta heading (degrees 0-360): {rmse_delta_heading}; RMSE displacement (meters): {rmse_displacement}")

    with open(f"{out_dir}/odometry_result.json", 'w') as f:
        json.dump(results, f, indent=4)


data_dir = "../../data/long_route_random_single"
result_dir = "../../result/long_route_random_single"
map_list = os.listdir(data_dir)
for map_name in map_list:
    if map_name != "map1_downtown_bk":
        continue
    print("====================================")
    print(f"Processing Map {map_name}")
    map_root_dir = f"{data_dir}/{map_name}"
    out_dir = f"{result_dir}/odometry/{map_name}"

    if os.path.exists(f"{out_dir}/batches"):
        batch_list = os.listdir(f"{out_dir}/batches")
    else:
        batch_list = []

    gt_route_path = f"{map_root_dir}/route_data.json"
    gt_route_data_full = json.load(open(gt_route_path))
    gt_frame_data_full = gt_route_data_full["frames"]
    if len(batch_list) != 0:
        for batch_name in batch_list:
            response_path = f"{out_dir}/batches/{batch_name}/Q&A.json"
            batch_idx_pair = ast.literal_eval(batch_name.split("_")[-1])
            print(batch_idx_pair)
            gt_route_data = gt_frame_data_full[batch_idx_pair[0]:batch_idx_pair[1]]
            eval_heading_displacement(response_path, gt_route_data, f"{out_dir}/batches/{batch_name}")


