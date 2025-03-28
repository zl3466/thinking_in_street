import sys
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def check_uploaded_file(file_path):
    creation_time = os.path.getctime(file_path)
    current_time = time.time()

    # Check if the file is older than 2 days (2 * 24 * 60 * 60 seconds)
    if (current_time - creation_time) > (2 * 86400):
        os.remove(file_path)
        print(f"The images have been uploaded but already expired, removing existing filename records: {file_path}")
    else:
        print(f"The images have been uploaded and are still valid on Gemini: {file_path}")


idx2letter = ["A", "B", "C", "D"]


def main():
    data_dir = "../data/long_route_random_single"
    result_dir = "../result/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop_auto"

    map_list = os.listdir(data_dir)

    # map_name = "map2_soho"
    for map_name in map_list:
        # for map_name in map_list:
        print("====================================")
        print(f"Processing Map {map_name}")

        map_root_dir = f"{data_dir}/{map_name}"
        out_dir = f"{data_dir}/{map_name}/coord_map"
        if not os.path.exists(map_root_dir):
            print(f"Directory not found: {map_root_dir}")
            return
        os.makedirs(out_dir, exist_ok=True)

        location_dict = json.load(open(f"{question_dir}/{map_name}.json"))
        location_list = list(location_dict.keys())
        gt_location_list = list(location_dict.values())
        gt_coord_list = [get_coord(location, api_key=GOOGLE_MAPS_API_KEY) for location in gt_location_list]
        gt_coord_list_normalized = normalize_coords(gt_coord_list)

        question_list = [
            f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
            f"within an area. You should be able to see these locations: {location_list}.\n"
            f"Suppose the first location {location_list[0]} is at [x, y] = [0, 0], "
            f"where positive x axis points to East direction, and positive y points to North direction."
            f"Give me the (x, y) coordinate of the other 5 locations {location_list[1:]} in the same coordinate frame, "
            f"using meter as the unit for x and y.\n"
            f"Give your answer in json format, and be sure to include the first location {location_list[0]} "
            f"at [0, 0] in your answer:\n "
            f"```json{{"
            f"{location_list[0]}: [0, 0],\n"
            f"location_name1: [x, y],\n"
            f"location_name2: [x, y],\n"
            f"location_name3: [x, y],\n"
            f"...}}\n"
            f"```\n"
            f"where location_name is the name of a location, "
            f"and [x, y] is the coordinate of the location in meters away from {location_list[0]}.\n",
        ]

        ''' ================== Inference ================== '''
        ''' upload images to model and ask questions '''
        results, _ = analyze_street_view(map_root_dir, question_list, out_dir=map_root_dir)

        ''' ================== save questions and answers in md and json ================== '''
        ''' the json could be used later for auto eval? '''
        if results:
            print("\nAnalysis Summary:")
            if os.path.exists(f"{out_dir}/Q&A.md"):
                os.remove(f"{out_dir}/Q&A.md")

            for question, answer in results.items():
                print(f"\n{question}")
                print(f"{answer}")
                with open(f"{out_dir}/Q&A.md", "a", encoding="utf-8") as file:
                    file.write(f"{question}")
                    file.write(f"{answer}")

            with open(f"{out_dir}/Q&A.json", 'w') as f:
                answers = list(results.values())
                json_dict = {"answers": answers,
                             "gt": gt_coord_list_normalized}
                json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    main()
