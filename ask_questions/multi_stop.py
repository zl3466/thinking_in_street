import json
import os
import re
import time

from model.gemini import GeminiModel
from model.gpt4v import GPT4VModel
from model.claude import ClaudeModel
from config import LLM_PROVIDER
from utils import *

from bs4 import BeautifulSoup
import random

def get_frame_number(filename):
    """Extract frame number from filename"""
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else 0


def get_model():
    """Get the appropriate model based on config"""
    if LLM_PROVIDER == "gemini":
        return GeminiModel()
    elif LLM_PROVIDER == "openai":
        return GPT4VModel()
    elif LLM_PROVIDER == "anthropic":
        return ClaudeModel()
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def analyze_street_view(image_directory, question_list, out_dir):
    """Analyze street view images in a directory and generate spatial data"""
    # Get and sort image paths from frames directory
    frames_dir = os.path.join(image_directory, "frames")
    # route_json = f"{image_directory}/route_data.json"

    if os.path.exists(frames_dir):
        image_directory = frames_dir

    # Get all image paths
    image_paths = []
    for image_file in os.listdir(image_directory):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_directory, image_file)
            image_paths.append(full_path)

    if not image_paths:
        print(f"No images found in {image_directory}!")
        return None

    # Sort by frame number in reverse order
    image_paths = sorted(image_paths, key=lambda x: get_frame_number(os.path.basename(x)), reverse=True)
    print(f"\nFound {len(image_paths)} images in {image_directory}")

    try:
        # Initialize model based on config
        model = get_model()
        results = model.analyze_images(image_paths, question_list, out_dir)
        return results

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


def generate_prompt_direction(route_json, use_distance=True, use_time=False):
    if use_distance and use_time:
        RuntimeError("You may only use one of distance or time as hint for the model")
    route = route_json["routes"][0]

    steps = route["legs"][0]["steps"]
    full_prompt = f"Choose from the four options below to fill in the blanks [?] in the following prompt with " \
                  f"direction keywords 'left' or 'right' to make the prompt describe a correct traversal route:\n"
    after_destination = ""
    destination_side = ""
    answer_list = []
    for i in range(len(steps)):
        step = steps[i]
        travel_mode = step["travel_mode"]

        step_html = step["html_instructions"]
        step_text = BeautifulSoup(step_html, "html.parser").get_text()

        if "Destination" in step_text and i == len(steps) - 1:
            before_destination, after_destination = step_text.split("Destination", 1)
            match = re.search(r'\b(left|right)\b', after_destination, re.IGNORECASE)
            if match:
                destination_side = match.group()  # Store the keyword
                after_destination = re.sub(r'\b' + destination_side + r'\b', '[?]', after_destination, flags=re.IGNORECASE)

            step_text = before_destination

        if use_distance:
            step_distance_text = step["distance"]["text"]
            step_text += f", then keep {travel_mode} for approximately {step_distance_text}. "
        elif use_time:
            step_duration_text = step["duration"]["text"]
            step_text += f", then keep {travel_mode} for approximately {step_duration_text}. "
        else:
            step_text += ".\n"

        if "maneuver" in step.keys():
            turn_direction = step["maneuver"].split("-")[1]
            step_text = re.sub(rf'\b{turn_direction}\b', '[?]', step_text)
            answer_list.append(turn_direction)

        if destination_side != "":
            answer_list.append(destination_side)
            destination_side = ""
            step_text += f"The destination{after_destination}."
        full_prompt += step_text

    wrong_ans_list = generate_random_ans(answer_list, num_wrong_ans=3)
    return full_prompt, answer_list, wrong_ans_list


def generate_prompt_distance(route_json, use_time=False):
    route = route_json["routes"][0]

    steps = route["legs"][0]["steps"]
    full_prompt = f"Choose from the four options below to fill in the blanks [?] in the following prompt with " \
                  f"distance values in different units to make the prompt describe a correct traversal route:\n"
    after_destination = ""
    destination_side = ""
    answer_list = []
    for i in range(len(steps)):
        step = steps[i]
        travel_mode = step["travel_mode"]

        step_html = step["html_instructions"]
        step_text = BeautifulSoup(step_html, "html.parser").get_text()

        if "Destination" in step_text and i == len(steps) - 1:
            before_destination, after_destination = step_text.split("Destination", 1)
            step_text = before_destination

        step_distance_text = step["distance"]["text"]
        step_text += f", then keep {travel_mode} for approximately [?]. "
        if use_time:
            step_duration_text = step["duration"]["text"]
            step_text += f" for {step_duration_text}. "
        else:
            step_text += "\n"

        # if "maneuver" in step.keys():
        #     turn_direction = step["maneuver"].split("-")[1]
        #     step_text = re.sub(rf'\b{turn_direction}\b', '[?]', step_text)
        #     answer_list.append(turn_direction)
        answer_list.append(step_distance_text)
        if destination_side != "":
            destination_side = ""
            step_text += f"The destination{after_destination}."
        full_prompt += step_text

    wrong_ans_list = generate_random_ans(answer_list, num_wrong_ans=3)

    return full_prompt, answer_list, wrong_ans_list


def generate_random_ans(ans_list, num_wrong_ans=3):
    wrong_ans_list = []
    for _ in range(num_wrong_ans):
        if ans_list[0] == "left" or ans_list[0] == "right":
            random_ans = random.choices(["left", "right"], k=len(ans_list))
            wrong_ans_list.append(random_ans)

        else:
            random_ans_dict = {}
            for ans in ans_list:
                val = float(ans.split(" ")[0])
                unit = ans.split(" ")[1]
                if unit not in random_ans_dict.keys():
                    random_ans_dict[unit] = [val]
                else:
                    random_ans_dict[unit].append(val)
            random_unit_list = random.choices(list(random_ans_dict.keys()), k=len(ans_list))

            wrong_ans = []
            for unit in random_unit_list:
                random_val = generate_random_within_range(list(random_ans_dict[unit]))
                wrong_ans.append(f"{random_val} {unit}")

            wrong_ans_list.append(wrong_ans)
    return wrong_ans_list


def generate_random_within_range(values):
    if len(values) == 1:
        min_val = values[0] * 0.2
        max_val = values[0] * 2
    else:
        min_val = min(values)
        max_val = max(values)

    value_range = max_val - min_val
    # print(f"range: {min_val - value_range}, {max_val + value_range}")
    random_value = random.uniform(min_val - value_range, max_val + value_range)

    # Ensure the value is greater than 0
    while random_value <= 0:
        random_value = random.uniform(min_val - value_range, max_val + value_range)
    random_value = round(random_value, 1)
    return random_value


def generate_prompt_and_choices(origin, destination, api_key, mode="driving", way_points=None, task="direction"):
    '''
        Input:
            origin
            final destination
            google maps api key
            navigation mode (driving, walking, transport)
            waypoints (middle stop points)
            task: prompt task type, "direction" to ask about left and right turns, "distance" to ask about travel distance at each step
    '''
    url = "https://maps.googleapis.com/maps/api/directions/json"
    if way_points is not None:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "waypoints": "|".join(way_points),
            "key": api_key
        }
    else:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": api_key
        }
    response = requests.get(url, params=params)
    data = response.json()

    if task == "direction":
        full_prompt, answer_list, wrong_ans_list = generate_prompt_direction(data)
    elif task == "distance":
        full_prompt, answer_list, wrong_ans_list = generate_prompt_distance(data)
    else:
        return RuntimeError("wrong task type provided for prompt generation")

    correct_idx = random.randint(0, len(wrong_ans_list))
    wrong_ans_list.insert(correct_idx, answer_list)
    final_choices = wrong_ans_list

    return full_prompt, final_choices, correct_idx


idx2letter = ["A", "B", "C", "D"]
def main():
    data_dir = "../data/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop"
    map_list = os.listdir(data_dir)

    map_name = "map5_chinatown"
    # for map_name in map_list:
    print("====================================")
    print(f"Processing Map {map_name}")

    map_root_dir = f"{data_dir}/{map_name}"
    out_dir = f"{data_dir}/{map_name}/multi_stop"
    if not os.path.exists(map_root_dir):
        print(f"Directory not found: {map_root_dir}")
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    location_dict = json.load(open(f"{question_dir}/{map_name}.json"))
    question_location_list = location_dict["Q"]
    GT_location_list = location_dict["GT"]
    GT_optimal_location_list = location_dict["GT_optimal"]
    walk_nav_template = location_dict["walk_nav_template"]
    walk_nav_choices = location_dict["walk_nav_choices"]
    walk_nav_ans = location_dict["walk_nav_ans"]
    drive_nav_template = location_dict["drive_nav_template"]
    drive_nav_choices = location_dict["drive_nav_choices"]
    drive_nav_ans = location_dict["drive_nav_ans"]
    optimal_nav_choices = location_dict["optimal_nav_choices"]
    optimal_nav_ans = location_dict["optimal_nav_ans"]

    formatted_locations = ", ".join(question_location_list)

    question_list = [
        "Give me five landmarks you saw in this footage.",

        f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
        f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
        f"following multiple choice question:\n"
        f"I want to traverse through the locations by foot: {formatted_locations} in this exact order, with the "
        f"starting position being facing {question_location_list[0]}. Since you are walking by foot, you do not "
        f"need to consider constraints such as a street being one-way drive."
        f"Choose from the four options below to fill in the blanks [?] with direction keyword 'right' or 'left' "
        f"in the following prompt to make it describe a correct traversal route: {walk_nav_template}\n"
        f"A. {walk_nav_choices[0]}      B. {walk_nav_choices[1]}\n"
        f"C. {walk_nav_choices[2]}      D. {walk_nav_choices[3]}\n",

        f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
        f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
        f"following multiple choice question:\n"
        f"I want to traverse through the locations by driving: {formatted_locations} in this exact order, with the "
        f"starting position being facing {question_location_list[0]}. Since you are driving, you must consider "
        f"constraints such as a street being one-way drive."
        f"Choose from the four options below to fill in the blanks [?] with direction keyword 'right' or 'left' "
        f"in the following prompt to make it describe a correct traversal route: {drive_nav_template}\n"
        f"A. {drive_nav_choices[0]}      B. {drive_nav_choices[1]}\n"
        f"C. {drive_nav_choices[2]}      D. {drive_nav_choices[3]}\n",

        f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
        f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
        f"following multiple choice question:\n"
        f"Can you choose the optimal (time-efficient) order of visiting all these locations: {formatted_locations} "
        f"within this area by driving? Since you are driving, you must consider constraints such as one-way drive "
        f"streets.\n"
        f"A. {optimal_nav_choices[0]}      B. {optimal_nav_choices[1]}\n"
        f"C. {optimal_nav_choices[2]}      D. {optimal_nav_choices[3]}\n",

    ]

    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    results = analyze_street_view(map_root_dir, question_list, out_dir=map_root_dir)

    ''' ================== save questions and answers in md and json ================== '''
    ''' the json could be used later for auto eval? '''
    if results:
        print("\nAnalysis Summary:")
        if os.path.exists(f"{out_dir}/Q&A.md"):
            os.remove(f"{out_dir}/Q&A.md")
        idx = 0
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")
            with open(f"{out_dir}/Q&A.md", "a", encoding="utf-8") as file:
                file.write(f"{question}")
                file.write(f"{answer}")
                file.write(f"\n----------\nTrue Correct Answer: {idx2letter[[walk_nav_ans, drive_nav_ans, optimal_nav_ans][idx-1]]}\n==========\n")
            idx += 1
        with open(f"{out_dir}/Q&A.json", 'w') as f:
            json.dump(results, f, indent=4)

    ''' ================== Save Walking Ground Truth Route Data ================== '''
    ''' the optimal order is manually regulated, and the route is generated by Google Maps '''
    route_data_walk = {
        "map": map_name,
        "ans": idx2letter[walk_nav_ans],
        "frames": [],
    }
    all_positions_and_headings, _, _ = get_routes_from_locations(GT_location_list,
                                                                 route_file_output=f"{out_dir}/route_map_multi_stop_walikng.html",
                                                                 route_mode="walking")
    for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                   total=len(all_positions_and_headings)):
        # Store frame data with the movement from all_positions_and_headings
        route_data_walk["frames"].append({
            "coordinates": {"lat": lat, "lng": lng},
            "heading": heading,
            "movement": movement  # This now correctly uses the movement we determined earlier
        })

    # Save route data to JSON
    route_data_file = f"{out_dir}/route_data_multi_stop_walking.json"
    with open(route_data_file, 'w') as f:
        json.dump(route_data_walk, f, indent=2)

    ''' ================== Save Driving Ground Truth Route Data ================== '''
    ''' the optimal order is manually regulated, and the route is generated by Google Maps '''
    route_data_drive = {
        "map": map_name,
        "ans": idx2letter[drive_nav_ans],
        "frames": [],
    }
    all_positions_and_headings, _, _ = get_routes_from_locations(GT_location_list,
                                                                 route_file_output=f"{out_dir}/route_map_multi_stop_driving.html",
                                                                 route_mode="driving")
    for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                   total=len(all_positions_and_headings)):
        # Store frame data with the movement from all_positions_and_headings
        route_data_drive["frames"].append({
            "coordinates": {"lat": lat, "lng": lng},
            "heading": heading,
            "movement": movement  # This now correctly uses the movement we determined earlier
        })

    # Save route data to JSON
    route_data_file = f"{out_dir}/route_data_multi_stop_driving.json"
    with open(route_data_file, 'w') as f:
        json.dump(route_data_drive, f, indent=2)

    ''' ================== Save Optimal Route Ground Truth Route Data ================== '''
    route_data_optimal = {
        "map": map_name,
        "ans": idx2letter[optimal_nav_ans],
        "frames": [],
    }
    all_positions_and_headings, _, _ = get_routes_from_locations(GT_optimal_location_list, f"{out_dir}/route_map_multi_stop_optimal.html")
    for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                   total=len(all_positions_and_headings)):
        # Store frame data with the movement from all_positions_and_headings
        route_data_optimal["frames"].append({
            "coordinates": {"lat": lat, "lng": lng},
            "heading": heading,
            "movement": movement  # This now correctly uses the movement we determined earlier
        })

    # Save optimal route data to JSON
    route_data_file = f"{out_dir}/route_data_multi_stop_optimal.json"
    with open(route_data_file, 'w') as f:
        json.dump(route_data_optimal, f, indent=2)


if __name__ == "__main__":
    main()
