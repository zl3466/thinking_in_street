import ast
import itertools
import json
import os
import random
import re
import urllib.parse

import requests
from google import genai
from bs4 import BeautifulSoup

from utils import *

client = genai.Client(api_key="GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


def parse_json(json_output):
    lines = json_output.splitlines()
    json_content = ""
    in_json_block = False

    for line in lines:
        if line.strip() == "```json":
            in_json_block = True
        elif line.strip() == "```" and in_json_block:
            in_json_block = False  # End of JSON block
        elif in_json_block:
            json_content += line + "\n"

    return json_content.strip()


def generate_prompt(origin, destination, api_key, mode="driving", way_points=None):
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
    print(data)
    # full_prompt, answer_list, wrong_ans_list = prompt_distance(data, use_time=False)

    full_prompt, answer_list, wrong_ans_list = generate_prompt_direction(data)
    with open("sample_route_instruction.json", 'w') as f:
        json.dump(data, f, indent=4)

    print(full_prompt)
    print(answer_list)
    # correct_idx = random.randint(0, len(wrong_ans_list))
    # wrong_ans_list.insert(correct_idx, answer_list)
    # final_choices = wrong_ans_list
    #
    # return final_choices, correct_idx


    # route = data["routes"][0]
    # if data["status"] != "OK":
    #     raise ValueError(f"Directions API error: {data.get('error_message', data['status'])}")
    # distance = route["legs"][0]["distance"]["text"]
    # duration = route["legs"][0]["duration"]["text"]
    # return route["overview_polyline"]["points"], distance, duration


def generate_prompt_direction(route_json, use_distance=True, use_time=False):
    if use_distance and use_time:
        RuntimeError("You may only use one of distance or time as hint for the model")
    route = route_json["routes"][0]

    full_prompt = f"Choose from the four options below to fill in the blanks [?] in the following prompt with " \
                  f"direction keywords 'left' or 'right' to make the prompt describe a correct traversal route:\n"
    answer_list = []
    for i in range(len(route["legs"])):
        leg = route["legs"][i]
        steps = leg["steps"]

        after_destination = ""
        destination_side = ""
        if i != 0:
            full_prompt += "To continue to the next destination, "
        for j in range(len(steps)):
            step = steps[j]
            travel_mode = step["travel_mode"]

            step_html = step["html_instructions"]
            step_text = BeautifulSoup(step_html, "html.parser").get_text()

            if "Destination" in step_text and j == len(steps) - 1:
                before_destination, after_destination = step_text.split("Destination", 1)
                match = re.search(r'\b(left|right)\b', after_destination, re.IGNORECASE)
                if match:
                    destination_side = match.group()  # Store the keyword
                    after_destination = re.sub(r'\b' + destination_side + r'\b', '[?]', after_destination,
                                               flags=re.IGNORECASE)

                step_text = before_destination

            if use_distance:
                step_distance_text = step["distance"]["text"]
                step_text += f", then keep {travel_mode} for approximately {step_distance_text}.\n"
            elif use_time:
                step_duration_text = step["duration"]["text"]
                step_text += f", then keep {travel_mode} for approximately {step_duration_text}.\n"
            else:
                step_text += ".\n"

            if "maneuver" in step.keys():
                turn_direction = step["maneuver"].split("-")[1]
                step_text = re.sub(rf'\b{turn_direction}\b', '[?]', step_text)
                answer_list.append(turn_direction)

            if destination_side != "":
                answer_list.append(destination_side)
                destination_side = ""
                step_text += f"The destination{after_destination}.\n"
            full_prompt += step_text

    wrong_ans_list = generate_random_ans(answer_list, num_wrong_ans=3)
    return full_prompt, answer_list, wrong_ans_list


def generate_prompt_distance(route_json, use_time=False):
    route = route_json["routes"][0]

    full_prompt = f"Choose from the four options below to fill in the blanks [?] in the following prompt with " \
                  f"distance values in different units to make the prompt describe a correct traversal route:\n"

    answer_list = []
    for i in range(len(route["legs"])):
        after_destination = ""

        leg = route["legs"][i]
        steps = leg["steps"]
        if i != 0:
            full_prompt += "To continue to the next destination, "
        for j in range(len(steps)):
            step = steps[j]
            travel_mode = step["travel_mode"]

            step_html = step["html_instructions"]
            step_text = BeautifulSoup(step_html, "html.parser").get_text()

            if "Destination" in step_text and j == len(steps) - 1:
                before_destination, after_destination = step_text.split("Destination", 1)
                step_text = before_destination

            step_distance_text = step["distance"]["text"]
            step_text += f", then keep {travel_mode} for approximately [?]. "
            if use_time:
                step_duration_text = step["duration"]["text"]
                step_text += f" for {step_duration_text}. "
            else:
                step_text += "\n"

            answer_list.append(step_distance_text)
            if after_destination != "":
                step_text += f"The destination{after_destination}.\n"
                after_destination = ""
            full_prompt += step_text

    wrong_ans_list = generate_random_ans(answer_list, num_wrong_ans=3)

    return full_prompt, answer_list, wrong_ans_list


# def generate_random_ans(ans_list, num_wrong_ans=3):
#     wrong_ans_list = []
#     for _ in range(num_wrong_ans):
#         if isinstance(ans_list[0], int):
#             wrong_ans_list.append(random.sample(ans_list, len(ans_list)))
#         elif ans_list[0] == "left" or ans_list[0] == "right":
#             random_ans = random.choices(["left", "right"], k=len(ans_list))
#             wrong_ans_list.append(random_ans)
#         else:
#             random_ans_dict = {}
#             for ans in ans_list:
#                 val = float(ans.split(" ")[0])
#                 unit = ans.split(" ")[1]
#                 if unit not in random_ans_dict.keys():
#                     random_ans_dict[unit] = [val]
#                 else:
#                     random_ans_dict[unit].append(val)
#             random_unit_list = random.choices(list(random_ans_dict.keys()), k=len(ans_list))
#
#             wrong_ans = []
#             for unit in random_unit_list:
#                 random_val = generate_random_within_range(list(random_ans_dict[unit]))
#                 wrong_ans.append(f"{random_val} {unit}")
#
#             wrong_ans_list.append(wrong_ans)
#     return wrong_ans_list

def generate_random_ans(ans_list, num_wrong_ans=3):
    seen_answers = set()  # To track seen wrong answers and prevent duplicates
    wrong_ans_list = []
    #   optimal route prompt
    if isinstance(ans_list[0], int):
        # If the answers are integers
        while len(wrong_ans_list) < num_wrong_ans:
            random_ans = random.sample(ans_list, len(ans_list))
            # Check for duplicates in the generated answer
            if tuple(random_ans) not in seen_answers:
                seen_answers.add(tuple(random_ans))
                wrong_ans_list.append(random_ans)
    #   direction prompt
    elif ans_list[0] == "left" or ans_list[0] == "right":
        # If the answers are directional ("left" or "right")
        while len(wrong_ans_list) < num_wrong_ans:
            random_ans = random.choices(["left", "right"], k=len(ans_list))
            if tuple(random_ans) not in seen_answers:
                seen_answers.add(tuple(random_ans))
                wrong_ans_list.append(random_ans)
    #   distance prompt
    else:
        # If the answers are of the form "value unit"
        random_ans_dict = {}
        for ans in ans_list:
            val = float(ans.split(" ")[0])
            unit = ans.split(" ")[1]
            if unit not in random_ans_dict:
                random_ans_dict[unit] = [val]
            else:
                random_ans_dict[unit].append(val)
        random_unit_list = random.choices(list(random_ans_dict.keys()), k=len(ans_list))

        while len(wrong_ans_list) < num_wrong_ans:
            wrong_ans_temp = []
            for unit in random_unit_list:
                random_val = generate_random_within_range(list(random_ans_dict[unit]))
                wrong_ans_temp.append(f"{random_val} {unit}")
            # Check for duplicates in the generated answers
            if tuple(wrong_ans_temp) not in seen_answers:
                seen_answers.add(tuple(wrong_ans_temp))
                wrong_ans_list.append(wrong_ans_temp)

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


def get_optimized_route(api_key, origin, destination, waypoints):
    base_url = "https://maps.googleapis.com/maps/api/directions/json?"

    # Encode the origin, destination, and waypoints
    origin_encoded = urllib.parse.quote_plus(origin)
    destination_encoded = urllib.parse.quote_plus(destination)
    waypoints_encoded = "|".join([urllib.parse.quote_plus(w) for w in waypoints])

    # Construct the URL with the encoded parameters
    url = f"{base_url}origin={origin_encoded}&destination={destination_encoded}&waypoints=optimize:true|{waypoints_encoded}&key={api_key}"

    # Send the request to the API
    response = requests.get(url)

    # Parse the response
    directions = response.json()

    if directions['status'] == 'OK':
        return directions['routes'][0]['waypoint_order']
    else:
        return directions['status']


# Example usage
# origin = "New York, NY"
# destination = "Los Angeles, CA"
# waypoints = ["Chicago, IL", "Philadelphia, PA", "Denver, CO"]
origin = "Goodwill NYNJ Store & Donation Center, 258 Livingston St, Brooklyn, NY 11201"
destination = "The Brooklyn Cancer Center, 86 Fleet Pl, Brooklyn, NY 11201"
waypoints = [
    "Brooklyn Paramount, 385 Flatbush Ave Ext, Brooklyn, NY 11201",
    "Atlantic Terminal, Brooklyn, NY 11217",
    "Goodwill NYNJ Store & Donation Center, 258 Livingston St, Brooklyn, NY 11201"
]


def generate_prompt_optimization(location_list):
    idx_list = list(range(len(location_list)))
    permutations = list(itertools.permutations(idx_list))
    optimal_permutation = permutations[0]
    min_distance = -1
    for permutation in permutations:
        permutation = list(permutation)
        location_reordered = [location_list[i] for i in permutation]

        way_points = []
        for i in range(1, len(location_reordered) - 1):
            location = location_reordered[i]

            way_points.append(location)
        start = location_reordered[0]
        end = location_reordered[-1]

        route_json, overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY, mode="driving",
                                                                       way_points=way_points)
        full_distance = route_json["legs"][0]["distance"]["value"]

        if min_distance == -1:
            min_distance = full_distance
            optimal_permutation = permutation
        else:
            if full_distance < min_distance:
                min_distance = full_distance
                optimal_permutation = permutation
    wrong_ans_list = generate_random_ans(optimal_permutation, num_wrong_ans=3)

    return optimal_permutation


def extract_model_answers(text):
    pattern = r"\[\[\[(.*?)\]\]\]"  # Non-greedy match inside [[[...]]]
    return re.findall(pattern, text)

# def extract_true_answers(text):
#     pattern = r"^-+\s*True Correct Answer:\s*(\S+)\s*=+"  # Match text between 'True Correct Answer:' and '==='
#     return re.findall(pattern, text, re.MULTILINE)

def extract_gt_answers(text):
    pattern = r"^-+\s*True Correct Answer:\s*([A-Z])\s*=+"  # Captures a single uppercase letter
    return re.findall(pattern, text, re.MULTILINE)

# # Example Usage
# result_dir = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/result/long_route_random_single"
# map_list = os.listdir(result_dir)
#
# final_results = {}
# for map_name in map_list:
#     set_list = os.listdir(f"{result_dir}/{map_name}/multi_stop_auto")
#     final_results[map_name] = {}
#     for set_name in set_list:
#         print(f"\nmap: {map_name}, set: {set_name}")
#         final_results[map_name][set_name] = {"model_answers": [], "gt_answers": []}
#         qna_json_file = f"{result_dir}/{map_name}/multi_stop_auto/{set_name}/Q&A.json"
#         if not os.path.exists(qna_json_file):
#             print("Analysis failed for this set. Continuing...")
#             continue
#         qna_dict = json.load(open(qna_json_file))
#         # with open(qna_md_file, "r", encoding="utf-8") as f:
#         #     md_text = f.read()
#
#         q_list = list(qna_dict.keys())
#         for q in q_list:
#             a = qna_dict[q]
#
#             model_answers = extract_model_answers(a)
#             gt_answers = extract_gt_answers(a)
#             # if len(model_answers) != len(true_answers):
#             #     print(f"Warning: map {map_name} set {set_name} has ill-formatted answers")
#             print(f"model answer: {model_answers}, true answer: {gt_answers}")
#
#             final_results[map_name][set_name]["model_answers"].append(model_answers)
#             final_results[map_name][set_name]["gt_answers"].append(gt_answers)
#
#
# with open(f"../final_result.json", 'w') as f:
#     json.dump(final_results, f, indent=4)

def get_active_project_id():
    """Retrieve the active Google Cloud project ID from gcloud configuration."""

    # Run gcloud command to get active project ID
    result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
    return result.stdout.strip()



def check_gemini_quota_with_api_key(api_key):
    """Check Gemini API quota usage using API key and active project."""

    project_id = get_active_project_id()
    if not project_id:
        print("Project ID not found. Please ensure you are authenticated.")
        return

    # The URL for the Google Service Usage API for Vertex AI
    quota_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/aiplatform.googleapis.com"

    # The headers with the API Key for authentication
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    # Make the request to fetch quota information
    response = requests.get(quota_url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        if 'quota' in data:
            for quota in data['quota']['limits']:
                metric_name = quota['metric']
                limit = quota['effectiveLimit']
                usage = quota.get('usage', 0)  # Get usage (default to 0 if missing)

                if limit and usage:
                    usage_percentage = (usage / limit) * 100
                    print(f"{metric_name}: {usage}/{limit} ({usage_percentage:.2f}%)")

                    if usage_percentage > 90:
                        print(f"⚠️ WARNING: {metric_name} is over 90% usage!")
    else:
        print(f"Error: {response.status_code}, {response.text}")


def convert_to_feet(value):
    """Convert distance strings to feet."""
    num, unit = value.split()
    num = float(num)
    if unit == "mi":  # Convert miles to feet
        return num * 5280
    return num  # Already in feet

def mean_squared_error(list1, list2):
    """Compute MSE between two lists of distance values."""
    feet1 = [convert_to_feet(x) for x in list1]
    feet2 = [convert_to_feet(x) for x in list2]

    squared_diffs = [(a - b) ** 2 for a, b in zip(feet1, feet2)]
    mse = sum(squared_diffs) / len(squared_diffs)
    return mse

def extract_choices(text):
    """Extract multiple-choice options (A, B, C, D) from a given string."""
    pattern = r"A\. (\[.*?\])\s*B\. (\[.*?\])\s*C\. (\[.*?\])\s*D\. (\[.*?\])"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        choices = {label: ast.literal_eval(match.group(i + 1)) for i, label in enumerate("ABCD")}
        return choices
    return None

text = """
A. ['right', 'left', 'right', 'right']      
B. ['left', 'right', 'left', 'left']
C. ['left', 'right', 'right', 'right']      
D. ['right', 'right', 'right', 'right']
"""

# # Extract choices
# choices = extract_choices(text)
#
# if choices:
#     for key, value in choices.items():
#         print(f"{key}: {value}")
# else:
#     print("No choices found!")


def is_ft_mi_list(lst):
    # Check if the list contains strings with exactly "ft" or "mi" as part of their value
    return all(" ft" in item or " mi" in item for item in lst)
