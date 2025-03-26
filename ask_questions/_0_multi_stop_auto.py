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
import itertools


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
                    after_destination = re.sub(r'\b' + destination_side + r'\b', '[?]', after_destination, flags=re.IGNORECASE)

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
                maneuver = step["maneuver"]
                turn_direction = maneuver.split("-")[-1]
                if turn_direction == "left" or turn_direction == "right":
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
                  f"distance values in different units to make the prompt describe the correct traversal route:\n"

    answer_list = []
    for i in range(len(route["legs"])):
        after_destination = ""
        destination_side = ""

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
            if destination_side != "":
                destination_side = ""
                step_text += f"The destination{after_destination}.\n"
            full_prompt += step_text

    wrong_ans_list = generate_random_ans(answer_list, num_wrong_ans=3)

    return full_prompt, answer_list, wrong_ans_list


def generate_prompt_optimization(location_list):
    idx_list = list(range(len(location_list)))
    permutations = list(itertools.permutations(idx_list))
    optimal_permutation = permutations[0]
    min_distance = -1

    for permutation in permutations:
        location_reordered = [location_list[i] for i in permutation]
        way_points = []
        for i in range(1, len(location_reordered) - 1):
            location = location_reordered[i]
            way_points.append(location)
        start = location_reordered[0]
        end = location_reordered[-1]

        route_json, overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY,
                                                                       mode="driving", way_points=way_points)
        full_distance = 0
        for leg in route_json["legs"]:
            full_distance += leg["distance"]["value"]

        if min_distance == -1:
            min_distance = full_distance
            optimal_permutation = permutation
        else:
            if full_distance < min_distance:
                min_distance = full_distance
                optimal_permutation = permutation
    # turn the idx into actual location strings
    wrong_ans_list = generate_random_ans(optimal_permutation, num_wrong_ans=3)
    for i in range(len(wrong_ans_list)):
        wrong_ans = wrong_ans_list[i]
        wrong_ans_list[i] = [location_list[j] for j in wrong_ans]

    location_optimal_order = [location_list[i] for i in optimal_permutation]

    return "", location_optimal_order, wrong_ans_list


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
        print(random_ans_dict)
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


def generate_prompt_and_choices(location_list, api_key, mode="driving", task="direction"):
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


    origin = location_list[0]
    destination = location_list[-1]
    way_points = []
    for i in range(1, len(location_list) - 1):
        location = location_list[i]
        way_points.append(location)

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
    elif task == "optimization":
        if mode != "driving":
            return RuntimeError("best to do optimization task in driving mode")
        full_prompt, answer_list, wrong_ans_list = generate_prompt_optimization(location_list)
    else:
        return RuntimeError("wrong task type provided for prompt generation")

    correct_idx = random.randint(0, len(wrong_ans_list))
    wrong_ans_list.insert(correct_idx, answer_list)
    final_choices = wrong_ans_list

    return full_prompt, final_choices, correct_idx


def generate_location_combinations(location_dict, num_stops=3, num_comb=5):
    location_names = list(location_dict.keys())
    unique_name_set = set()
    task_list = []
    while len(unique_name_set) < num_comb:
        random_names = tuple(random.sample(location_names, num_stops))
        unique_name_set.add(random_names)

    for name_set in unique_name_set:
        names = list(name_set)
        addrs = []
        for name in names:
            addr = location_dict[name]
            addrs.append(addr)

        task_list.append({"location_names": names,
                         "location_addrs": addrs})

    return task_list


idx2letter = ["A", "B", "C", "D"]
def main():
    data_dir = "../data/long_route_random_single"
    result_dir = "../result/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop_auto"
    map_list = os.listdir(data_dir)

    # map_name = "map1_downtown_bk"
    for map_name in map_list:
        # if map_name == "map1_downtown_bk":
        #     continue
        print("====================================")
        print(f"Processing Map {map_name}")

        map_root_dir = f"{data_dir}/{map_name}"
        out_dir = f"{result_dir}/multi_stop_auto/{map_name}"

        if not os.path.exists(map_root_dir):
            print(f"Directory not found: {map_root_dir}")
            return

        os.makedirs(out_dir, exist_ok=True)

        location_dict = json.load(open(f"{question_dir}/{map_name}.json"))
        task_list = generate_location_combinations(location_dict, num_stops=3, num_comb=5)
        print(f"Generating multi-stop tasks from {len(location_dict.keys())} locations")

        # question_list = []
        for i in range(len(task_list)):
            print(f"Map {map_name} Set {i} ........")
            task = task_list[i]
            set_out_dir = f"{out_dir}/set_{i}"
            os.makedirs(set_out_dir, exist_ok=True)

            question_location_list = task["location_names"]
            GT_location_list = task["location_addrs"]

            formatted_locations = ", ".join(question_location_list)

            full_prompt_driving, final_choices_driving, correct_idx_driving = \
                generate_prompt_and_choices(GT_location_list, GOOGLE_MAPS_API_KEY, mode="driving",
                                            task="direction")
            full_prompt_walking, final_choices_walking, correct_idx_walking = \
                generate_prompt_and_choices(GT_location_list, GOOGLE_MAPS_API_KEY, mode="walking",
                                            task="direction")
            full_prompt_distance_driving, final_choices_distance_driving, correct_idx_distance_driving = \
                generate_prompt_and_choices(GT_location_list, GOOGLE_MAPS_API_KEY, mode="driving",
                                            task="distance")
            full_prompt_distance_walking, final_choices_distance_walking, correct_idx_distance_walking = \
                generate_prompt_and_choices(GT_location_list, GOOGLE_MAPS_API_KEY, mode="walking",
                                            task="distance")
            _, final_choices_optm, correct_idx_optm = \
                generate_prompt_and_choices(GT_location_list, GOOGLE_MAPS_API_KEY, mode="driving",
                                            task="optimization")


            ans_choice_dict = {"direction_driving": {"gt": correct_idx_driving, "choices": final_choices_driving},
                               "direction_walking": {"gt": correct_idx_walking, "choices": final_choices_walking},
                               "distance_driving": {"gt": correct_idx_distance_driving, "choices": final_choices_distance_driving},
                               "distance_walking": {"gt": correct_idx_distance_walking, "choices": final_choices_distance_walking},
                               "optimal_route": {"gt": correct_idx_optm, "choices": final_choices_optm}}

            with open(f"{set_out_dir}/choice_gt.json", 'w') as f:
                json.dump(ans_choice_dict, f, indent=4)

            question_list = [
                f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
                f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
                f"following multiple choice question:\n"
                f"I want to traverse through the locations by foot: {formatted_locations} in this exact order, with the "
                f"starting position being facing {question_location_list[0]}. Since you are walking by foot, you do not "
                f"need to consider constraints such as a street being one-way drive.\n"
                f"Put your single letter choice answer in json format: ```json [letter]```\n"
                f"{full_prompt_walking}\n"
                f"A. {final_choices_walking[0]}      B. {final_choices_walking[1]}\n"
                f"C. {final_choices_walking[2]}      D. {final_choices_walking[3]}\n",

                f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
                f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
                f"following multiple choice question:\n"
                f"I want to traverse through the locations by driving: {formatted_locations} in this exact order, with the "
                f"starting position being facing {question_location_list[0]}. Since you are driving, you must consider "
                f"constraints such as a street being one-way drive.\n"
                f"Put your single letter choice answer in json format: ```json [letter]```\n"
                f"{full_prompt_driving}\n"
                f"A. {final_choices_driving[0]}      B. {final_choices_driving[1]}\n"
                f"C. {final_choices_driving[2]}      D. {final_choices_driving[3]}\n",

                f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
                f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
                f"following multiple choice question:\n"
                f"I want to traverse through the locations by foot: {formatted_locations} in this exact order, with the "
                f"starting position being facing {question_location_list[0]}. Since you are walking by foot, you do not "
                f"need to consider constraints such as a street being one-way drive.\n"
                f"Put your single letter choice answer in json format: ```json [letter]```\n"
                f"{full_prompt_distance_walking}\n"
                f"A. {final_choices_distance_walking[0]}      B. {final_choices_distance_walking[1]}\n"
                f"C. {final_choices_distance_walking[2]}      D. {final_choices_distance_walking[3]}\n",

                f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
                f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
                f"following multiple choice question:\n"
                f"I want to traverse through the locations by driving: {formatted_locations} in this exact order, with the "
                f"starting position being facing {question_location_list[0]}. Since you are driving, you must consider "
                f"constraints such as a street being one-way drive.\n"
                f"Put your single letter choice answer in json format: ```json [letter]```\n"
                f"{full_prompt_distance_driving}\n"
                f"A. {final_choices_distance_driving[0]}      B. {final_choices_distance_driving[1]}\n"
                f"C. {final_choices_distance_driving[2]}      D. {final_choices_distance_driving[3]}\n",

                f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
                f"within an area. Using the spatial information you collect from the images I uploaded, answer the "
                f"following multiple choice question:\n"
                f"Can you choose the optimal (time-efficient) order of visiting all these locations: {formatted_locations} "
                f"within this area by driving? Since you are driving, you must consider constraints such as one-way drive "
                f"streets.\n"
                f"Put your single letter choice answer in json format: ```json [letter]```\n"
                f"A. {final_choices_optm[0]}      B. {final_choices_optm[1]}\n"
                f"C. {final_choices_optm[2]}      D. {final_choices_optm[3]}\n"
            ]

            ''' ================== Inference ================== '''
            ''' upload images to model and ask questions '''
            results, _ = analyze_street_view(map_root_dir, question_list, out_dir=map_root_dir)

            ''' ================== save questions and answers in md and json ================== '''
            ''' the json could be used later for auto eval? '''
            if results:
                print("\nAnalysis Summary:")
                if os.path.exists(f"{set_out_dir}/Q&A.md"):
                    os.remove(f"{set_out_dir}/Q&A.md")
                idx = 0
                for question, answer in results.items():
                    print(f"\n{question}")
                    print(f"{answer}")
                    with open(f"{set_out_dir}/Q&A.md", "a", encoding="utf-8") as file:
                        file.write(f"{question}")
                        file.write(f"{answer}")
                        file.write(f"\n----------\nTrue Correct Answer: "
                                   f"{idx2letter[[correct_idx_walking, correct_idx_driving, correct_idx_distance_walking, correct_idx_distance_driving, correct_idx_optm][idx]]}\n==========\n")

                    results[question] += f"\n----------\nTrue Correct Answer: " \
                                         f"{idx2letter[[correct_idx_walking, correct_idx_driving, correct_idx_distance_walking, correct_idx_distance_driving, correct_idx_optm][idx]]}\n==========\n"
                    idx += 1

                with open(f"{set_out_dir}/Q&A.json", 'w') as f:
                    json.dump(results, f, indent=4)

            # if not os.path.exists(f"{set_out_dir}/Q&A.json"):
            #     return RuntimeError(f"something wrong: Q&A save failed")

            ''' ================== Save Walking Ground Truth Route Data ================== '''
            ''' the optimal order is manually regulated, and the route is generated by Google Maps '''
            route_data_walk = {
                "map": map_name,
                "ans": idx2letter[correct_idx_walking],
                "frames": [],
            }
            all_positions_and_headings, _, _ = get_routes_from_locations(GT_location_list,
                                                                         route_file_output=f"{set_out_dir}/route_map_multi_stop_walikng.html",
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
            route_data_file = f"{set_out_dir}/route_data_multi_stop_walking.json"
            with open(route_data_file, 'w') as f:
                json.dump(route_data_walk, f, indent=2)

            ''' ================== Save Driving Ground Truth Route Data ================== '''
            ''' the optimal order is manually regulated, and the route is generated by Google Maps '''
            route_data_drive = {
                "map": map_name,
                "ans": idx2letter[correct_idx_driving],
                "frames": [],
            }
            all_positions_and_headings, _, _ = get_routes_from_locations(GT_location_list,
                                                                         route_file_output=f"{set_out_dir}/route_map_multi_stop_driving.html",
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
            route_data_file = f"{set_out_dir}/route_data_multi_stop_driving.json"
            with open(route_data_file, 'w') as f:
                json.dump(route_data_drive, f, indent=2)

            ''' ================== Save Optimal Route Ground Truth Route Data ================== '''
            route_data_optimal = {
                "map": map_name,
                "ans": idx2letter[correct_idx_optm],
                "frames": [],
            }
            all_positions_and_headings, _, _ = get_routes_from_locations(final_choices_optm[correct_idx_optm], f"{set_out_dir}/route_map_multi_stop_optimal.html")
            for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                           total=len(all_positions_and_headings)):
                # Store frame data with the movement from all_positions_and_headings
                route_data_optimal["frames"].append({
                    "coordinates": {"lat": lat, "lng": lng},
                    "heading": heading,
                    "movement": movement  # This now correctly uses the movement we determined earlier
                })

            # Save optimal route data to JSON
            route_data_file = f"{set_out_dir}/route_data_multi_stop_optimal.json"
            with open(route_data_file, 'w') as f:
                json.dump(route_data_optimal, f, indent=2)


if __name__ == "__main__":
    main()
