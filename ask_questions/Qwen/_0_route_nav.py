import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.gemini_utils import *

from bs4 import BeautifulSoup
import random
import itertools


def generate_prompt_direction(route_json, use_distance=True, use_time=False):
    if use_distance and use_time:
        RuntimeError("You may only use one of distance or time as hint for the model")
    route = route_json["routes"][0]

    full_prompt = f"Choose from the four options below to fill in the blanks [?] in the following prompt with " \
                  f"direction keywords 'left' or 'right' to make the prompt describe a correct route:\n"
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
    if len(answer_list) < 2:
        return full_prompt, None, None
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


def generate_prompt_optimization(location_list, q_loc_list):
    '''
    location_list: list of detailed location addresses for Google Maps to query
    q_loc_list: list of location names for LLM input
    '''
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
        # get optimal order by shortest travel distance
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
        wrong_ans_list[i] = [q_loc_list[j] for j in wrong_ans]

    location_optimal_order = [q_loc_list[i] for i in optimal_permutation]

    return "", location_optimal_order, wrong_ans_list


def generate_random_ans(ans_list, num_wrong_ans=3):
    seen_answers = set()  # To track seen wrong answers and prevent duplicates
    seen_answers.add(tuple(ans_list))
    wrong_ans_list = []
    #   optimal route prompt
    if isinstance(ans_list[0], int):
        # If the answers are integers indices
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
            # print(seen_answers, random_ans)
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
        # print(random_ans_dict)
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

    if min_val == max_val:
        min_val = min_val * 0.2
        max_val = min_val * 2

    value_range = max_val - min_val
    # print(f"range: {min_val - value_range}, {max_val + value_range}")
    random_value = random.uniform(min_val - value_range, max_val + value_range)

    # Ensure the value is greater than 0
    while random_value <= 0:
        random_value = random.uniform(min_val - value_range, max_val + value_range)
    random_value = round(random_value, 1)
    return random_value


def generate_prompt_and_choices(location_list, q_loc_list, api_key, mode="driving", task="direction"):
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
        # if mode != "driving":
        #     return RuntimeError("best to do optimization task in driving mode")
        full_prompt, answer_list, wrong_ans_list = generate_prompt_optimization(location_list, q_loc_list)
    else:
        return RuntimeError("wrong task type provided for prompt generation")

    if answer_list is None or wrong_ans_list is None:
        return full_prompt, answer_list, wrong_ans_list
    correct_idx = random.randint(0, len(wrong_ans_list))
    final_choices = wrong_ans_list
    final_choices.insert(correct_idx, answer_list)

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
    data_dir = "../../dataset/google_streetview"
    map_dir = f"../../dataset/maps"
    out_dir = f"../../dataset/examples/eval"
    map_list = os.listdir(data_dir)

    # map_name = "map1_downtown_bk"
    example_list = []
    for map_name in map_list:
        # if map_name == "map1_downtown_bk":
        #     continue
        print("====================================")
        print(f"Processing Map {map_name}")

        map_root_dir = f"{data_dir}/{map_name}"

        if not os.path.exists(map_root_dir):
            print(f"Directory not found: {map_root_dir}")
            return

        os.makedirs(out_dir, exist_ok=True)

        location_dict = json.load(open(f"{map_dir}/{map_name}.json"))
        task_list = generate_location_combinations(location_dict, num_stops=2, num_comb=5)
        task_list_multi_loc = generate_location_combinations(location_dict, num_stops=3, num_comb=5)
        print(f"Generating multi-stop tasks from {len(location_dict.keys())} locations")

        # question_list = []
        for i in range(len(task_list)):
            print(f"Map {map_name} Set {i} ........")
            task = task_list[i]
            task_multi = task_list_multi_loc[i]

            question_location_list = task["location_names"]
            GT_location_list = task["location_addrs"]

            q_loc_list_multi = task_multi["location_names"]
            GT_loc_list_multi = task_multi["location_addrs"]

            formatted_locations = ", ".join(question_location_list)
            formatted_locations_multi = ", ".join(q_loc_list_multi)

            full_prompt_walking, final_choices_walking, correct_idx_walking = \
                generate_prompt_and_choices(GT_location_list, question_location_list, GOOGLE_MAPS_API_KEY,
                                            mode="walking",
                                            task="direction")
            full_prompt_distance_walking, final_choices_distance_walking, correct_idx_distance_walking = \
                generate_prompt_and_choices(GT_location_list, question_location_list, GOOGLE_MAPS_API_KEY,
                                            mode="walking",
                                            task="distance")
            _, final_choices_optm, correct_idx_optm = \
                generate_prompt_and_choices(GT_loc_list_multi, q_loc_list_multi, GOOGLE_MAPS_API_KEY, mode="walking",
                                            task="optimization")

            ''' save questions into examples '''
            video_path = f"{data_dir}/{map_name}/{map_name}.mp4"
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            # video_img_paths = os.listdir(f"{data_dir}/{map_name}/frames")
            # video_img_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            # video_img_paths = [f"{map_name}/frames/{video_img_paths[i]}" for i in range(len(video_img_paths))]

            problem_id_offset = 0
            example_rel_direction = {
                "problem_id": f"{map_name}_{i}_{problem_id_offset}",
                "dataset_name": "google_streetview",
                "problem": f"These are frames of a video. They cover all streets within an area. \n"
                           f"If I am standing in front of {question_location_list[0]} and facing it, "
                           f"is {question_location_list[1]} located to my front, back, left, or right direction?\n"
                           f"Put your single letter choice answer between the <answer> </answer> tags tags\n",
                "data_type": "video",
                "problem_type": "list",
                "problem_subject": "rel_direction",
                "options": [f"A. front",
                            f"B. back",
                            f"C. left",
                            f"D. right"],
                "solution": f"<answer></answer>",
                "path": [video_path],
                "reward": 1.0,
                "select": True
            }
            problem_id_offset += 1
            example_list.append(example_rel_direction)

            if final_choices_walking is not None and correct_idx_walking is not None:
                example_direction = {
                    "problem_id": f"{map_name}_{i}_{problem_id_offset}",
                    "dataset_name": "google_streetview",
                    "problem": f"These are frames of a video. They cover all streets within an area. \n"
                               f"I want to go from {question_location_list[0]} to {question_location_list[1]}, "
                               f"with the starting position being facing {question_location_list[0]}. \n"
                               f"Put your single letter choice answer between the <answer> </answer> tags tags\n"
                               f"{full_prompt_walking}\n",
                    "data_type": "video",
                    "problem_type": "list",
                    "problem_subject": "route_nav_direction",
                    "options": [f"A. {final_choices_walking[0]}",
                                f"B. {final_choices_walking[1]}",
                                f"C. {final_choices_walking[2]}",
                                f"D. {final_choices_walking[3]}"],
                    "solution": f"<answer>{idx2letter[correct_idx_walking]}</answer>",
                    "path": [video_path],
                    "reward": 1.0,
                    "select": True
                }
                problem_id_offset += 1
                example_list.append(example_direction)

            example_distance = {
                "problem_id": f"{map_name}_{i}_{problem_id_offset}",
                "dataset_name": "google_streetview",
                "problem": f"These are frames of a video. They cover all streets within an area. \n"
                           f"I want to go from {question_location_list[0]} to {question_location_list[1]}, "
                           f"with the starting position being facing {question_location_list[0]}. \n"
                           f"Put your single letter choice answer between the <answer> </answer> tags tags\n"
                           f"{full_prompt_distance_walking}\n",
                "data_type": "video",
                "problem_type": "list",
                "problem_subject": "route_nav_distance",
                "options": [f"A. {final_choices_distance_walking[0]}",
                            f"B. {final_choices_distance_walking[1]}",
                            f"C. {final_choices_distance_walking[2]}",
                            f"D. {final_choices_distance_walking[3]}"],
                "solution": f"<answer>{idx2letter[correct_idx_distance_walking]}</answer>",
                "path": [video_path],
                "reward": 1.0,
                "select": True
            }
            problem_id_offset += 1
            example_list.append(example_distance)

            example_optim = {
                "problem_id": f"{map_name}_{i}_{problem_id_offset}",
                "dataset_name": "google_streetview",
                "problem": f"These are frames of a video. They cover all streets within an area. \n"
                           f"I want to visit these {len(q_loc_list_multi)} locations: {formatted_locations_multi}.\n"
                           f"Can you choose the order of visiting these locations that gives the shortest route?\n"
                           f"Put your single letter choice answer between the <answer> </answer> tags tags\n",
                "data_type": "video",
                "problem_type": "list",
                "problem_subject": "route_nav_optim",
                "options": [f"A. {final_choices_optm[0]}",
                            f"B. {final_choices_optm[1]}",
                            f"C. {final_choices_optm[2]}",
                            f"D. {final_choices_optm[3]}"],
                "solution": f"<answer>{idx2letter[correct_idx_optm]}</answer>",
                "path": [video_path],
                "reward": 1.0,
                "select": True
            }
            example_list.append(example_optim)

    with open(f"{out_dir}/route_nav.json", 'w') as f:
        json.dump(example_list, f, indent=4)


if __name__ == "__main__":
    main()
