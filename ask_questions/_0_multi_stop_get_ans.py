import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google import genai
from bs4 import BeautifulSoup
import ast

from utils import *

client = genai.Client(api_key="GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

def extract_model_answers(text):
    pattern = r"\[\[\[(.*?)\]\]\]"  # Non-greedy match inside [[[...]]]
    return re.findall(pattern, text)

def extract_gt_answers(text):
    pattern = r"^-+\s*True Correct Answer:\s*([A-Z])\s*=+"  # Captures a single uppercase letter
    return re.findall(pattern, text, re.MULTILINE)


def extract_choices(text):
    """Extract multiple-choice options (A, B, C, D) from a given string."""
    pattern = r"A\. (\[.*?\])\s*B\. (\[.*?\])\s*C\. (\[.*?\])\s*D\. (\[.*?\])"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        choices = {label: ast.literal_eval(match.group(i + 1)) for i, label in enumerate("ABCD")}
        return choices
    return None


def convert_to_feet(value):
    """Convert distance strings to feet."""
    num, unit = value.split()
    num = float(num)
    if unit == "mi":  # Convert miles to feet
        return num * 5280
    return num  # Already in feet

def mean_squared_error(list1, list2):
    # print(list1, list2)
    """Compute MSE between two lists of distance values."""
    feet1 = [convert_to_feet(x) for x in list1]
    feet2 = [convert_to_feet(x) for x in list2]

    squared_diffs = [(a - b) ** 2 for a, b in zip(feet1, feet2)]
    mse = sum(squared_diffs) / len(squared_diffs)

    avg1 = sum(feet1) / len(feet1)
    avg2 = sum(feet2) / len(feet2)
    return mse, avg1, avg2

# Example Usage
result_dir = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/result/long_route_random_single_V1"
map_list = os.listdir(result_dir)

final_results = {}
for map_name in map_list:
    set_list = os.listdir(f"{result_dir}/{map_name}/multi_stop_auto")
    final_results[map_name] = {}
    for set_name in set_list:
        print(f"\nmap: {map_name}, set: {set_name}")
        final_results[map_name][set_name] = {"model_answers": [], "gt_answers": [], "model_choice": [], "gt_choice": [], "mse": []}
        qna_json_file = f"{result_dir}/{map_name}/multi_stop_auto/{set_name}/Q&A.json"
        if not os.path.exists(qna_json_file):
            print("Analysis failed for this set. Continuing...")
            continue
        qna_dict = json.load(open(qna_json_file))
        # with open(qna_md_file, "r", encoding="utf-8") as f:
        #     md_text = f.read()

        q_list = list(qna_dict.keys())
        for q in q_list:
            a = qna_dict[q]

            model_answers = parse_json_from_response(a)[0]
            gt_answers = extract_gt_answers(a)
            choices = extract_choices(q)
            # if len(model_answers) != len(true_answers):
            #     print(f"Warning: map {map_name} set {set_name} has ill-formatted answers")
            print(f"model answer: {model_answers}, true answer: {gt_answers}")
            mse = -1
            model_avg = -1
            gt_avg = -1
            final_results[map_name][set_name]["model_answers"].append(model_answers)
            final_results[map_name][set_name]["gt_answers"].append(gt_answers)

            model_ans_flag = False
            gt_ans_flag = False
            if len(model_answers) > 0 and len(model_answers[0]) == 1:
                final_results[map_name][set_name]["model_choice"].append(choices[model_answers[0]])
                model_ans_flag = True
            else:
                final_results[map_name][set_name]["model_choice"].append([])
            if len(gt_answers) > 0 and len(gt_answers[0]) == 1:
                final_results[map_name][set_name]["gt_choice"].append(choices[gt_answers[0]])
                gt_ans_flag = True
            else:
                final_results[map_name][set_name]["gt_choice"].append([])

            if model_ans_flag and gt_ans_flag:
                if all(" ft" in item or " mi" in item for item in choices[gt_answers[0]]):

                    mse, model_avg, gt_avg = mean_squared_error(choices[model_answers[0]], choices[gt_answers[0]])
                    final_results[map_name][set_name]["mse"].append([mse, model_avg, gt_avg])
            print(f"mse | model_avg | gt_avg: {[mse, model_avg, gt_avg]}")

with open(f"../final_result.json", 'w') as f:
    json.dump(final_results, f, indent=4)
