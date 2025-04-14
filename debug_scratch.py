import ast
import math
import re

import numpy as np

import transformers

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def sigmoid(x, a=1, b=0):
    return 1 / (1 + math.exp(a*(-x-b)))

def cal_reward_list(output_ans, gt_ans, epsilon=0.1):
    output_ans_list = ast.literal_eval(output_ans)
    gt_list = ast.literal_eval(gt_ans)
    if len(output_ans_list) != len(gt_list):
        reward = 0.0
        print("===============")
        print(f"false length, reward: {reward:.2f}")
    else:
        squared_diffs = [(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)]
        rmse = math.sqrt(sum(squared_diffs) / len(squared_diffs))
        mse = sum(squared_diffs) / len(squared_diffs)

        # # Calculate relative squared differences with epsilon to avoid division by zero
        # relative_squared_diffs = [((p - gt) / max(epsilon, abs(gt))) ** 2 for p, gt in zip(output_ans_list, gt_list)]
        #
        # # Calculate Relative MSE and RMSE
        # mse = sum(relative_squared_diffs) / len(relative_squared_diffs)
        # rmse = math.sqrt(mse)

        reward_rmse = 2 * (1 - sigmoid(rmse, a=0.2, b=0))
        reward_mse = 2*(1 - sigmoid(mse, a=0.2, b=0))

        # mse = np.mean([(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)])
        # rmse = math.sqrt(mse)
        # reward_mse = 1 / (1 + mse)
        # reward_rmse = 1 / (1 + rmse)

        # mse = np.mean([(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)])
        # rmse = math.sqrt(mse)
        # reward_rmse = math.exp(-rmse)
        # reward_mse = math.exp(-mse)
        print("===============")
        print(f"MSE: {mse:.2f}, reward: {reward_mse:.2f}")
        print(f"RMSE: {rmse:.2f}, reward: {reward_rmse:.2f}")
    # return reward_rmse


# output = "[1, 2, 1, 3, 4, 5.6]"
# gt_list0 = "[1, 2, 1, 3, 4, 5.6]"
# gt_list1 = "[1, 2, 1, 4, 4, 7.6]"
# gt_list2 = "[1, 2, 3, 4, 5, 7.6]"
# gt_list3 = "[1, 0.02, 3, 5, 76]"
# gt_list4 = "[1, 0.01, 1, 2, 3, 5.6]"
# gt_list5 = "[10, 10, 10, 10, 10, 10.6]"
# gt_list6 = "[100, 100, 100, 100, 100, 100.6]"
# gt_list7 = "[1, 0, 1, 2, 3, 56]"
# cal_reward_list(output, gt_list0)
# cal_reward_list(output, gt_list1)
# cal_reward_list(output, gt_list2)
# cal_reward_list(output, gt_list3)
# cal_reward_list(output, gt_list4)
# cal_reward_list(output, gt_list5)
# cal_reward_list(output, gt_list6)
# cal_reward_list(output, gt_list7)


def format_reward(completions):

    pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    all_rewards = []
    for content in completion_contents:
        match = re.fullmatch(pattern, content, re.DOTALL)
        if match:
            # if only the tag format is matched, rerward 0.4
            reward = 0.4
            try:
                output_ans = extract_answer(content)
                output_ans_dict = ast.literal_eval(output_ans)
                # if dict format is matched, rerward 0.7
                if isinstance(output_ans_dict, dict):
                    reward = 0.7
                    all_correct_flag = True
                    # check if each val in dict is list
                    for key in output_ans_dict.keys():
                        if not isinstance(output_ans_dict[key], list):
                            all_correct_flag = False
                            break
                    # give full reward 1 only if all elements in the dict are list, otherwise only reward 0.7
                    if all_correct_flag:
                        reward = 1
            except Exception as e:
                reward = 0.4
        else:
            reward = 0

        all_rewards.append(reward)
    return all_rewards




# completions = [
#     [{"content": "<answer>[1, 2, 3]</answer>"}],      # Valid list
#     [{"content": "<answer>42</answer>"}],              # Not a list
#     [{"content": "<answer>[]</answer>"}],               # Empty list
#     [{"content": "<answer>['a', 'b', 'c']</answer>"}], # Valid list of strings
#     [{"content": "No answer tag"}]                     # No answer tag
# ]

# completions = [
#     [{"content": "<answer>{'x': [1.2, 2.4, 3.6], 'y': [-0.5, 0.0, 0.5], 'z': [0.0, 1.0, 2.0], 'roll': [5.0, 10.0, 15.0], 'pitch': [0.0, -5.0, -10.0], 'yaw': [90.0, 180.0, 270.0]}</answer>"}],      # Valid list
#     [{"content": "<answer>{x: [3.3, 3.2, 3.1], y: [0.0, -0.1, -0.2], z: [1.0, 0.9, 0.8], roll: [180.0, 170.0, 160.0], 'pitch': [0.0, 10.0, 20.0], 'yaw': [270.0, 280.0, 290.0]}</answer>"}],              # Not a list
#     [{"content": "<answer>{}</answer>"}],               # Empty list
#     [{"content": "<answer>{'x': [0.0, 1.0, 2.0, 3.0], 'y': [1.0, 1.5, 2.0, 2.5], 'z': [0.0, -0.1, -0.2, -0.3], 'roll': [0.0, 45.0, 90.0, 135.0], 'pitch': [10.0, 5.0, 0.0, -5.0], 'yaw': [0.0, 90.0, 180.0, 270.0]}</answer>"}], # Valid list of strings
#     [{"content": "No answer tag"}] ,                    # No answer tag
#     [{"content": "<answer>{'x': 0.0, 1.0, 2.0, 3.0, 'y': 1.0, 1.5, 2.0, 2.5, 'z': 0.0, -0.1, -0.2, -0.3, roll: [0.0, 45.0, 90.0, 135.0], 'pitch': [10.0, 5.0, 0.0, -5.0], 'yaw': [0.0, 90.0, 180.0, 270.0]}</answer>"}],
#     # Valid list of strings
# ]


def sigmoid(x, a=1, b=0):
    return 1 / (1 + math.exp(a * (-x + b)))
# print(format_reward(completions))
for val in [0.1, 0.2, 0.5, 1, 2, 2.5, 3, 4]:
    print(sigmoid(val, a=2, b=2/4))