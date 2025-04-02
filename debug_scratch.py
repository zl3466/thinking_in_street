import ast
import math
import re

import numpy as np

import transformers

print("success")

from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
print("success2")

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


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    all_rewards = []
    for content in completion_contents:
        match = re.fullmatch(pattern, content, re.DOTALL)
        if match:
            reward = 0.5
            list_pattern = r"<answer>\[.*?\]</answer>"
            list_match = re.fullmatch(list_pattern, content, re.DOTALL)
            if list_match:
                reward = 1.0
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
# print(format_reward(completions))
