# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import math
import os
import sys
import json

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from datasets import Dataset, DatasetDict

from rouge_score import rouge_scorer
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from train.data_loader.nuscenes import NuScenesDataset
import cv2
import random
import time

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "list": " Please provide the list answer (e.g., [val, val, val, ...]) within the <answer> </answer> tags."
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None


def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[m][n] / max(1, m)


def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure


def sigmoid(x, a=1, b=0):
    return 1 / (1 + math.exp(a * (-x + b)))


def accuracy_reward(completions, solution, **kwargs):
    question_type = kwargs['problem_type'][0]

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):

        try:
            # print("content: ", content)
            output_ans = extract_answer(content)
            # print("output_ans: ", output_ans)
            gt_ans = extract_answer(sol)
            # print("gt_ans: ", gt_ans)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            elif question_type == "list":
                # print("output_ans: ", output_ans)
                output_ans_list = ast.literal_eval(output_ans)
                gt_list = ast.literal_eval(gt_ans)
                if len(output_ans_list) != len(gt_list):
                    reward = 0.0
                else:
                    if isinstance(gt_list[0], (int, float)):
                        # print(f"========================== Val output_ans_list: {output_ans_list} ")
                        ''' reward = 2 * sigmoid(rmse) -> range(0, 1). Use a = 0.2 to get steep sigmoid '''
                        squared_diffs = [(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)]
                        rmse = math.sqrt(sum(squared_diffs) / len(squared_diffs))
                        reward = 2 * (1 - sigmoid(rmse, a=0.2, b=0))
                        ''' we ensure a minimum reward of 0.1 if the number of elements in the list is correct '''
                        reward = max(0.1, reward)
                    else:
                        # print(f"========================== String output_ans_list: {output_ans_list} ")
                        ''' the general_dir case where the list contains string keywords like forward, left, slight right, etc. '''
                        reward = sum([a.lower() == b.lower() for a, b in zip(output_ans_list, gt_list)]) / len(gt_list)
                        reward = max(0.1, reward)
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            # print(f"logging to {log_path}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    # print(f"\n========\nbatch rewards: {rewards}\n========\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def format_reward_list(completions, **kwargs):
    if kwargs['problem_type'][0] != "list":
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    else:
        """
        Reward function for list answers.
        First we chech if answer has the <answer></anwswer> tag, reward=0.5
        Then we check if the answer tag contains a list
        """
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        all_rewards = []
        for content in completion_contents:
            match = re.fullmatch(pattern, content, re.DOTALL)
            if match:
                reward = 0.5
                try:
                    output_ans = extract_answer(content)
                    output_ans_list = ast.literal_eval(output_ans)
                    if isinstance(output_ans_list, list):
                        reward = 1.0
                except Exception as e:
                    reward = 0.5
            else:
                reward = 0

            all_rewards.append(reward)
        return all_rewards



def prepare_dataset_nusc(example):
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']

    if example["problem_type"] == 'list':
        msg = {
            "prompt":
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            example['data_type']: [f"{os.getenv('DATASET_DIR')}/{file_path}" for file_path in
                                                   example['path']]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                    ]
                }]
        }
    else:
        msg = {
            "prompt":
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                    ]
                }]
        }
    return msg


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
    #     dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    # else:
    #     # Load the dataset
    #     dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Load dataset
    num_cam = 1
    train_num_scene = int(os.getenv("NUM_TRAIN_SCENE"))
    test_num_scene = min(150, train_num_scene // 4)
    # sample_rate = 2
    train_scene_idx_list = []
    test_scene_idx_list = []
    for i in range(train_num_scene):
        train_scene_idx_list.append(i)
    for i in range(test_num_scene):
        test_scene_idx_list.append(i)

    root_path = script_args.dataset_name
    example_json_path = f"{root_path}/examples/qwen_vllm_3q"

    avail_train_example = os.listdir(f"{example_json_path}/train")
    avail_test_example = os.listdir(f"{example_json_path}/test")

    train_example_dict_list = []
    end_scene_idx=0
    for folder_name in avail_train_example:
        end_scene_idx = int(folder_name.split("_")[1].split("-")[-1])
        train_example_dict = json.load(open(f"{example_json_path}/train/{folder_name}/train_examples.json"))
        train_example_dict_list.append(train_example_dict)
        if train_num_scene <= end_scene_idx+1:
            break
    if end_scene_idx+1 < train_num_scene:
        train_num_scene = end_scene_idx+1
        print(f"Not enough scene to train, using {end_scene_idx+1} scenes that is available")

    test_example_dict_list = []
    end_scene_idx = 0
    for folder_name in avail_test_example:
        end_scene_idx = int(folder_name.split("_")[1].split("-")[-1])
        test_example_dict = json.load(open(f"{example_json_path}/train/{folder_name}/train_examples.json"))
        test_example_dict_list.append(test_example_dict)
        if test_num_scene <= end_scene_idx + 1:
            break
    if end_scene_idx + 1 < test_num_scene:
        test_num_scene = end_scene_idx+1
        print(f"Not enough scene to test, using {end_scene_idx + 1} scenes that is available")


    train_example_list = []
    test_example_list = []
    for example_dict in train_example_dict_list:
        for scene in example_dict.keys():
            examples = example_dict[scene]
            train_example_list += examples
    for example_dict in test_example_dict_list:
        for scene in example_dict.keys():
            examples = example_dict[scene]
            test_example_list += examples

    dataset = DatasetDict({
        "train": Dataset.from_list(train_example_list),
        "test": Dataset.from_list(test_example_list)
    })
    dataset = dataset.map(prepare_dataset_nusc)
    

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)
    print(f"train {train_num_scene} scenes, test {test_num_scene} scenes")
    # Initialize the GRPO trainer
    print(model_args.model_name_or_path)
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
