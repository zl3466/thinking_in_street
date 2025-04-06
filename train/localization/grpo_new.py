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
                    ''' reward = 2 * sigmoid(rmse) -> range(0, 1). Use a = 0.2 to get steep sigmoid '''
                    squared_diffs = [(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)]
                    rmse = math.sqrt(sum(squared_diffs) / len(squared_diffs))
                    reward = 2 * (1 - sigmoid(rmse, a=0.2, b=0))
                    ''' we ensure a minimum reward of 0.1 if the number of elements in the list is correct '''
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



def quaternion_to_yaw(q):
    """
    Convert quaternion (w, x, y, z) to yaw angle.
    """
    w, x, y, z = q
    # negate to make right turn positive and left turn negative (clockwise)
    yaw = -math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return yaw


def calculate_displacement(t1, t2):
    """
    Calculate Euclidean distance between two translation vectors.
    """
    return math.sqrt((t2[0] - t1[0]) ** 2 + (t2[1] - t1[1]) ** 2)


def calculate_delta_heading(yaw1, yaw2):
    """
    Calculate the delta heading in degrees clockwise, range [-180, 180].
    Turning right is positive, turning left is negative.
    """
    delta = math.degrees(yaw2 - yaw1)
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    return delta

def nusc_to_examples(nusc_dataset, video_out_dir="", sample_rate=10, batch_size=4, visualize=False):
    if visualize:
        ''' prepare video writer '''
        sample_img = cv2.imread(nusc_dataset.img_filepaths[0])
        w = sample_img.shape[1]
        h = sample_img.shape[0]
        video_width = w
        video_height = h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out0 = cv2.VideoWriter(video_out_dir, fourcc, sample_rate, (video_width, video_height))
        frame_text = ""
        ''' video writer code ends here '''


    # raw data is 10hz
    step_size = 10 // sample_rate

    meta_dict = nusc_dataset.meta_dict
    cam_list = nusc_dataset.camera_list
    cam = cam_list[0]

    meta_data = meta_dict[cam]
    ego_pose_list = meta_data["ego_pose_original"]

    prev_yaw = None
    prev_translation = None

    disp_gt_all = []
    delta_heading_gt_all = []
    img_list_all = []

    batch_disp_gt = []
    batch_delta_heading_gt = []
    batch_img_list = []
    sample_count = 0
    
    for frame_i in range(len(ego_pose_list)):
        if frame_i % step_size != 0:
            continue
        ego_pose = ego_pose_list[frame_i]

        # Note that NuScenes's Ego Pose rotation is in (w, x, y, z) quat format
        # translation is (x, y, z=0) in meters
        translation = ego_pose["translation"]
        rotation = ego_pose["rotation"]
        yaw = quaternion_to_yaw(rotation)

        batch_img_list.append(nusc_dataset.rel_img_filepaths[frame_i])
        if prev_yaw is not None and prev_translation is not None:
            disp = calculate_displacement(prev_translation, translation)
            delta_heading = calculate_delta_heading(prev_yaw, yaw)
            batch_disp_gt.append(disp)
            batch_delta_heading_gt.append(delta_heading)
            frame_text = f"delta heading: {delta_heading}, displacement: {disp}"
            sample_count += 1

        prev_yaw = yaw
        prev_translation = translation

        if sample_count != 0 and sample_count % (batch_size - 1) == 0:
            if len(batch_disp_gt) != batch_size - 1 or len(batch_delta_heading_gt) != batch_size - 1 or len(batch_img_list) != batch_size:
                # if this is residue at the end of a scene, don't use it

                if frame_i == len(ego_pose_list) - 1:
                    break
                else:
                    assert len(
                        batch_img_list) == batch_size, f"image path batch size mismatch {len(batch_img_list)}, {batch_size}"
                    assert len(
                        batch_disp_gt) == batch_size - 1, f"displacement batch size mismatch {len(batch_disp_gt)}, {batch_size - 1}"
                    assert len(
                        batch_delta_heading_gt) == batch_size - 1, f"delta_heading batch size mismatch {len(batch_delta_heading_gt)}, {batch_size - 1}"

            disp_gt_all.append(batch_disp_gt)
            delta_heading_gt_all.append(batch_delta_heading_gt)
            img_list_all.append(batch_img_list)
            batch_disp_gt = []
            batch_delta_heading_gt = []
            batch_img_list = []
            prev_yaw = None
            prev_translation = None
            sample_count = 0

        if visualize:
            img = cv2.imread(f"{nusc_dataset.rel_img_filepaths[frame_i]}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, frame_text, (10, 30), font, fontScale=1, color=(0, 0, 255), thickness=2,
                        lineType=cv2.LINE_AA)
            out0.write(img)


    example_list = []
    problem_id_offset = 0
    for batch_i in range(len(disp_gt_all)):
        batch_disp_gt = disp_gt_all[batch_i]
        batch_delta_heading_gt = delta_heading_gt_all[batch_i]
        batch_img_list = img_list_all[batch_i]

        example_heading = {
            "problem_id": batch_i + problem_id_offset,
            "problem": f"I uploaded {len(batch_img_list)} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's change in heading direction between each frame and its previous frame.\n"
                       f"Give your answer in degree values ranging between -180 and 180 degrees, with veer to the right "
                       f"being positive degrees and to the left being negative degrees.\n"
                       f"Keep all degree values in one list. "
                       f"You should have {len(batch_disp_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "options": [],
            "solution": f"<answer>{batch_delta_heading_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }
        problem_id_offset += 1

        example_disp = {
            "problem_id": batch_i + problem_id_offset,
            "problem": f"I uploaded {len(batch_img_list)} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's displacement between each frame and its previous frame.\n"
                       f"Give your answer in numerical values in meter unit.\n"
                       f"Keep all displacement values in one list. "
                       f"You should have {len(batch_disp_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "options": [],
            "solution": f"<answer>{batch_disp_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }

        example_list.append(example_heading)
        example_list.append(example_disp)

    return example_list
    

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
                            example['data_type']: [f"{os.getenv("DATASET_DIR")}/{file_path}" for file_path in example['path']]
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
    example_json_path = f"{root_path}/examples/qwen_vllm"

    train_out_path = f"{out_path}/train"
    test_out_path = f"{out_path}/test"
    
    dataset =  DatasetDict({
        "train": Dataset.from_json(f"{example_json_path}/train/scene_{train_scene_idx_list[0]}-{train_scene_idx_list[-1]}_cam_{num_cam}/train_examples.json"), 
        "test": Dataset.from_json(f"{example_json_path}/test/scene_{train_scene_idx_list[0]}-{train_scene_idx_list[-1]}_cam_{num_cam}/test_examples.json")
    })
    dataset = dataset.map(prepare_dataset_nusc)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

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
