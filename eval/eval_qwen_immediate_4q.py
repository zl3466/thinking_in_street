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
import argparse
import ast
import math
import os
import sys
import json

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from datasets import Dataset, DatasetDict

from rouge_score import rouge_scorer
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from train.data_loader.nuscenes import NuScenesDataset
from train.data_loader.scannet import ScanNetDataset
import cv2
import random
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.qwen_utils import *

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
    "list": " Please provide the list answer (e.g., [val, val, val, ...]) within the <answer> </answer> tags.",
    "dict": " Please provide the dictionary answer (e.g. {key: val, key: val, ...}) within the <answer> </answer> tags."
}
general_direction_options = ["stationary", "forward", "backward", "left", "slight left", "back left", "slight back left", "right", "slight right", "back right", "slight back right"]


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


def calc_accuracy_score(output_text, gt, question_type):
    try:
        # print("content: ", content)
        output_ans = extract_answer(output_text)
        # print("output_ans: ", output_ans)
        gt_ans = extract_answer(gt)
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
        elif question_type == "dict":
            output_ans_dict = ast.literal_eval(output_ans)
            gt_dict = ast.literal_eval(gt_ans)
            # reward_unit = 1 / len(gt_dict.keys())
            reward = 0
            for key in output_ans_dict.keys():
                # the keys x, y, z, roll, pitch, yaw must all match
                if key not in gt_dict.keys():
                    sub_reward = 0
                    break
                else:
                    output_ans_list = output_ans_dict[key]
                    gt_list = gt_dict[key]
                    if len(output_ans_list) != len(gt_list):
                        sub_reward = 0.0
                    else:
                        squared_diffs = [(p - gt) ** 2 for p, gt in zip(output_ans_list, gt_list)]
                        rmse = math.sqrt(sum(squared_diffs) / len(squared_diffs))
                        sub_reward = 2 * (1 - sigmoid(rmse, a=0.2, b=0))
                        ''' we ensure a minimum reward of 0.1 if the number of elements in the list is correct '''
                        # reward range for each key is 0.1 to 1
                        sub_reward = max(0.1, sub_reward)
                reward += sub_reward
            # normalize total reward for the dict to between 0 - 1
            reward = reward / len(gt_dict.keys())
        else:
            reward = 0.0
    except Exception as e:
        print(f"Error in calc_accuracy_score for question_type '{question_type}': {e}")
        reward = 0.0

    return reward


def calc_format_score(output_text, problem_type):
    if problem_type == "list":
        """
        Reward function for list answers.
        First we chech if answer has the <answer></anwswer> tag, reward=0.5
        Then we check if the answer tag contains a list
        """
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        match = re.fullmatch(pattern, output_text, re.DOTALL)
        if match:
            reward = 0.5
            try:
                output_ans = extract_answer(output_text)
                output_ans_list = ast.literal_eval(output_ans)
                if isinstance(output_ans_list, list):
                    reward = 1.0
            except Exception as e:
                reward = 0.5
        else:
            reward = 0
        return reward
    elif problem_type == "dict":
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        match = re.fullmatch(pattern, output_text, re.DOTALL)
        if match:
            # if only the tag format is matched, rerward 0.4
            reward = 0.4
            try:
                output_ans = extract_answer(output_text)
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

        return reward



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
    disp = math.sqrt((t2[0] - t1[0]) ** 2 + (t2[1] - t1[1]) ** 2)
    return round(disp, 3)


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
    return round(delta, 3)
    # return delta


def calc_general_dir(prev_translation, translation, prev_yaw, yaw, mode="outdoor"):
    """
    Calculate the agent's general heading direction.

    Parameters:
    - prev_translation: Previous position (x, y, z)
    - translation: Current position (x, y, z)
    - prev_yaw: Previous yaw angle in radians (clockwise positive, from quaternion_to_yaw)
    - yaw: Current yaw angle in radians (clockwise positive, from quaternion_to_yaw)

    Returns:
    - str: One of ["forward", "backward", "left", "slight left", "right", "slight right", "stationary"]
    """
    if mode == "outdoor":
        thresholds = {
            "stationary_disp": 0.05,
            "stationary_yaw": 0.00175,
            "forward_degree": 1,
            "slight_degree": 5,
            "turn_degree": 90,
            "back_turn_degree": 175,
            "back_slight_degree": 179
        }
    else:
        thresholds = {
            "stationary_disp": 0.1,
            "stationary_yaw": 0.005,
            "forward_degree": 0.3,
            "slight_degree": 3,
            "turn_degree": 90,
            "back_turn_degree": 175,
            "back_slight_degree": 179
        }
    # Calculate movement vector
    movement = np.array(translation) - np.array(prev_translation)

    # Check if the agent is stationary (very little movement and rotation)
    movement_magnitude = np.linalg.norm(movement)
    yaw_diff = abs((yaw - prev_yaw) % (2 * np.pi))
    if yaw_diff > np.pi:
        yaw_diff = 2 * np.pi - yaw_diff

    # If there's almost no movement and minimal rotation, agent is stationary
    if movement_magnitude < thresholds["stationary_disp"] and yaw_diff < thresholds["stationary_yaw"]:
        return "stationary"

    # If there is rotation, use the rotation change

    # Normalize the yaw difference to [-π, π]
    yaw_diff = (yaw - prev_yaw) % (2 * np.pi)
    if yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    angle_diff = math.degrees(yaw_diff)
    # Determine direction based on rotation
    # Note: Since yaw is clockwise positive, positive yaw_diff means turning right
    if abs(angle_diff) < thresholds["forward_degree"]:  # 1 degree, Almost no rotation
        return "forward"
    elif abs(angle_diff) < thresholds["slight_degree"]:  # Less than 5 degrees
        return "slight right" if yaw_diff > 0 else "slight left"
    elif abs(angle_diff) < thresholds["turn_degree"]:  # Less than 90 degrees
        return "right" if yaw_diff > 0 else "left"
    elif abs(angle_diff) < thresholds["back_turn_degree"]:  # Less than 175 degrees
        return "back right" if yaw_diff > 0 else "back left"
    elif abs(angle_diff) < thresholds["back_slight_degree"]:  # Less than 179 degrees
        return "slight back right" if yaw_diff > 0 else "slight back left"
    else:
        return "backward"


def calc_transformation_dict(prev_translation, translation, prev_quat, quat):
    movement = np.array(translation) - np.array(prev_translation)
    x_diff = movement[0]
    y_diff = movement[1]
    z_diff = movement[2]

    prev_q = [prev_quat[1], prev_quat[2], prev_quat[3], prev_quat[0]]
    q = [quat[1], quat[2], quat[3], quat[0]]
    
    try:
        prev_r = R.from_quat(prev_q)
        r = R.from_quat(q)
    except Exception as e:
        # print(f"Invalid quaternion: {prev_quat}, {quat} || {e}")
        return {"x": None, "y": None, "z": None, "roll": None, "pitch": None, "yaw": None}

    prev_roll, prev_pitch, prev_yaw = prev_r.as_euler('xyz', degrees=True)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)

    roll_diff = roll - prev_roll
    pitch_diff = pitch - prev_pitch
    yaw_diff = yaw - prev_yaw
    return {"x": x_diff, "y": y_diff, "z": z_diff, "roll": roll_diff, "pitch": pitch_diff, "yaw": yaw_diff}
    # return x_diff, y_diff, z_diff, roll_diff, pitch_diff, yaw_diff



def has_invalid_values(lst):
    return any(x is None or (isinstance(x, float) and math.isnan(x)) for x in lst)

def nusc_to_examples(nusc_dataset, mode, dataset_name, step_size=1, batch_size=4, video_out_dir=""):
    '''
        dataset_name: nusc or scannet
        /NuScenes/train_test or /ScanNet/decoded
    '''
    if video_out_dir != "":
        ''' prepare video writer '''
        sample_img = cv2.imread(nusc_dataset.img_filepaths[0])
        w = sample_img.shape[1]
        h = sample_img.shape[0]
        video_width = w
        video_height = h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out0 = cv2.VideoWriter(video_out_dir, fourcc, 10//step_size, (video_width, video_height))
        frame_text = "start of new sequence, no prev"
        ''' video writer code ends here '''
        
    meta_dict = nusc_dataset.meta_dict
    cam_list = nusc_dataset.camera_list
    cam = cam_list[0]

    meta_data = meta_dict[cam]
    ego_pose_list = meta_data["ego_pose_original"]

    prev_rotation = None
    prev_yaw = None
    prev_translation = None
    
    general_dir_gt_all = []
    disp_gt_all = []
    delta_heading_gt_all = []
    transformation_gt_all = []
    img_list_all = []

    batch_general_dir_gt = []
    batch_disp_gt = []
    batch_delta_heading_gt = []
    batch_transformation_gt = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []}
    batch_img_list = []
    sample_count = 0
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
            general_dir = calc_general_dir(prev_translation, translation, prev_yaw, yaw, mode=mode)
            disp = calculate_displacement(prev_translation, translation)
            delta_heading = calculate_delta_heading(prev_yaw, yaw)
            transformation_dict = calc_transformation_dict(prev_translation, translation, prev_rotation, rotation)

            batch_general_dir_gt.append(general_dir)
            batch_disp_gt.append(disp)
            batch_delta_heading_gt.append(delta_heading)
            for key in transformation_dict.keys():
                batch_transformation_gt[key].append(transformation_dict[key])

            frame_text = f"dir: {general_dir}, delta heading: {delta_heading}, displacement: {disp}"
            sample_count += 1
        else:
            frame_text = "start of new sequence, no prev"

        prev_rotation = rotation
        prev_yaw = yaw
        prev_translation = translation

        if sample_count != 0 and sample_count % (batch_size - 1) == 0:
            if len(batch_general_dir_gt) != batch_size - 1 or len(batch_disp_gt) != batch_size - 1 or len(batch_delta_heading_gt) != batch_size - 1 or len(
                    batch_img_list) != batch_size:
                
                # if this is residue at the end of a scene, don't use it
                if frame_i == len(ego_pose_list) - 1:
                    break
                else:
                    assert len(
                        batch_img_list) == batch_size, f"image path batch size mismatch {len(batch_img_list)}, {batch_size}"
                    assert len(
                        batch_general_dir_gt) == batch_size - 1, f"general direction batch size mismatch {len(batch_general_dir_gt)}, {batch_size - 1}"
                    assert len(
                        batch_disp_gt) == batch_size - 1, f"displacement batch size mismatch {len(batch_disp_gt)}, {batch_size - 1}"
                    assert len(
                        batch_delta_heading_gt) == batch_size - 1, f"delta_heading batch size mismatch {len(batch_delta_heading_gt)}, {batch_size - 1}"
            
            # if something is wrong in the dataset pose, causing Nan value in translation or rotation (observed in some ScanNet data), don't use this example
            if has_invalid_values(batch_disp_gt) or has_invalid_values(batch_delta_heading_gt) or has_invalid_values(batch_transformation_gt):
                print(f"Skipping invalid example in {dataset_name}: batch_disp_gt: {batch_disp_gt}, batch_delta_heading_gt: {batch_delta_heading_gt}, batch_transformation_gt: {batch_transformation_gt}")
                batch_general_dir_gt = []
                batch_disp_gt = []
                batch_delta_heading_gt = []
                batch_transformation_gt = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []}
                batch_img_list = []

                prev_yaw = None
                prev_translation = None
                sample_count = 0
            else:
                general_dir_gt_all.append(batch_general_dir_gt)
                disp_gt_all.append(batch_disp_gt)
                delta_heading_gt_all.append(batch_delta_heading_gt)
                transformation_gt_all.append(batch_transformation_gt)
                img_list_all.append(batch_img_list)

                batch_general_dir_gt = []
                batch_disp_gt = []
                batch_delta_heading_gt = []
                batch_transformation_gt = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []}
                batch_img_list = []
                prev_yaw = None
                prev_translation = None
                sample_count = 0


        if video_out_dir != "":
            img = cv2.imread(f"{nusc_dataset.img_filepaths[frame_i]}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, frame_text, (10, 30), font, fontScale=1, color=(0, 0, 255), thickness=2,
                        lineType=cv2.LINE_AA)
            out0.write(img)
        
    example_list = []
    problem_id_offset = 0
    for batch_i in range(len(disp_gt_all)):
        batch_general_dir_gt = general_dir_gt_all[batch_i]
        batch_disp_gt = disp_gt_all[batch_i]
        batch_delta_heading_gt = delta_heading_gt_all[batch_i]
        batch_transformation_gt = transformation_gt_all[batch_i]

        batch_img_list = img_list_all[batch_i]
        
        example_general_dir = {
            "problem_id": batch_i + problem_id_offset,
            "dataset_name": dataset_name,
            "problem": f"I uploaded {len(batch_img_list)} frames from an agent's dash cam video. The agent can be a vehicle or a person. \n"
                       f"Determine the agent's movement direction between each frame and its previous frame.\n"
                       f"Get your answer from the following options: {general_direction_options}.\n"
                       f"Keep all answers in one list. "
                       f"You should have {len(batch_general_dir_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "problem_subject": "general direction",
            "options": [],
            "solution": f"<answer>{batch_general_dir_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }
        problem_id_offset += 1

        example_heading = {
            "problem_id": batch_i + problem_id_offset,
            "dataset_name": dataset_name,
            "problem": f"I uploaded {len(batch_img_list)} frames from an agent's dash cam video. The agent can be a vehicle or a person. \n"
                       f"Determine the agent's change in heading direction (yaw) between each frame and its previous frame.\n"
                       f"Give your answer in degree values ranging between -180 and 180 degrees, with veer to the right "
                       f"being positive degrees and to the left being negative degrees.\n"
                       f"Keep all degree values in one list. "
                       f"You should have {len(batch_delta_heading_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "problem_subject": "heading",
            "options": [],
            "solution": f"<answer>{batch_delta_heading_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }
        problem_id_offset += 1

        example_disp = {
            "problem_id": batch_i + problem_id_offset,
            "dataset_name": dataset_name,
            "problem": f"I uploaded {len(batch_img_list)} frames from an agent's dash cam video. The agent can be a vehicle or a person. \n"
                       f"Determine the agent's displacement between each frame and its previous frame.\n"
                       f"Give your answer in numerical values in meter unit.\n"
                       f"Keep all displacement values in one list. "
                       f"You should have {len(batch_disp_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "problem_subject": "displacement",
            "options": [],
            "solution": f"<answer>{batch_disp_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }

        example_transformation = {
            "problem_id": batch_i + problem_id_offset,
            "dataset_name": dataset_name,
            "problem": f"I uploaded {len(batch_img_list)} frames from an agent's dash cam video. The agent can be a vehicle or a person. \n"
                       f"Determine the agent's exact translation (x, y, z) and rotation (roll, pitch yaw) values between each frame and its previous frame.\n"
                       f"For translation, give your answer in numerical values in meter unit.\n"
                       f"For rotation, give your answer in degree values ranging between -180 and 180 degrees, with veer to the right "
                       f"being positive degrees and to the left being negative degrees.\n"
                       f"Put your answer in a dictionary format. The dictionary sould have six keys: x, y, z, roll, pitch yaw. "
                       f"For each key, the value should be a list of meter or degree values. for keys x, y, and z, the list should contain {len(batch_disp_gt)} meter values. "
                       f"For keys roll, pitch and yaw, the list should contain {len(batch_disp_gt)} degree values. \n",
            "data_type": "video",
            "problem_type": "dict",
            "problem_subject": "transformation",
            "options": [],
            "solution": f"<answer>{batch_transformation_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }



        example_list.append(example_general_dir)
        example_list.append(example_heading)
        example_list.append(example_disp)
        example_list.append(example_transformation)

    return example_list


def prepare_dataset_nusc(example):
    dataset_name = example["dataset_name"]
    if dataset_name == "NuScenes":
        dataset_dir_specific = "NuScenes/train_test"
    elif dataset_name == "ScanNet":
        dataset_dir_specific = "ScanNet/decoded"
    else:
        return RuntimeError("dataset name not supported")
    
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']

    if example["problem_type"] == 'list' or example["problem_type"] == 'dict':
        msg = {
            "prompt":
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            example['data_type']: [f"{os.getenv('DATASET_DIR')}/{dataset_dir_specific}/{file_path}" for file_path in
                                                   example['path']]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                    ]
                }]
        }
    return msg



def eval_qwen(data, llm, sampling_params):
    '''
    input data: a list of examples
    this func evaluates examples in batches
    '''
    messages = []
    gt_list = []
    q_type_list = []

    for example in data:

        dataset_name = example["dataset_name"]
        if dataset_name == "NuScenes":
            dataset_dir_specific = "NuScenes/train_test"
        elif dataset_name == "ScanNet":
            dataset_dir_specific = "ScanNet/decoded"
        else:
            return RuntimeError("dataset name not supported")

        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: [f"{os.getenv('DATASET_DIR')}/{dataset_dir_specific}/{file_path}" for file_path in example['path']]
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        }]
        messages.append(msg)
        gt_list.append(example["solution"])
        q_type_list.append(example["problem_type"])

        # gt_text = example["solution"]
        #
        # prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs, video_kwargs = process_vision_info([msg], return_video_kwargs=True)
        # try:
        #     image_idx = 0
        #     video_idx = 0
        #     llm_inputs = []
        #     mm_type = msg[0]['content'][0]['type']
        #
        #     sample_mm_data = {}
        #     sample_video_kw = {}
        #     if mm_type == 'image':
        #         sample_mm_data["image"] = image_inputs[image_idx]
        #         image_idx += 1
        #     elif mm_type == 'video':
        #         sample_mm_data["video"] = video_inputs[video_idx]
        #         for key, value in video_kwargs.items():
        #             sample_video_kw[key] = value[video_idx]
        #         video_idx += 1
        #
        #     llm_inputs.append({
        #         "prompt": prompt,
        #         "multi_modal_data": sample_mm_data,
        #         "mm_processor_kwargs": sample_video_kw,
        #     })
        #
        #     outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        #     output_text = outputs[0].outputs[0].text
        #
        # except Exception as e:
        #     output_text = ['<answer>error</answer>']
        #
        # format_score = calc_format_score(output_text, example["problem_type"])
        # accuracy_score = calc_accuracy_score(output_text, gt_text, example["problem_type"])


    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
               messages]

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    image_idx = 0
    video_idx = 0
    llm_inputs = []

    for idx, prompt in enumerate(prompts):
        mm_type = messages[idx][0]['content'][0]['type']

        sample_mm_data = {}
        sample_video_kw = {}
        if mm_type == 'image':
            sample_mm_data["image"] = image_inputs[image_idx]
            image_idx += 1
        elif mm_type == 'video':
            sample_mm_data["video"] = video_inputs[video_idx]
            for key, value in video_kwargs.items():
                sample_video_kw[key] = value[video_idx]
            video_idx += 1

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": sample_mm_data,
            "mm_processor_kwargs": sample_video_kw,
        })

    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    output_text = [out.outputs[0].text for out in outputs]

    format_score_list = []
    accuracy_score_list = []
    for i in tqdm(range(len(messages)), desc="generating thought processes for a scene"):
        ans_text = extract_answer(output_text[i])
        gt_text = gt_list[i]
        q_type = q_type_list[i]

        format_score = calc_format_score(output_text[i], q_type)
        accuracy_score = calc_accuracy_score(output_text[i], gt_text, q_type)

        format_score_list.append(format_score)
        accuracy_score_list.append(accuracy_score)

    final_format_score = sum(format_score_list) / len(format_score_list)
    final_accuracy_score = sum(accuracy_score_list) / len(format_score_list)

    return final_format_score, final_accuracy_score



SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(args):
    # Load dataset
    num_cam = args.num_cam
    batch_size = args.batch_size
    eval_scene_start = args.eval_scene_start
    eval_scene_end = args.eval_scene_end
    root_path = os.getenv('DATASET_DIR')

    eval_example_list = []

    ''' =========================== load NuScenes dataset ============================= '''
    nusc_data_path = f"{root_path}/NuScenes/train_test"
    nusc_scene_idx_list = []
    for i in range(eval_scene_start, eval_scene_end):
        nusc_scene_idx_list.append(i)
    nusc_eval = NuScenes(version="v1.0-trainval", dataroot=nusc_data_path, verbose=True)
    ''' --------------------------- NuScenes train split --------------------------- '''
    print(f"Processing NuScenes train dataset into examples...")
    for scene_idx in nusc_scene_idx_list:
        step_size = random.randint(1, 10)
        print(f"NUSC, step size: {step_size}")
        dataset = NuScenesDataset(data_path=nusc_data_path,
                                  meta_out_path="",
                                  num_cams=num_cam,
                                  nusc=nusc_eval,
                                  scene_idx=scene_idx,
                                  save_meta=False)
        scene_example_list = nusc_to_examples(nusc_dataset=dataset, mode="outdoor", dataset_name="NuScenes", step_size=step_size, batch_size=batch_size)
        eval_example_list += scene_example_list

    ''' =========================== load ScanNet dataset ============================= '''
    scannet_data_path = f"{root_path}/ScanNet/decoded"
    scannet_scene_idx_list = []
    for i in range(eval_scene_start, eval_scene_end):
        nusc_scene_idx_list.append(i)
    ''' --------------------------- ScanNet train split --------------------------- '''
    print(f"Processing ScanNet train dataset into examples...")
    for scene_idx in scannet_scene_idx_list:
        step_size = random.randint(1, 10)
        print(f"ScanNet, step size: {step_size}")
        dataset = ScanNetDataset(data_path=scannet_data_path,
                                meta_out_path="",
                                scene_idx=scene_idx,
                                save_meta=False)
        scene_example_list = nusc_to_examples(nusc_dataset=dataset, mode="indoor", dataset_name="ScanNet", step_size=step_size, batch_size=batch_size, video_out_dir="")
        eval_example_list += scene_example_list

    ''' =========================== shuffle train examples ============================= '''
    random.shuffle(eval_example_list)
    final_format_score, final_accuracy_score = eval_qwen(eval_example_list, llm, sampling_params)
    print(f"evaluated {eval_scene_end - eval_scene_start} scenes with {len(eval_example_list)} examples (questions).")
    print(f"format score: {final_format_score}, accuracy score: {final_accuracy_score}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--num_cam", type=int, default=1)
    parser.add_argument("--eval_scene_start", type=int, default=0)
    parser.add_argument("--eval_scene_end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="/scratch/zl3466/github/thinking_in_street/model/Qwen")

    args = parser.parse_args()

    # MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_name = args.model
    model_path = args.model_path

    llm = LLM(
        model=model_name,
        download_dir=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype="float16"
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    main(args)
