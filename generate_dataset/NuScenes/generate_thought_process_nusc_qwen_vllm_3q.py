import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch

import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from train.data_loader.nuscenes import NuScenesDataset
import cv2
import random
import time
import math

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "list": " Please provide the list answer (e.g., [val, val, val, ...]) within the <answer> </answer> tags."
}

general_direction_options = ["stationary", "forward", "backward", "left", "slight left", "back left", "slight back left", "right", "slight right", "back right", "slight back right"]


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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


def compute_bleu_score(reference, hypothesis):
    try:
        smoothing = SmoothingFunction().method1
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0


def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure


def reward_fn(sample, model_output, question_type):
    try:
        output_ans = extract_answer(model_output)
        gt_ans = extract_answer(sample.get("solution", ""))
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "OCR":
            error_rate = wer(gt_ans, output_ans)
            reward = 1 - error_rate
            return max(0.0, min(1.0, reward))
        elif question_type == "free-form":
            score = compute_rouge_score(gt_ans, output_ans)
            return max(0.0, min(1.0, score))
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
            rel_diff = min(1.0, max(0.0, rel_diff))
            return 1 - rel_diff
        else:
            return 0.0
    except Exception as e:
        print(f"Error in reward_fn for question_type '{question_type}': {e}")
        return 0.0


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




def nusc_to_examples(nusc_dataset, mode, step_size=1, batch_size=4, video_out_dir=""):
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

    prev_yaw = None
    prev_translation = None
    
    general_dir_gt_all = []
    disp_gt_all = []
    delta_heading_gt_all = []
    img_list_all = []

    batch_general_dir_gt = []
    batch_disp_gt = []
    batch_delta_heading_gt = []
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

            batch_general_dir_gt.append(general_dir)
            batch_disp_gt.append(disp)
            batch_delta_heading_gt.append(delta_heading)
            frame_text = f"dir: {general_dir}, delta heading: {delta_heading}, displacement: {disp}"
            sample_count += 1
        else:
            frame_text = "start of new sequence, no prev"

        prev_yaw = yaw
        prev_translation = translation

        if sample_count != 0 and sample_count % (batch_size - 1) == 0:
            if len(batch_disp_gt) != batch_size - 1 or len(batch_delta_heading_gt) != batch_size - 1 or len(
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

            general_dir_gt_all.append(batch_general_dir_gt)
            disp_gt_all.append(batch_disp_gt)
            delta_heading_gt_all.append(batch_delta_heading_gt)
            img_list_all.append(batch_img_list)
            batch_general_dir_gt = []
            batch_disp_gt = []
            batch_delta_heading_gt = []
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
        batch_img_list = img_list_all[batch_i]

        example_general_dir = {
            "problem_id": batch_i + problem_id_offset,
            "problem": f"I uploaded {len(batch_img_list)} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's movement direction between each frame and its previous frame.\n"
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
            "problem": f"I uploaded {len(batch_img_list)} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's change in heading direction between each frame and its previous frame.\n"
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
            "problem": f"I uploaded {len(batch_img_list)} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's displacement between each frame and its previous frame.\n"
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

        example_list.append(example_general_dir)
        example_list.append(example_heading)
        example_list.append(example_disp)

    return example_list


def ask_qwen(model, processor, messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def gen_thought_process(data, llm, sampling_params):
    '''
    input data: a list of examples
    '''
    messages = []
    for x in data:
        if x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            question = x['problem']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: [f"{os.getenv('DATASET_DIR')}/{file_path}" for file_path in x['path']]
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                }
            ]
        }]
        messages.append(msg)

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
               messages]

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    try:
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

    except Exception as e:
        output_text = ['<answer>error</answer>']

    for i in tqdm(range(len(messages)), desc="generating thought processes for a scene"):
        think_chain = extract_think(output_text[i])
        if think_chain:
            data[i]["process"] = f"<think>{think_chain}</think>"
        else:
            data[i]["process"] = f"<think></think>"

    return data


def main(args):
    ''' ========================== generate examples from NuScenes dataset ============================ '''
    num_cam = args.num_cam
    # train_num_scene = args.train_num_scene
    # test_num_scene = args.test_num_scene
    train_scene_start = args.train_scene_start
    train_scene_end = args.train_scene_end
    test_scene_start = args.test_scene_start
    test_scene_end = args.test_scene_end

    # sample_rate = args.sample_rate
    batch_size = args.batch_size
    data_root_path = args.data_root_path
    ''' =============================================================================== '''

    train_scene_idx_list = []
    test_scene_idx_list = []
    for i in range(train_scene_start, train_scene_end):
        train_scene_idx_list.append(i)
    for i in range(test_scene_start, test_scene_end):
        test_scene_idx_list.append(i)

    train_out_path = f"{data_root_path}/examples/qwen_vllm_3q/train/scene_{train_scene_idx_list[0]}-{train_scene_idx_list[-1]}_cam_{num_cam}_frame_{batch_size}"
    test_out_path = f"{data_root_path}/examples/qwen_vllm_3q/test/scene_{test_scene_idx_list[0]}-{test_scene_idx_list[-1]}_cam_{num_cam}_frame_{batch_size}"

    train_data_path = f"{data_root_path}/train_test"
    test_data_path = f"{data_root_path}/train_test"

    os.makedirs(train_out_path, exist_ok=True)
    os.makedirs(test_out_path, exist_ok=True)

    nusc_train = NuScenes(version="v1.0-trainval", dataroot=train_data_path, verbose=True)
    nusc_test = NuScenes(version="v1.0-test", dataroot=test_data_path, verbose=True)

    train_example_json_dict = {}
    print(f"Processing train dataset into examples...")
    for scene_idx in train_scene_idx_list:
        step_size = random.randint(1, 10)
        print(f"step size: {step_size}")
        dataset = NuScenesDataset(data_path=train_data_path,
                                  meta_out_path="",
                                  num_cams=num_cam,
                                  nusc=nusc_train,
                                  scene_idx=scene_idx,
                                  save_meta=False)
        scene_example_list = nusc_to_examples(nusc_dataset=dataset, mode="outdoor", step_size=step_size, batch_size=batch_size)
        train_example_json_dict[f'scene_{scene_idx}'] = gen_thought_process(scene_example_list, llm, sampling_params)

        with open(f"{train_out_path}/train_examples.json", 'w') as f:
            json.dump(train_example_json_dict, f, indent=4)

    test_example_json_dict = {}
    print(f"Processing test dataset into examples...")
    for scene_idx in test_scene_idx_list:
        step_size = random.randint(1, 10)
        print(f"step size: {step_size}")
        dataset = NuScenesDataset(data_path=test_data_path,
                                  meta_out_path="",
                                  num_cams=num_cam,
                                  nusc=nusc_test,
                                  scene_idx=scene_idx,
                                  save_meta=False)
        scene_example_list = nusc_to_examples(nusc_dataset=dataset, mode="outdoor", step_size=step_size, batch_size=batch_size)
        test_example_json_dict[f'scene_{scene_idx}'] = gen_thought_process(scene_example_list, llm, sampling_params)

        with open(f"{test_out_path}/test_examples.json", 'w') as f:
            json.dump(test_example_json_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--num_cam", type=int, default=1)
    parser.add_argument("--train_scene_start", type=int, default=0)
    parser.add_argument("--train_scene_end", type=int, default=10)
    parser.add_argument("--test_scene_start", type=int, default=0)
    parser.add_argument("--test_scene_end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_root_path", type=str, default="NuScenes")
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
        max_tokens=512,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(model_name,
                                              cache_dir="/scratch/zl3466/github/thinking_in_street/model/Qwen")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir="/scratch/zl3466/github/thinking_in_street/model/Qwen")
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    main(args)
