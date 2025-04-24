import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

from utils.train_utils import *


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

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
               messages]
    # print(f"messages: \n{messages}\n")
    # print(f"Prompts: \n{prompts}\n")
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

    # except Exception as e:
    #     output_text = ['<answer>error</answer>']

    for i in tqdm(range(len(messages)), desc="generating thought processes for a scene"):
        think_chain = extract_think(output_text[i])
        if think_chain:
            data[i]["process"] = f"<think>{think_chain}</think>"
            data[i]["select"] = True
        else:
            data[i]["process"] = f"<think></think>"
            data[i]["select"] = False

    return data


def main(args):
    ''' ========================== generate examples from NuScenes dataset ============================ '''
    example_dir = args.example_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    full_data_dict = {"NuScenes": {}, "ScanNet": {}}
    for step_size in range(1, 11):

        full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/cold_start/step_{step_size}/nusc_examples.json"))
        full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/cold_start/step_{step_size}/scannet_examples.json"))

    scene_start = args.scene_start
    scene_end = args.scene_end

    # json file to save the examples with thought process filled
    nusc_json = f"{out_dir}/nusc_thought_process.json"
    scannet_json = f"{out_dir}/scannet_thought_process.json"

    # add to existing json if the json already exists
    if os.path.exists(nusc_json):
        nusc_result_dict = json.load(open(nusc_json))
    else:
        nusc_result_dict = {"forward": {}, "backward": {}}

    if os.path.exists(nusc_json):
        scannet_result_dict = json.load(open(scannet_json))
    else:
        scannet_result_dict = {"forward": {}, "backward": {}}
    
    for scene_idx in range(scene_start, scene_end):
        # random step size (frame rate) and video length (number of frames)
        step_size = random.randint(1, 10)
        video_length = random.choice(video_length_list)

        print(f"using scene {scene_idx}, step {step_size}, video len {video_length}")
        # print(full_data_dict["NuScenes"][str(step_size)]["forward"][str(video_length)])
        # forward examples
        nusc_example_list = full_data_dict["NuScenes"][str(step_size)]["forward"][str(video_length)][f"scene_{scene_idx}"]
        scannet_example_list = full_data_dict["ScanNet"][str(step_size)]["forward"][str(video_length)][f"scene_{scene_idx}"]
        # backward examples
        nusc_backward_example_list = full_data_dict["NuScenes"][str(step_size)]["backward"][str(video_length)][f"scene_{scene_idx}"]
        scannet_backward_example_list = full_data_dict["ScanNet"][str(step_size)]["backward"][str(video_length)][f"scene_{scene_idx}"]

        # generate thought process for forward examples
        filled_example_list = gen_thought_process(nusc_example_list, llm, sampling_params)
        nusc_result_dict["forward"][f"scene_{scene_idx}"] = filled_example_list
        filled_example_list = gen_thought_process(scannet_example_list, llm, sampling_params)
        scannet_result_dict["forward"][f"scene_{scene_idx}"] = filled_example_list
        # generate thought process for backward examples
        filled_example_list = gen_thought_process(nusc_backward_example_list, llm, sampling_params)
        nusc_result_dict["backward"][f"scene_{scene_idx}"] = filled_example_list
        filled_example_list = gen_thought_process(scannet_backward_example_list, llm, sampling_params)
        scannet_result_dict["backward"][f"scene_{scene_idx}"] = filled_example_list

        # save results for this scene
        with open(nusc_json, 'w') as f:
            json.dump(nusc_result_dict, f, indent=4)
        with open(scannet_json, 'w') as f:
            json.dump(scannet_result_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--example_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--scene_start", type=int, default=500)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    
    args = parser.parse_args()

    # MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_name = args.model_name
    model_path = args.model_path

    llm = LLM(
        model=model_name,
        download_dir=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 10, "video": 10}
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
