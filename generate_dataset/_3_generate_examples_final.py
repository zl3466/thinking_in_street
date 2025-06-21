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
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datetime import datetime
from dataclasses import dataclass, field
import random
from utils.train_utils import *



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def main(args):
    '''
    NuScenes RGB image is 10Hz
    ScanNet RGB image is originally 60Hz, we downsampled it to 10Hz

    step_size = 1 -> 10Hz
    step_size = 2 -> 5Hz
    step_size = 3 -> 3.3Hz
    ...
    step_size = 10 -> 1Hz

    
    video_length: number of frames per video
    These frames will be sent to model for training / inference


    num_cam: always 1 for now.
    TODO implement surround view (multiple cam perspectives concatenated together)?
    '''

    # Load dataset
    example_dir = os.getenv("EXAMPLE_DIR")
    idx_list_dict = {
        "sft": {"NuScenes": list(range(0, 100)), "ScanNet": list(range(0, 150)), "Waymo": list(range(0, 100))},
        "grpo": {"NuScenes": list(range(0, 650)), "ScanNet": list(range(0, 1163)), "Waymo": list(range(0, 698))},
        "test": {"NuScenes": list(range(0, 150)), "ScanNet": list(range(0, 100)), "Waymo": list(range(0, 150))},
        "eval": {"NuScenes": list(range(0, 100)), "ScanNet": list(range(0, 200)), "Waymo": list(range(0, 202))},
        
    }
    # nusc_sft_idx_list = list(range(0, 100))
    # nusc_grpo_idx_list = list(range(0, 650))
    # nusc_test_idx_list = list(range(0, 150))
    # nusc_eval_idx_list = list(range(0, 100))

    # scannet_sft_idx_list = list(range(0, 150))
    # scannet_grpo_idx_list = list(range(0, 1163))
    # scannet_test_idx_list = list(range(0, 100))
    # scannet_eval_idx_list = list(range(0, 200))

    # waymo_sft_idx_list = list(range(0, 100))
    # waymo_grpo_idx_list = list(range(0, 698))
    # waymo_test_idx_list = list(range(0, 150))
    # waymo_eval_idx_list = list(range(0, 202))

    sft_full_data_dict = {"NuScenes": {}, "ScanNet": {}, "Waymo": {}}
    grpo_full_data_dict = {"NuScenes": {}, "ScanNet": {}, "Waymo": {}}
    test_full_data_dict = {"NuScenes": {}, "ScanNet": {}, "Waymo": {}}
    eval_full_data_dict = {"NuScenes": {}, "ScanNet": {}, "Waymo": {}}
    for step_size in range(1, 11):

        sft_full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/sft/step_{step_size}/nusc_examples.json"))
        sft_full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/sft/step_{step_size}/scannet_examples.json"))
        sft_full_data_dict["Waymo"][str(step_size)] = json.load(open(f"{example_dir}/sft/step_{step_size}/waymo_examples.json"))

        grpo_full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/train/step_{step_size}/nusc_examples.json"))
        grpo_full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/train/step_{step_size}/scannet_examples.json"))
        grpo_full_data_dict["Waymo"][str(step_size)] = json.load(open(f"{example_dir}/train/step_{step_size}/waymo_examples.json"))

        test_full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/test/step_{step_size}/nusc_examples.json"))
        test_full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/test/step_{step_size}/scannet_examples.json"))
        test_full_data_dict["Waymo"][str(step_size)] = json.load(open(f"{example_dir}/test/step_{step_size}/waymo_examples.json"))

        eval_full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/eval/step_{step_size}/nusc_examples.json"))
        eval_full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/eval/step_{step_size}/scannet_examples.json"))
        eval_full_data_dict["Waymo"][str(step_size)] = json.load(open(f"{example_dir}/eval/step_{step_size}/waymo_examples.json"))


    # Collect train and test data with random step size but fixed video length
    # Put NuScenes and ScanNet data together
    video_length = int(os.getenv("VIDEO_LENGTH"))

    ''' ====================================================================================== '''
    ''' ========= GRPO Train split: for each scene, choose a step_size (frame rate) between 1-10 ========= '''
    sft_example_list = []
    grpo_example_list = []
    test_example_list = []
    eval_example_list = []
    # each step size has a list of scenes; 
    # scenes available for each step size are not necessarily the same, so we need to load the list in each loop
    for dataset_name in ["NuScenes", "ScanNet", "Waymo"]:
        ''' =============== SFT split =============== '''
        step_size = 1
        for scene_idx in tqdm(idx_list_dict["sft"][dataset_name], desc=f"SFT {dataset_name}"):
            if step_size > 10:
                step_size = 1
            
            scene_list = list(sft_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)].keys())
            scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            try:
                scene = scene_list[scene_idx]
                example_list = sft_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)][scene]
                print(f"{dataset_name} scene {scene_idx} step_size {step_size}: {len(example_list)}")
                sft_example_list += example_list
            except:
                print(f"there is a total of {len(scene_list)} scenes in {dataset_name} train dataset, requesting {scene_idx}th, does not exist")
            step_size += 1
        
        ''' =============== GRPO split =============== '''
        step_size = 1
        for scene_idx in tqdm(idx_list_dict["grpo"][dataset_name], desc=f"GRPO {dataset_name}"):
            if step_size > 10:
                step_size = 1
            
            scene_list = list(grpo_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)].keys())
            scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            try:
                scene = scene_list[scene_idx]
                example_list = grpo_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)][scene]
                print(f"{dataset_name} scene {scene_idx} step_size {step_size}: {len(example_list)}")
                grpo_example_list += example_list
            except:
                print(f"there is a total of {len(scene_list)} scenes in {dataset_name} train dataset, requesting {scene_idx}th, does not exist")
            step_size += 1
        
        ''' =============== Test split =============== '''
        step_size = 1
        for scene_idx in tqdm(idx_list_dict["test"][dataset_name], desc=f"Test {dataset_name}"):
            if step_size > 10:
                step_size = 1
            
            scene_list = list(test_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)].keys())
            scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            try:
                scene = scene_list[scene_idx]
                example_list = test_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)][scene]
                print(f"{dataset_name} scene {scene_idx} step_size {step_size}: {len(example_list)}")
                test_example_list += example_list
            except:
                print(f"there is a total of {len(scene_list)} scenes in {dataset_name} train dataset, requesting {scene_idx}th, does not exist")
            step_size += 1
        
        ''' =============== Eval split =============== '''
        step_size = 1
        for scene_idx in tqdm(idx_list_dict["eval"][dataset_name], desc=f"Eval {dataset_name}"):
            if step_size > 10:
                step_size = 1
            
            scene_list = list(eval_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)].keys())
            scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            try:
                scene = scene_list[scene_idx]
                example_list = eval_full_data_dict[dataset_name][str(step_size)]["forward"][str(video_length)][scene]
                print(f"{dataset_name} scene {scene_idx} step_size {step_size}: {len(example_list)}")
                eval_example_list += example_list
            except:
                print(f"there is a total of {len(scene_list)} scenes in {dataset_name} train dataset, requesting {scene_idx}th, does not exist")
            step_size += 1


    ''' =========================== shuffle examples for each split ============================= '''
    print(f"generated {len(sft_example_list)} sft examples,\n {len(grpo_example_list)} grpo examples,\n {len(test_example_list)} test examples,\n {len(eval_example_list)} eval examples")
    random.shuffle(sft_example_list)
    random.shuffle(grpo_example_list)
    random.shuffle(test_example_list)
    random.shuffle(eval_example_list)

    # save the shuffled training and testing examples to json
    with open(f"{args.out_dir}/sft/sft_examples.json", 'w') as f:
        json.dump(sft_example_list, f, indent=4)
    with open(f"{args.out_dir}/train/train_examples.json", 'w') as f:
        json.dump(grpo_example_list, f, indent=4)
    with open(f"{args.out_dir}/test/test_examples.json", 'w') as f:
        json.dump(test_example_list, f, indent=4)
    with open(f"{args.out_dir}/eval/eval_examples.json", 'w') as f:
        json.dump(eval_example_list, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final examples for all splits")
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    main(args)
