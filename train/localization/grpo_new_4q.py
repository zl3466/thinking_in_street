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
from train.data_loader.scannet import ScanNetDataset
import cv2
import random
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.train_utils import *



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
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

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load dataset

    # get number of scenes to train / test on
    # we get equal number of scenes from NuScenes and Scannet -- each half of the total NUM_TRAIN_SCENE
    # train - test split = 5:1
    # test split is only used if the transformer.training_args has eval_strategy set to not "no". We keep it no for now.
    example_dir = os.getenv("EXAMPLE_DIR")
    # get same number of scenes for NuScenes and ScanNet
    # train_scene is scene 0-400
    train_scene_start = int(os.getenv("TRAIN_SCENE_START"))
    train_scene_end = int(os.getenv("TRAIN_SCENE_END"))
    # test_scene is scene 400-500
    test_scene_start = train_scene_start // 4
    test_scene_end = train_scene_end // 4

    # Collect train and test data with random step size but fixed video length
    # Put NuScenes and ScanNet data together
    video_length = int(os.getenv("VIDEO_LENGTH"))
    train_example_list = json.load(open(f"{example_dir}/train/train_examples.json"))
    test_example_list = json.load(open(f"{example_dir}/test/test_examples.json"))

    ''' =========================== assemble into full dataset ============================= '''
    dataset = DatasetDict({
        "train": Dataset.from_list(train_example_list),
        "test": Dataset.from_list(test_example_list)
    })
    ''' ------------------ convert examples into input messages for Qwen model ------------------ '''
    dataset = dataset.map(prepare_dataset_nusc)

    ''' =========================== Start trainer ============================= '''
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)
    print(f"\n====== train {train_scene_start}-{train_scene_end} scenes, test {test_scene_start}-{test_scene_end} scenes ======\n")

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
        print(f"resuming from checkpoint {training_args.resume_from_checkpoint}")
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
    print(training_args)
    main(script_args, training_args, model_args)
