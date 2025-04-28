# Copyright 2024. All rights reserved.
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
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any
from nuscenes.nuscenes import LidarPointCloud, NuScenes

from train.data_loader.nuscenes import NuScenesDataset
from utils.train_utils import *

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")


def prepare_dataset_sft(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
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

    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: [f"{os.getenv('DATASET_DIR')}/{dataset_dir_specific}/{file_path}" for
                                            file_path in
                                            example['path']]
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
        }
    ]

    return messages



def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"],
                                                                           return_video_kwargs=True)

        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ''' ========================= load dataset ========================= '''
    # Load dataset

    # get number of scenes to train / test on
    # we get equal number of scenes from NuScenes and Scannet -- each half of the total NUM_TRAIN_SCENE
    # train - test split = 5:1
    # test split is only used if the transformer.training_args has eval_strategy set to not "no". We keep it no for now.
    example_dir = os.getenv("EXAMPLE_DIR")
    train_full_data_dict = {"NuScenes": {}, "ScanNet": {}}
    for step_size in range(1, 11):
        train_full_data_dict["NuScenes"][str(step_size)] = json.load(open(f"{example_dir}/sft/step_{step_size}/nusc_examples.json"))
        train_full_data_dict["ScanNet"][str(step_size)] = json.load(open(f"{example_dir}/sft/step_{step_size}/scannet_examples.json"))

    # get same number of scenes for NuScenes and ScanNet
    # num_train_scene = int(os.getenv("NUM_TRAIN_SCENE"))
    train_scene_start = int(os.getenv("TRAIN_SCENE_START"))
    train_scene_end = int(os.getenv("TRAIN_SCENE_END"))


    # Collect train and test data with random step size but fixed video length
    # Put NuScenes and ScanNet data together
    video_length = int(os.getenv("VIDEO_LENGTH"))
    ''' Train split: for each scene, choose a random step_size (frame rate) '''
    train_example_list = []
    for scene_idx in range(train_scene_start, train_scene_end):
        step_size = random.randint(1, 11)
        nusc_scene_list = list(train_full_data_dict["NuScenes"][str(step_size)]["forward"][str(video_length)].keys())
        scannet_scene_list = list(train_full_data_dict["ScanNet"][str(step_size)]["forward"][str(video_length)].keys())
        
        nusc_scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scannet_scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # get the set of examples for specified video length
        try:
            nusc_scene = nusc_scene_list[scene_idx]
            nusc_example_list = train_full_data_dict["NuScenes"][str(step_size)]["forward"][str(video_length)][nusc_scene]
            train_example_list += nusc_example_list 
        except:
            print(f"there is a total of {nusc_scene_list} scenes in nusc train dataset, requesting {scene_idx}th, does not exist")
            
        try:
            scannet_scene = scannet_scene_list[scene_idx]
            scannet_example_list = train_full_data_dict["ScanNet"][str(step_size)]["forward"][str(video_length)][scannet_scene]
            train_example_list += scannet_example_list
        except:
            print(f"there is a total of {scannet_scene_list} scenes in scannet train dataset, requesting {scene_idx}th, does not exist")

    random.shuffle(train_example_list)

    ''' ========================= Prepare dataset (process examples into messages) ========================= '''
    dataset = DatasetDict({
        "train": Dataset.from_list(train_example_list)
    })
    prepared_dataset = [prepare_dataset_sft(example) for example in dataset['train']]
    
    ''' ========================= setup model ========================= '''
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )

    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
