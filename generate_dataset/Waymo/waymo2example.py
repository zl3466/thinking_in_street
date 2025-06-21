import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
from utils.train_utils import *
import argparse


def main(args):
    # Load dataset
    root_path = os.getenv('DATASET_DIR')
    video_length_list = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

    ''' =========================== load Waymo dataset ============================= '''
    waymo_data_path = f"{root_path}/Waymo"

    waymo_sft_idx_list = list(range(0, 100))
    waymo_grpo_idx_list = list(range(100, 798))
    waymo_test_idx_list = list(range(0, 150))
    waymo_eval_idx_list = list(range(0, 202))

    ''' =========================== generate examples for waymo dataset ============================= '''
    ''' --------------------------- Waymo SFT Train Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/sft/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)

        waymo_out_json = f"{step_out_dir}/waymo_examples.json"
        
        waymo_example_dict = {"forward": {}, 
                                "backward": {}}

        print(f"Processing Waymo sft train split examples...")
        for scene_idx in tqdm(waymo_sft_idx_list, desc=f"Processing waymo scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = WaymoDataset(data_path=waymo_data_path,
                                    meta_out_path="",
                                    split="training",
                                    scene_idx=scene_idx,
                                    save_meta=False)


            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in waymo_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    waymo_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["forward"][str(video_length)].keys():
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if len(backward_example_list) == 0:
                    continue
                if str(video_length) not in waymo_example_dict["backward"].keys():
                    waymo_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["backward"][str(video_length)].keys():
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save Waymo examples for this step size ============================= '''
        with open(waymo_out_json, "w") as outfile:
            json.dump(waymo_example_dict, outfile, indent=4)
            print(f"Waymo forward and backward examples saved to {waymo_out_json}")

    ''' --------------------------- Waymo GRPO Train Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/train/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        waymo_out_json = f"{step_out_dir}/waymo_examples.json"
        waymo_example_dict = {"forward": {}, 
                                "backward": {}}
        
        print(f"Processing Waymo grpo train split examples...")
        for scene_idx in tqdm(waymo_grpo_idx_list, desc=f"Processing waymo scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = WaymoDataset(data_path=waymo_data_path,
                                    meta_out_path="",
                                    split="training",
                                    scene_idx=scene_idx,
                                    save_meta=False)

            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in waymo_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    waymo_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["forward"][str(video_length)].keys():
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in waymo_example_dict["backward"].keys():
                    waymo_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["backward"][str(video_length)].keys():
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save Waymo examples for this step size ============================= '''
        with open(waymo_out_json, "w") as outfile:
            json.dump(waymo_example_dict, outfile, indent=4)
            print(f"Waymo forward and backward examples saved to {waymo_out_json}")

    ''' --------------------------- Waymo Test Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/test/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        waymo_out_json = f"{step_out_dir}/waymo_examples.json"
        waymo_example_dict = {"forward": {}, 
                                "backward": {}}
        
        print(f"Processing Waymo test split examples...")
        for scene_idx in tqdm(waymo_test_idx_list, desc=f"Processing waymo scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = WaymoDataset(data_path=waymo_data_path,
                                    meta_out_path="",
                                    split="testing",
                                    scene_idx=scene_idx,
                                    save_meta=False)


            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in waymo_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    waymo_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["forward"][str(video_length)].keys():
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in waymo_example_dict["backward"].keys():
                    waymo_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["backward"][str(video_length)].keys():
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save Waymo examples for this step size ============================= '''
        with open(waymo_out_json, "w") as outfile:
            json.dump(waymo_example_dict, outfile, indent=4)
            print(f"Waymo forward and backward examples saved to {waymo_out_json}")

    ''' --------------------------- Waymo Eval Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/eval/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        waymo_out_json = f"{step_out_dir}/waymo_examples.json"
        waymo_example_dict = {"forward": {}, 
                                "backward": {}}
            
        print(f"Processing Waymo eval split examples...")
        for scene_idx in tqdm(waymo_eval_idx_list, desc=f"Processing waymo scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = WaymoDataset(data_path=waymo_data_path,
                                    meta_out_path="",
                                    split="validation",
                                    scene_idx=scene_idx,
                                    save_meta=False)


            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in waymo_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    waymo_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["forward"][str(video_length)].keys():
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="Waymo", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in waymo_example_dict["backward"].keys():
                    waymo_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in waymo_example_dict["backward"][str(video_length)].keys():
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        waymo_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save Waymo examples for this step size ============================= '''
        with open(waymo_out_json, "w") as outfile:
            json.dump(waymo_example_dict, outfile, indent=4)
            print(f"Waymo forward and backward examples saved to {waymo_out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--out_dir", type=str)

    args = parser.parse_args()

    main(args)

