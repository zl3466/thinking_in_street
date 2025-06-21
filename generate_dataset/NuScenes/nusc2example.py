import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
from utils.train_utils import *
import argparse
from nuscenes.nuscenes import NuScenes

def main(args):
    # Load dataset
    root_path = os.getenv('DATASET_DIR')
    video_length_list = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

    ''' =========================== load NuScenes dataset ============================= '''
    nusc_data_path = f"{root_path}/NuScenes/train_test"

    nusc_sft_idx_list = list(range(0, 100))
    nusc_grpo_idx_list = list(range(100, 750))
    nusc_test_idx_list = list(range(0, 150))
    nusc_eval_idx_list = list(range(750, 850))

    nusc_trainval = NuScenes(version="v1.0-trainval", dataroot=nusc_data_path, verbose=True)
    nusc_test = NuScenes(version="v1.0-test", dataroot=nusc_data_path, verbose=True)

    ''' =========================== generate examples for NuScenes dataset ============================= '''
    ''' --------------------------- NuScenes SFT Train Split --------------------------- '''
    
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/sft/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)

        nusc_out_json = f"{step_out_dir}/nusc_examples.json"
        # if os.path.exists(nusc_out_json):
        #     nusc_example_dict = json.load(open(nusc_out_json))
        # else:
        #     nusc_example_dict = {"forward": {}, 
        #                          "backward": {}}
        nusc_example_dict = {"forward": {}, 
                            "backward": {}}
        print(f"Processing NuScenes sft train split examples...")
        for scene_idx in tqdm(nusc_sft_idx_list, desc=f"Processing NuScenes scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = NuScenesDataset(data_path=nusc_data_path,
                                      meta_out_path="",
                                      num_cams=1,
                                      nusc=nusc_trainval,
                                      scene_idx=scene_idx,
                                      save_meta=False)

            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in nusc_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    nusc_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["forward"][str(video_length)].keys():
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if len(backward_example_list) == 0:
                    continue
                if str(video_length) not in nusc_example_dict["backward"].keys():
                    nusc_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["backward"][str(video_length)].keys():
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save NuScenes examples for this step size ============================= '''
        with open(nusc_out_json, "w") as outfile:
            json.dump(nusc_example_dict, outfile, indent=4)
            print(f"NuScenes forward and backward examples saved to {nusc_out_json}")

    ''' --------------------------- NuScenes GRPO Train Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/train/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        
        nusc_out_json = f"{step_out_dir}/nusc_examples.json"
        nusc_example_dict = {"forward": {}, 
                            "backward": {}}
        
        print(f"Processing NuScenes grpo train split examples...")
        for scene_idx in tqdm(nusc_grpo_idx_list, desc=f"Processing NuScenes scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = NuScenesDataset(data_path=nusc_data_path,
                                      meta_out_path="",
                                      num_cams=1,
                                      nusc=nusc_trainval,
                                      scene_idx=scene_idx,
                                      save_meta=False)

            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in nusc_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    nusc_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["forward"][str(video_length)].keys():
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in nusc_example_dict["backward"].keys():
                    nusc_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["backward"][str(video_length)].keys():
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save NuScenes examples for this step size ============================= '''
        with open(nusc_out_json, "w") as outfile:
            json.dump(nusc_example_dict, outfile, indent=4)
            print(f"NuScenes forward and backward examples saved to {nusc_out_json}")

    ''' --------------------------- NuScenes Test Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/test/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        nusc_out_json = f"{step_out_dir}/nusc_examples.json"
        nusc_example_dict = {"forward": {}, 
                            "backward": {}}
        
        print(f"Processing NuScenes test split examples...")
        for scene_idx in tqdm(nusc_test_idx_list, desc=f"Processing NuScenes scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = NuScenesDataset(data_path=nusc_data_path,
                                      meta_out_path="",
                                      num_cams=1,
                                      nusc=nusc_test,
                                      scene_idx=scene_idx,
                                      save_meta=False)


            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in nusc_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    nusc_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["forward"][str(video_length)].keys():
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in nusc_example_dict["backward"].keys():
                    nusc_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["backward"][str(video_length)].keys():
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save NuScenes examples for this step size ============================= '''
        with open(nusc_out_json, "w") as outfile:
            json.dump(nusc_example_dict, outfile, indent=4)
            print(f"NuScenes forward and backward examples saved to {nusc_out_json}")

    ''' --------------------------- NuScenes Eval Split --------------------------- '''
    for step_size in range(1, 11):
        step_out_dir = f"{args.out_dir}/eval/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)
        nusc_out_json = f"{step_out_dir}/nusc_examples.json"
        nusc_example_dict = {"forward": {}, 
                            "backward": {}}
            
        print(f"Processing NuScenes eval split examples...")
        for scene_idx in tqdm(nusc_eval_idx_list, desc=f"Processing NuScenes scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = NuScenesDataset(data_path=nusc_data_path,
                                      meta_out_path="",
                                      num_cams=1,
                                      nusc=nusc_trainval,
                                      scene_idx=scene_idx,
                                      save_meta=False)


            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                # sometimes a scene may not have enough images for the stepsize * video length combo (e.g. scene only has 300 images, not enough for step_size = 10 * video_len = 32)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in nusc_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    nusc_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["forward"][str(video_length)].keys():
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="outdoor", dataset_name="NuScenes", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in nusc_example_dict["backward"].keys():
                    nusc_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in nusc_example_dict["backward"][str(video_length)].keys():
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        nusc_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

                    
        # ''' =========================== save NuScenes examples for this step size ============================= '''
        with open(nusc_out_json, "w") as outfile:
            json.dump(nusc_example_dict, outfile, indent=4)
            print(f"NuScenes forward and backward examples saved to {nusc_out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--out_dir", type=str)

    args = parser.parse_args()

    main(args)

