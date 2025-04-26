import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from utils.train_utils import *
import argparse
from nuscenes.nuscenes import NuScenes

def main(args):
    # Load dataset
    num_cam = args.num_cam
    scene_start = args.scene_start
    scene_end = args.scene_end
    root_path = os.getenv('DATASET_DIR')
    out_root = args.out_dir

    video_length_list = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

    ''' =========================== load NuScenes dataset ============================= '''
    nusc_data_path = f"{root_path}/NuScenes/train_test"
    nusc_scene_idx_list = []
    for i in range(scene_start, scene_end):
        nusc_scene_idx_list.append(i)
    nusc_eval = NuScenes(version="v1.0-trainval", dataroot=nusc_data_path, verbose=True)

    ''' =========================== load ScanNet dataset ============================= '''
    scannet_data_path = f"{root_path}/ScanNet/decoded"
    scannet_scene_idx_list = []
    for i in range(scene_start, scene_end):
        scannet_scene_idx_list.append(i)

    ''' =========================== generate examples for both datasets ============================= '''
    for step_size in range(1, 11):
        step_out_dir = f"{out_root}/step_{step_size}"
        os.makedirs(step_out_dir, exist_ok=True)

        # we manage NuScenes and ScanNet seperately
        nusc_out_json = f"{step_out_dir}/nusc_examples.json"
        scannet_out_json = f"{step_out_dir}/scannet_examples.json"
        if os.path.exists(nusc_out_json):
            nusc_example_dict = json.load(open(nusc_out_json))
        else:
            nusc_example_dict = {"forward": {}, 
                                 "backward": {}}
        if os.path.exists(scannet_out_json):
            scannet_example_dict = json.load(open(scannet_out_json))
        else:
            scannet_example_dict = {"forward": {}, 
                                    "backward": {}}


        ''' --------------------------- NuScenes --------------------------- '''
        print(f"Processing NuScenes train dataset into examples...")
        for scene_idx in tqdm(nusc_scene_idx_list, desc=f"Processing nusc scenes, step_sze {step_size}"):
            # print(f"NUSC, step size: {step_size}")
            dataset = NuScenesDataset(data_path=nusc_data_path,
                                      meta_out_path="",
                                      num_cams=num_cam,
                                      nusc=nusc_eval,
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

        ''' --------------------------- ScanNet --------------------------- '''
        print(f"Processing ScanNet train dataset into examples...")
        for scene_idx in tqdm(scannet_scene_idx_list, desc=f"Processing scannet scenes, step_sze {step_size}"):
            # print(f"ScanNet, step size: {step_size}")
            dataset = ScanNetDataset(data_path=scannet_data_path,
                                     meta_out_path="",
                                     scene_idx=scene_idx,
                                     save_meta=False)
            # Generate datasets for different batch size (video length)
            for video_length in video_length_list:
                ''' generate forward examples '''
                forward_example_list = dataset_to_examples(dataset=dataset, mode="indoor", dataset_name="ScanNet", scene_idx=scene_idx, step_size=step_size, batch_size=video_length)
                if len(forward_example_list) == 0:
                    continue
                if str(video_length) not in scannet_example_dict["forward"].keys():
                    # nusc_example_dict["forward"][str(video_length)] = forward_example_list
                    scannet_example_dict["forward"][str(video_length)] = {f"scene_{scene_idx}": forward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in scannet_example_dict["forward"][str(video_length)].keys():
                        scannet_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] = forward_example_list
                    # if this scene is already in the dict
                    else:
                        scannet_example_dict["forward"][str(video_length)][f"scene_{scene_idx}"] += forward_example_list

                ''' generate reversed examples '''
                backward_example_list = dataset_to_examples(dataset=dataset, mode="indoor", dataset_name="ScanNet", scene_idx=scene_idx, step_size=step_size, batch_size=video_length, reverse=True)
                if str(video_length) not in scannet_example_dict["backward"].keys():
                    scannet_example_dict["backward"][str(video_length)] = {f"scene_{scene_idx}": backward_example_list}
                else:
                    # if this is a new scene
                    if f"scene_{scene_idx}" not in scannet_example_dict["backward"][str(video_length)].keys():
                        scannet_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] = backward_example_list
                    # if this scene is already in the dict
                    else:
                        scannet_example_dict["backward"][str(video_length)][f"scene_{scene_idx}"] += backward_example_list

        # ''' =========================== save ScanNet examples for this step size ============================= '''
        with open(scannet_out_json, "w") as outfile:
            json.dump(scannet_example_dict, outfile, indent=4)
            print(f"ScanNet forward and backward examples saved to {scannet_out_json}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thought process from NuScenes with Gemini.")
    parser.add_argument("--num_cam", type=int, default=1)
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=10)
    parser.add_argument("--out_dir", type=str)

    args = parser.parse_args()

    main(args)

