from nuscenes.nuscenes import LidarPointCloud, NuScenes
from train.data_loader.nuscenes import NuScenesDataset
import cv2
import random
import time
import json

def nusc_to_examples(nusc_dataset, video_out_dir, sample_rate=10, batch_size=4):
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

        if prev_yaw is not None and prev_translation is not None:
            disp = calculate_displacement(prev_translation, translation)
            delta_heading = calculate_delta_heading(prev_yaw, yaw)
            batch_disp_gt.append(disp)
            batch_delta_heading_gt.append(delta_heading)
            batch_img_list.append(nusc_dataset.img_filepaths[frame_i])

            frame_text = f"delta heading: {delta_heading}, displacement: {disp}"

        prev_yaw = yaw
        prev_translation = translation

        if sample_count != 0 and sample_count % (batch_size - 1) == 0:
            # assert len(
            #     batch_disp_gt) == batch_size - 1, f"displacement batch size mismatch {len(batch_disp_gt)}, {batch_size - 1}"
            # assert len(
            #     batch_delta_heading_gt) == batch_size - 1, f"delta_heading batch size mismatch {len(batch_delta_heading_gt)}, {batch_size - 1}"
            if len(batch_disp_gt) != batch_size - 1 or len(batch_delta_heading_gt) != batch_size - 1:
                # if this is residue at the end of a scene, don't use it

                if frame_i == len(ego_pose_list) - 1:
                    break
                else:
                    assert len(batch_disp_gt) == batch_size - 1, f"displacement batch size mismatch {len(batch_disp_gt)}, {batch_size - 1}"
                    assert len(batch_delta_heading_gt) == batch_size - 1, f"delta_heading batch size mismatch {len(batch_delta_heading_gt)}, {batch_size - 1}"

            disp_gt_all.append(batch_disp_gt)
            delta_heading_gt_all.append(batch_delta_heading_gt)
            img_list_all.append(batch_img_list)
            batch_disp_gt = []
            batch_delta_heading_gt = []
            batch_img_list = []


        sample_count += 1

    example_list = []
    problem_id_offset = 0
    for batch_i in range(len(disp_gt_all)):
        batch_disp_gt = disp_gt_all[batch_i]
        batch_delta_heading_gt = delta_heading_gt_all[batch_i]
        batch_img_list = img_list_all[batch_i]

        example_heading = {
            "problem_id": batch_i + problem_id_offset,
            "problem": f"I uploaded {len(batch_disp_gt)+1} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's change in heading direction between each frame and its previous frame.\n"
                       f"Give your answer in degree values ranging between -180 and 180 degrees, with veer to the right "
                       f"being positive degrees and to the left being negative degrees.\n"
                       f"Keep all degree values in one list. "
                       f"You should have {len(batch_disp_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "options": [],
            "process": "",
            "solution": f"<answer>{batch_delta_heading_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }
        problem_id_offset += 1

        example_disp = {
            "problem_id": batch_i + problem_id_offset,
            "problem": f"I uploaded {len(batch_disp_gt)+1} frames from a vehicle dash cam video. \n"
                       f"Determine the vehicle's displacement between each frame and its previous frame.\n"
                       f"Give your answer in numerical values in meter unit.\n"
                       f"Keep all displacement values in one list. "
                       f"You should have {len(batch_disp_gt)} values in the list.\n",
            "data_type": "video",
            "problem_type": "list",
            "options": [],
            "process": "",
            "solution": f"<answer>{batch_disp_gt}</answer>",
            "path": batch_img_list,
            "reward": 1.0,
            "select": True
        }

        example_list.append(example_heading)
        example_list.append(example_disp)

    return example_list

'''  
Generate the gt for thinking process using Gemini
We use a uniform frame rate of 2hz for inputs
One set of gt for each batch size (# of frames per video): 4, 8, 16
'''
def main():
    num_cam = 1
    train_num_scene = 850
    test_num_scene = 150
    sample_rate = 2
    video_length = 4

    train_scene_idx_list = []
    test_scene_idx_list = []
    for i in range(train_num_scene):
        train_scene_idx_list.append(i)
    for i in range(test_num_scene):
        test_scene_idx_list.append(i)

    root_path = script_args.dataset_name
    out_path = f"./train_result/scene_{train_scene_idx_list[0]}-{train_scene_idx_list[-1]}_cam_{num_cam}"

    train_data_path = f"{root_path}/train"
    test_data_path = f"{root_path}/test"

    train_out_path = f"{out_path}/train"
    test_out_path = f"{out_path}/test"
    
    # nusc_train = None
    # nusc_test = None
    nusc_train = NuScenes(version="v1.0-trainval", dataroot=train_data_path, verbose=True)
    nusc_test = NuScenes(version="v1.0-test", dataroot=test_data_path, verbose=True)
    
    train_example_list = []
    print(f"Processing train dataset into examples...")
    for scene_idx in train_scene_idx_list:
        scene_out_path = f"{train_out_path}/scene_{scene_idx}"
        os.makedirs(scene_out_path, exist_ok=True)

        dataset = NuScenesDataset(data_path=train_data_path,
                                  meta_out_path=f"{scene_out_path}/meta_dict.json",
                                  num_cams=num_cam,
                                  nusc=nusc_train,
                                  scene_idx=scene_idx)

        train_example_list += nusc_to_examples(dataset, video_out_dir=f"{scene_out_path}/scene_{scene_idx}.mp4", sample_rate=sample_rate, batch_zize=video_length)

    test_example_list = []
    print(f"Processing test dataset into examples...")
    for scene_idx in test_scene_idx_list:
        scene_out_path = f"{test_out_path}/scene_{scene_idx}"
        os.makedirs(scene_out_path, exist_ok=True)

        dataset = NuScenesDataset(data_path=test_data_path,
                                  meta_out_path=f"{scene_out_path}/meta_dict.json",
                                  num_cams=num_cam,
                                  nusc=nusc_test,
                                  scene_idx=scene_idx)

        test_example_list += nusc_to_examples(dataset, video_out_dir=f"{scene_out_path}/scene_{scene_idx}.mp4", sample_rate=sample_rate, batch_zize=video_length)

    # Format into conversation
    # dataset = dataset.map(make_conversation_image_and_video)
    # prepared_dataset_train = [prepare_dataset_nusc(example) for example in train_example_list]
    # prepared_dataset_test = [prepare_dataset_nusc(example) for example in test_example_list]
    with open(f"{out_path}/train_examples.json", 'w') as f:
        json.dump(train_example_list, f, indent=4)
    with open(f"{out_path}/test_examples.json", 'w') as f:
        json.dump(test_example_list, f, indent=4)