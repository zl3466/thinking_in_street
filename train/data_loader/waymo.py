import logging
import os
from typing import Dict

import numpy as np
import torch
import json

from datasets.base.scene_dataset import SceneDataset
from scipy.spatial.transform import Rotation as R
from utils.qwen_utils import NumpyEncoder
logger = logging.getLogger()


class WaymoDataset():
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    def __init__(
        self,
        data_path: str,
        meta_out_path: str,
        num_cams: int = 1,
        split: str = 'training',
        scene_idx: int = 0,
        start_timestep: int = 0,
        end_timestep: int = -1,
        save_meta=True
    ):
        logger.info("Loading new Waymo dataset.")
        self.data_path = data_path
        self.meta_out_path = meta_out_path
        self.num_cams = num_cams
        self.split = split
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.save_meta = save_meta

        self.scene_idx = scene_idx
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()
        self.load_calibrations()


    def create_or_load_metas(self):
        # ---- define `camera list ---- #
        self.camera_list = ["CAM_FRONT"]
        
        if os.path.exists(self.meta_out_path):
            # print(self.meta_out_path)
            with open(self.meta_out_path, "r") as f:
                meta_dict = json.load(f)
            logger.info(f"[Waymo] Loaded camera meta from {self.meta_out_path}")
            return meta_dict
        else:
            logger.info(f"[Waymo] Creating camera meta at {self.meta_out_path}")

        scene_list = os.listdir(self.data_path)
        scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scene_name = scene_list[self.scene_idx]

        # if self.SensorData is None:
        #     self.SensorData = SensorData(self.data_path)

        total_camera_list = ["CAM_FRONT"]

        meta_dict = {
            camera: {
                "timestamp": [],
                "filepath": [],
                "ego_pose_original": [],
                "ego_pose_matrix": [],
                "cam_id": [],
                "extrinsics": [],
                "intrinsics": [],
            }
            for i, camera in enumerate(total_camera_list)
        }

        # start loading data into meta_dict
        img_dir = f"{self.data_path}/{scene_name}/images"
        pose_dir = f"{self.data_path}/{scene_name}/ego_pose"
        intrinsic_dir = f"{self.data_path}/{scene_name}/intrinsics/0.txt"
        extrinsic_dir = f"{self.data_path}/{scene_name}/extrinsics/0.txt"

        img_filelist = os.listdir(img_dir)
        img_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        pose_filelist = os.listdir(pose_dir)
        pose_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        camera = "CAM_FRONT"
        for i in range(len(img_filelist)):
            img_filename = img_filelist[i]
            img_path = f"{img_dir}/{img_filename}"
            pose_filename = pose_filelist[i]
            pose_path = f"{pose_dir}/{pose_filename}"

            # ---- timestamp and cam_id ---- #
            meta_dict[camera]["cam_id"].append(0)
            meta_dict[camera]["timestamp"].append(int(img_filename.split(".")[0]))
            meta_dict[camera]["filepath"].append(f"{self.split}/{scene_name}/images/{img_filename}")

            # ---- intrinsics and extrinsics ---- #
            # intrinsics
            intrinsic = np.loadtxt(intrinsic_dir)
            meta_dict[camera]["intrinsics"].append(np.array(intrinsic))

            # extrinsics
            extrinsic = np.loadtxt(extrinsic_dir)
            meta_dict[camera]["extrinsics"].append(extrinsic)

            # ---- ego pose ---- #
            ego_pose_matrix = np.loadtxt(pose_path)
            rotation = ego_pose_matrix[:3, :3]
            translation = ego_pose_matrix[:3, 3]

            # # orthogonalize rotation matrix using svd
            # U, _, Vt = np.linalg.svd(rotation)
            # rotation = U @ Vt
            try:
                U, _, Vt = np.linalg.svd(rotation + np.eye(3) * 1e-10)
                rotation = U @ Vt
            except np.linalg.LinAlgError:
                # If SVD fails, keep rotation as it is
                pass

            # q = Quaternion(matrix=rotation)
            # rotation_quat = np.array([q.w, q.x, q.y, q.z])
            r = R.from_matrix(rotation)
            quat_xyzw = r.as_quat()
            quat_wxyz = [quat_xyzw[3], *quat_xyzw[:3]]

            meta_dict[camera]["ego_pose_original"].append({"rotation": quat_wxyz, "translation": translation})
            meta_dict[camera]["ego_pose_matrix"].append(ego_pose_matrix)


        if self.save_meta:
            with open(self.meta_out_path, "w") as f:
                json.dump(meta_dict, f, cls=NumpyEncoder)
            logger.info(f"[Pixel] Saved camera meta to {self.meta_out_path}")
        return meta_dict


    def create_all_filelist(self):
        # ---- find the minimum shared scene length ---- #
        num_timestamps = 100000000
        for camera in self.camera_list:
            if len(self.meta_dict[camera]["timestamp"]) < num_timestamps:
                num_timestamps = len(self.meta_dict[camera]["timestamp"])
        logger.info(f"[Pixel] Min shared scene length: {num_timestamps}")
        self.scene_total_num_timestamps = num_timestamps

        if self.end_timestep == -1:
            self.end_timestep = num_timestamps - 1
        else:
            self.end_timestep = min(self.end_timestep, num_timestamps - 1)

        # to make sure the last timestep is included
        self.end_timestep += 1
        self.start_timestep = min(self.start_timestep, self.end_timestep - 1)

        logger.info(f"[Pixel] Start timestep: {self.start_timestep}")
        logger.info(f"[Pixel] End timestep: {self.end_timestep}")

        img_filepaths, rel_img_filepaths, feat_filepaths, sky_mask_filepaths = [], [], [], []
        # TODO: support dynamic masks

        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepath = os.path.join(
                    self.data_path, self.meta_dict[cam_idx]["filepath"][t]
                )
                img_filepaths.append(img_filepath)
                rel_img_filepaths.append(self.meta_dict[cam_idx]["filepath"][t])

        self.img_filepaths = np.array(img_filepaths)
        self.rel_img_filepaths = np.array(rel_img_filepaths)


    def load_calibrations(self):
        # compute per-image poses and intrinsics
        cam_to_worlds = []
        intrinsics, timesteps, cam_ids = [], [], []
        timestamps = []

        # we tranform the camera poses w.r.t. the first timestep to make the origin of
        # the first ego pose  as the origin of the world coordinate system.
        initial_ego_to_global = self.meta_dict["CAM_FRONT"]["ego_pose_matrix"][
            self.start_timestep
        ]
        global_to_initial_ego = np.linalg.inv(initial_ego_to_global)

        min_timestamp = 1e20
        max_timestamp = 0
        for cam_name in self.camera_list:
            min_timestamp = min(
                min_timestamp,
                self.meta_dict[cam_name]["timestamp"][self.start_timestep],
            )
            max_timestamp = max(
                max_timestamp,
                self.meta_dict[cam_name]["timestamp"][self.end_timestep - 1],
            )
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp

        for t in range(self.start_timestep, self.end_timestep):
            for cam_name in self.camera_list:
                cam_to_ego = self.meta_dict[cam_name]["extrinsics"][t]
                ego_to_global_current = self.meta_dict[cam_name]["ego_pose_matrix"][t]
                # compute ego_to_world transformation
                ego_to_world = global_to_initial_ego @ ego_to_global_current
                # Because we use opencv coordinate system to generate camera rays,
                # we need to store the transformation from opencv coordinate system to dataset
                # coordinate system. However, the nuScenes dataset uses the same coordinate
                # system as opencv, so we just store the identity matrix.
                # opencv coordinate system: x right, y down, z front
                cam_to_ego = cam_to_ego @ self.OPENCV2DATASET
                cam2world = ego_to_world @ cam_to_ego
                cam_to_worlds.append(cam2world)
                intrinsics.append(self.meta_dict[cam_name]["intrinsics"][t])
                timesteps.append(t - self.start_timestep)
                cam_ids.append(self.meta_dict[cam_name]["cam_id"][t])
                timestamps.append(
                    self.meta_dict[cam_name]["timestamp"][t]
                    * np.ones_like(
                        self.meta_dict[cam_name]["cam_id"][t], dtype=np.float64
                    )
                )

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()

        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.global_to_initial_ego = torch.from_numpy(global_to_initial_ego).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.tensor(timestamps, dtype=torch.float64)
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()