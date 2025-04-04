import json
import logging
import os

import numpy as np
import torch
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from pyquaternion import Quaternion
from torch import Tensor
from utils.qwen_utils import NumpyEncoder

logger = logging.getLogger()

class NuScenesDataset():
    # ORIGINAL_SIZE = [[900, 1600] for _ in range(6)]
    OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def __init__(
        self,
        data_path: str,
        meta_out_path: str,
        num_cams: int = 1,
        nusc: NuScenes = None,
        split: str = 'train',
        scene_idx: int = 0,
        start_timestep: int = 0,
        end_timestep: int = -1
    ):

        logger.info("Loading new NuScenes dataset.")
        self.data_path = data_path
        self.meta_out_path = meta_out_path
        self.num_cams = num_cams
        self.split = split
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.nusc = nusc
        self.scene_idx = scene_idx
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()
        self.load_calibrations()


    def create_or_load_metas(self):
        # ---- define camera list ---- #
        if self.num_cams == 1:
            self.camera_list = ["CAM_FRONT"]
        elif self.num_cams == 3:
            self.camera_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        elif self.num_cams == 6:
            self.camera_list = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for nuscenes dataset"
            )

        if os.path.exists(self.meta_out_path):
            # print(self.meta_out_path)
            with open(self.meta_out_path, "r") as f:
                meta_dict = json.load(f)
            logger.info(f"[Nuscenes] Loaded camera meta from {self.meta_out_path}")
            return meta_dict
        else:
            logger.info(f"[Nuscenes] Creating camera meta at {self.meta_out_path}")

        if self.nusc is None:
            if self.split == "train":
                self.nusc = NuScenes(
                    version="v1.0-trainval", dataroot=self.data_path, verbose=True
                )
            else:
                self.nusc = NuScenes(
                    version="v1.0-test", dataroot=self.data_path, verbose=True
                )
        self.scene = self.nusc.scene[self.scene_idx]
        total_camera_list = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

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

        # ---- get the first sample of each camera ---- #
        current_camera_data_tokens = {camera: None for camera in total_camera_list}
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        for camera in total_camera_list:
            current_camera_data_tokens[camera] = first_sample["data"][camera]

        while not all(token == "" for token in current_camera_data_tokens.values()):
            for i, camera in enumerate(total_camera_list):
                # skip if the current camera data token is empty
                if current_camera_data_tokens[camera] == "":
                    continue

                current_camera_data = self.nusc.get(
                    "sample_data", current_camera_data_tokens[camera]
                )

                # ---- timestamp and cam_id ---- #
                meta_dict[camera]["cam_id"].append(i)
                meta_dict[camera]["timestamp"].append(current_camera_data["timestamp"])
                meta_dict[camera]["filepath"].append(current_camera_data["filename"])

                # ---- intrinsics and extrinsics ---- #
                calibrated_sensor_record = self.nusc.get(
                    "calibrated_sensor", current_camera_data["calibrated_sensor_token"]
                )
                # intrinsics
                intrinsic = calibrated_sensor_record["camera_intrinsic"]
                meta_dict[camera]["intrinsics"].append(np.array(intrinsic))

                # extrinsics
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = Quaternion(
                    calibrated_sensor_record["rotation"]
                ).rotation_matrix
                extrinsic[:3, 3] = np.array(calibrated_sensor_record["translation"])
                meta_dict[camera]["extrinsics"].append(extrinsic)

                # ---- ego pose ---- #
                ego_pose_record = self.nusc.get(
                    "ego_pose", current_camera_data["ego_pose_token"]
                )
                ego_pose = np.eye(4)
                ego_pose[:3, :3] = Quaternion(
                    ego_pose_record["rotation"]
                ).rotation_matrix
                ego_pose[:3, 3] = np.array(ego_pose_record["translation"])
                meta_dict[camera]["ego_pose_original"].append(ego_pose_record)
                meta_dict[camera]["ego_pose_matrix"].append(ego_pose)

                current_camera_data_tokens[camera] = current_camera_data["next"]

        with open(self.meta_out_path, "w") as f:
            json.dump(meta_dict, f, cls=NumpyEncoder)
        logger.info(f"[Pixel] Saved camera meta to {self.meta_out_path}")
        return meta_dict

    def create_all_filelist(self):
        # NuScenes dataset is not synchronized, so we need to find the minimum shared
        # scene length, and only use the frames within the shared scene length.
        # we also define the start and end timestep within the shared scene length

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

        img_filepaths, feat_filepaths, sky_mask_filepaths = [], [], []
        # TODO: support dynamic masks

        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepath = os.path.join(
                    self.data_path, self.meta_dict[cam_idx]["filepath"][t]
                )
                img_filepaths.append(img_filepath)

        self.img_filepaths = np.array(img_filepaths)


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

        # # scale the intrinsics according to the load size
        # self.intrinsics[..., 0, 0] *= (
        #     self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        # )
        # self.intrinsics[..., 1, 1] *= (
        #     self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        # )
        # self.intrinsics[..., 0, 2] *= (
        #     self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        # )
        # self.intrinsics[..., 1, 2] *= (
        #     self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        # )

        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.global_to_initial_ego = torch.from_numpy(global_to_initial_ego).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.tensor(timestamps, dtype=torch.float64)
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()