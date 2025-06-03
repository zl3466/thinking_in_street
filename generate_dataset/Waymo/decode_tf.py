import argparse

import numpy as np

# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import box_utils, range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops


# To make waymo preprocessing works without installing mmcv
# Code is modified from mmcv.utils
import sys
import time
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(
                f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:"
            )
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.start_time = time.time()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(
                self.bar_width,
                int(self.terminal_width - len(msg)) + 2,
                int(self.terminal_width * 0.6),
            )
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,"
                f" {fps:.1f} tasks/s"
            )
        self.file.flush()


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(
    func,
    tasks,
    nproc,
    initializer=None,
    initargs=None,
    bar_width=50,
    chunksize=1,
    skip_first=False,
    keep_order=True,
    file=sys.stdout,
):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write("\n")
    pool.close()
    pool.join()
    return results


MOVEABLE_OBJECTS_IDS = [
    cs_pb2.CameraSegmentation.TYPE_CAR,
    cs_pb2.CameraSegmentation.TYPE_TRUCK,
    cs_pb2.CameraSegmentation.TYPE_BUS,
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE,
    cs_pb2.CameraSegmentation.TYPE_BICYCLE,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE,
    cs_pb2.CameraSegmentation.TYPE_TRAILER,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN,
    cs_pb2.CameraSegmentation.TYPE_CYCLIST,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST,
    cs_pb2.CameraSegmentation.TYPE_BIRD,
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT,
]


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(
            range_image_origins, dtype=range_image_polar_dtype
        )
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian


def parse_range_image_flow_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if (
            len(laser.ri_return1.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytes(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytes(range_image_top_pose_str_tensor.numpy())
                )

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytes(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if (
            len(laser.ri_return2.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytes(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytes(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud_flow(
    frame,
    range_images,
    range_images_flow,
    camera_projections,
    range_image_top_pose,
    ri_index=0,
):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    origins, points, cp_points = [], [], []
    points_intensity = []
    points_elongation = []
    points_flow = []
    laser_ids = []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4])
    )
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_flow = range_images_flow[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_flow_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]

        flow_x = range_image_flow_tensor[..., 0]
        flow_y = range_image_flow_tensor[..., 1]
        flow_z = range_image_flow_tensor[..., 2]
        flow_class = range_image_flow_tensor[..., 3]

        mask_index = tf.where(range_image_mask)

        (origins_cartesian, points_cartesian,) = extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        origins_cartesian = tf.squeeze(origins_cartesian, axis=0)
        points_cartesian = tf.squeeze(points_cartesian, axis=0)

        origins_tensor = tf.gather_nd(origins_cartesian, mask_index)
        points_tensor = tf.gather_nd(points_cartesian, mask_index)

        points_intensity_tensor = tf.gather_nd(range_image_intensity, mask_index)
        points_elongation_tensor = tf.gather_nd(range_image_elongation, mask_index)

        points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, mask_index), axis=1)
        points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, mask_index), axis=1)
        points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, mask_index), axis=1)
        points_flow_class_tensor = tf.expand_dims(
            tf.gather_nd(flow_class, mask_index), axis=1
        )

        origins.append(origins_tensor.numpy())
        points.append(points_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())
        laser_ids.append(np.full_like(points_intensity_tensor.numpy(), c.name - 1))

        points_flow.append(
            tf.concat(
                [
                    points_flow_x_tensor,
                    points_flow_y_tensor,
                    points_flow_z_tensor,
                    points_flow_class_tensor,
                ],
                axis=-1,
            ).numpy()
        )

    return (
        origins,
        points,
        points_flow,
        cp_points,
        points_intensity,
        points_elongation,
        laser_ids,
    )


class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "calib",
            "pose"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            "_FRONT"
        ]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)
        self.tfrecord_pathnames = [
            f"{self.load_dir}/{prefix}/{f}" for f in os.listdir(f"{self.load_dir}/{prefix}")
        ]
        # self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames = sum(1 for _ in dataset)
        for frame_idx, data in enumerate(
            tqdm(dataset, desc=f"File {file_idx}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(data.numpy()))
            if (
                self.selected_waymo_locations is not None
                and frame.context.stats.location not in self.selected_waymo_locations
            ):
                continue
            if "images" in self.process_keys:
                self.save_image(frame, file_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(frame, file_idx, frame_idx)
            # if "lidar" in self.process_keys:
            #     self.save_lidar(frame, file_idx, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(frame, file_idx, frame_idx)
            # if "dynamic_masks" in self.process_keys:
            #     self.save_dynamic_mask(frame, file_idx, frame_idx)
            # if frame_idx == 0:
            #     self.save_interested_labels(frame, file_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            if not str(img.name - 1) == "0":
                continue
            img_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        extrinsics = []
        intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/extrinsics/"
                + f"{str(i)}.txt",
                extrinsics[i],
            )
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/intrinsics/"
                + f"{str(i)}.txt",
                intrinsics[i],
            )

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{str(file_idx).zfill(3)}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            pose,
        )


    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/ego_pose",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/extrinsics",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{str(i).zfill(3)}/intrinsics",
                exist_ok=True,
            )




if __name__ == "__main__":
    """
    Waymo Dataset preprocessing script
    ===========================

    This script facilitates the preprocessing of the Waymo dataset

    Usage:
    ------
    python preprocess.py \
        --data_root <path_to_waymo_data> \
        --target_dir <output_directory> \
        [additional_arguments]

    Example:
    --------
    python preprocess.py --data_root data/waymo/raw/ --target_dir data/waymo/processed --split training --workers 3 --scene_ids 700 754 114

    Arguments:
    ----------
    --data_root (str):
        The root directory where the Waymo dataset is stored. This is a required argument.

    --split (str):
        Specifies the name of the data split. Default is set to "training".

    --target_dir (str):
        Designates the directory where the processed data will be saved. This is a mandatory argument.

    --workers (int):
        The number of processing threads. Default is set to 4.

    --scene_ids (list[int]):
        List of specific scene IDs for processing. Should be integers separated by spaces.

    --split_file (str):
        If provided, indicates the path to a file located in `data/waymo_splits` that contains the desired scene IDs.

    --start_idx (int):
        Used in conjunction with `num_scenes` to generate a list of scene IDs when neither `scene_ids` nor `split_file` are provided.

    --num_scenes (int):
        The total number of scenes to be processed.

    --process_keys (list[str]):
        Denotes the types of data components to be processed. Options include but aren't limited to "images", "lidar", "calib", "pose", etc.

    Notes:
    ------
    The logic of the script ensures that if specific scene IDs (`scene_ids`) are provided, they are prioritized. 
    If a split file (`split_file`) is indicated, it is utilized next. 
    If neither is available, the script uses the `start_idx` and `num_scenes` parameters to determine the scene IDs.
    """
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of waymo dataset"
    )
    parser.add_argument("--split", type=str, default="training", help="split name")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    # priority: scene_ids > split_file > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "calib",
            "pose"
        ],
    )
    args = parser.parse_args()
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    elif args.num_scenes is not None:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)
    else:
        scene_ids_list = None

    waymo_processor = WaymoProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        prefix=args.split,
        process_keys=args.process_keys,
        process_id_list=scene_ids_list,
        workers=args.workers,
    )
    if args.scene_ids is not None and args.workers == 1:
        for scene_id in args.scene_ids:
            waymo_processor.convert_one(scene_id)
    else:
        waymo_processor.convert()
