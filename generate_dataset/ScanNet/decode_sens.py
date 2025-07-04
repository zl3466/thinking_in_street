import argparse
import os, sys

from SensorData import SensorData
from tqdm import tqdm

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--dataset_path', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--scene_start', default=0)
parser.add_argument('--scene_end', default=1)
parser.add_argument('--scene_idx', default=-1)
parser.add_argument('--frame_skip', default=5)
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')

parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()


def main():
    scannet_path = opt.dataset_path
    scene_list = os.listdir(scannet_path)
    scene_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    scene_start_idx = int(opt.scene_start)
    scene_end_idx = int(opt.scene_end)
    scene_idx = int(opt.scene_idx)

    frame_skip = int(opt.frame_skip)

    if scene_idx != -1:
        scene_name = scene_list[scene_idx]
        sens_file_path = f"{scannet_path}/{scene_name}/{scene_name}.sens"
        scene_out_path = f"{opt.output_path}/{scene_name}"
        if not os.path.exists(scene_out_path):
            os.makedirs(scene_out_path, exist_ok=True)

            # load the data
            sys.stdout.write('loading %s...' % sens_file_path)
            sd = SensorData(sens_file_path)
            sys.stdout.write('loaded!\n')
            if opt.export_depth_images:
                sd.export_depth_images(os.path.join(scene_out_path, 'depth'), frame_skip=frame_skip)
            if opt.export_color_images:
                sd.export_color_images(os.path.join(scene_out_path, 'color'), frame_skip=frame_skip)
            if opt.export_poses:
                sd.export_poses(os.path.join(scene_out_path, 'pose'), frame_skip=frame_skip)
            if opt.export_intrinsics:
                sd.export_intrinsics(os.path.join(scene_out_path, 'intrinsic'))
        else:
                print(f"scene {scene_name} alread exists at {scene_out_path}")
    else:
        if scene_end_idx is None:
            target_scene_list = scene_list[scene_start_idx:]
        else:
            target_scene_list = scene_list[scene_start_idx:scene_end_idx]
        for i in tqdm(range(len(target_scene_list)), desc="total scenes"):
            scene_name = target_scene_list[i]
            print(f"\n=========== decoding scene {scene_name} ===========")
            sens_file_path = f"{scannet_path}/{scene_name}/{scene_name}.sens"
            scene_out_path = f"{opt.output_path}/{scene_name}"
            if not os.path.exists(scene_out_path):
                os.makedirs(scene_out_path, exist_ok=True)
                # load the data
                sys.stdout.write('loading %s...' % sens_file_path)
                sd = SensorData(sens_file_path)
                sys.stdout.write('loaded!\n')
                if opt.export_depth_images:
                    sd.export_depth_images(os.path.join(scene_out_path, 'depth'), frame_skip=frame_skip)
                if opt.export_color_images:
                    sd.export_color_images(os.path.join(scene_out_path, 'color'), frame_skip=frame_skip)
                if opt.export_poses:
                    sd.export_poses(os.path.join(scene_out_path, 'pose'), frame_skip=frame_skip)
                if opt.export_intrinsics:
                    sd.export_intrinsics(os.path.join(scene_out_path, 'intrinsic'))
            else:
                print(f"scene {scene_name} alread exists at {scene_out_path}")


if __name__ == '__main__':
    main()
