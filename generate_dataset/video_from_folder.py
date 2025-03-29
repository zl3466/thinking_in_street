import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *



root_dir = "/home/zl3466/Documents/github/thinking_in_street/data/long_route_random_single"
map_name = "map1_downtown_bk"
data_dir = f"{root_dir}/{map_name}"
img_folder = f"{data_dir}/frames"
video_out_file = f"{data_dir}/{map_name}.mp4"


success = generate_video_from_img(img_folder, video_out_file, frame_rate=10, reverse=False)

if success:
    print(f"Video generation complete. Check {video_out_file}")
else:
    print("Failed to generate video")