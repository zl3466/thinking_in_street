import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.qwen_utils import *

def generate_ranges(total_length, clip_size=20):
    ranges = []
    start = 0
    while start < total_length:
        end = min(start + clip_size, total_length)
        ranges.append([start, end])
        start = end
    return ranges


def main():
    data_dir = "../../data/long_route_random_single"
    result_dir = "../../result/long_route_random_single/qwen"
    map_list = os.listdir(data_dir)

    # map_name = "map1_downtown_bk"
    for map_name in map_list:
        if map_name != "map1_downtown_bk":
            continue
        print("====================================")
        print(f"Processing Map {map_name}")
        map_root_dir = f"{data_dir}/{map_name}"
        out_dir = f"{result_dir}/odometry/{map_name}"
        if not os.path.exists(map_root_dir):
            print(f"Directory not found: {map_root_dir}")
            return

        os.makedirs(out_dir, exist_ok=True)

        # total_length = len(os.listdir(f"{map_root_dir}/frames"))
        total_length = 25
        clip_length = 5
        batches = generate_ranges(total_length, clip_length)

        # question_list = [
        #     f"The images I uploaded contains {clip_length} frames of images from a vehicle dash cam video. \n"
        #     f"Determine the vehicle's change in heading direction and displacement between each two consecutive frames.\n"
        #     f"That is, I want delta heading and delta distance between frame n+1 and frame n for n > 0.\n"
        #     f"In total, for the {clip_length} images I uploaded, you should be able to generate {clip_length-1} pairs "
        #     f"of direction and displacement.\n"
        #     f"For delta heading, give your answer in degree value ranging between -180 and 180 degrees (clockwise)\n"
        #     f"Here, -180 and +180 degree would mean turning around and go in the complete opposite way compared to "
        #     f"the previous frame.\n"
        #     f"For displacement, give your answer values in meter unit.\n"
        #     f"Make sure that for each frame you are calculating delta heading and displacement relative to "
        #     f"the previous frame, not cumulatively relative to the first frame.\n"
        #     f"Keep all delta headings in one list, and all displacements in another list. "
        #     f"Make sure you have {clip_length - 1} values in the direction list, and {clip_length - 1} values in the displacement list.\n"
        #     f"Give your response in json format:\n"
        #     f"```json {{"
        #     f"\"delta_heading\": [degree1, degree2, ... degree{clip_length-1}],\n"
        #     f"\"displacement\": [displacement1, displacement2, ... displacement{clip_length-1}]\n"
        #     f"}}```\n\n"
        # ]

        question_list = [
            f"The images I uploaded contains {clip_length} frames of images from a vehicle dash cam video. \n"
            f"Determine the vehicle's change in heading direction and displacement between each two consecutive frames.\n"
            f"For delta heading, give your answer in degree value ranging between -180 and 180 degrees (clockwise)\n"
            f"Here, -180 and +180 degree would mean turning around and go in the complete opposite way compared to "
            f"the previous frame.\n"
            f"For displacement, give your answer values in meter unit.\n"
            f"Make sure that for each frame you are calculating delta heading and displacement relative to "
            f"the previous frame, not cumulatively relative to the first frame.\n"
            f"Keep all delta headings in one list, and all displacements in another list. "
            f"Make sure you have {clip_length - 1} values in the direction list, and {clip_length - 1} values in the displacement list.\n"
            f"Give your response in json format:\n"
            f"```json {{"
            f"\"delta_heading\": [degree1, degree2, ... degree{clip_length - 1}],\n"
            f"\"displacement\": [displacement1, displacement2, ... displacement{clip_length - 1}]\n"
            f"}}```\n\n"
        ]


        ''' ================== Inference ================== '''
        ''' upload images to model and ask questions '''
        results, image_paths = analyze_street_view_qwen(image_directory=map_root_dir, question_list=question_list, total_length=total_length, batches=batches)

        ''' ================== save questions and answers in md and json ================== '''
        ''' the json could be used later for auto eval? '''
        if os.path.exists(f"{out_dir}/Q&A.md"):
            os.remove(f"{out_dir}/Q&A.md")
        if results:
            print("\nAnalysis Summary:")
            for question, answer in results.items():
                print(f"\n{question}")
                print(f"{answer}")
                ''' if the images were processed in one large batch '''
                if isinstance(answer, str):
                    with open(f"{out_dir}/Q&A.md", "a", encoding="utf-8") as file:
                        file.write(f"used frames {image_paths[0]} to {image_paths[-1]}\n\n")

                        file.write(f"{question}")
                        file.write(f"{answer}")
                    with open(f"{out_dir}/Q&A.json", 'w') as f:
                        json.dump(results, f, indent=4)
                    ''' if the images were processed in several small batches '''
                elif isinstance(answer, list):
                    for i in range(len(answer)):
                        os.makedirs(f"{out_dir}/batches_{clip_length}_{total_length}/batch_{batches[i]}", exist_ok=True)
                        if os.path.exists(f"{out_dir}/batches_{clip_length}_{total_length}/batch_{batches[i]}/Q&A.md"):
                            os.remove(f"{out_dir}/batches_{clip_length}_{total_length}/batch_{batches[i]}/Q&A.md")
                        ans = answer[i]
                        with open(f"{out_dir}/batches_{clip_length}_{total_length}/batch_{batches[i]}/Q&A.md", "a", encoding="utf-8") as file:
                            file.write(f"\nused frames {batches[i]}\n\n")
                            file.write(f"{question}")
                            file.write(f"{ans}")
                        with open(f"{out_dir}/batches_{clip_length}_{total_length}/batch_{batches[i]}/Q&A.json", 'w') as f:
                            json.dump({question: ans}, f, indent=4)

        ''' ================== save questions and answers in md and json ================== '''



if __name__ == "__main__":
    main()
