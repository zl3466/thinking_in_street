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
        total_length = 30
        clip_length = 3
        batches = generate_ranges(total_length, clip_length)

        question_list = [
            f"The images I uploaded contains {clip_length} frames of images from a vehicle dash cam video. \n"
            f"Determine the vehicle's movement direction (delta heading) and displacement (delta distance) "
            f"between each frame. That is, for the {clip_length} I uploaded, "
            f"you should be able to generate {clip_length-1} pairs of direction and displacement.\n"
            f"For delta heading, assume the vehicle's starting position in the first frame is heading 0 degree."
            f"You should then represent all later delta headings in degree values ranging between -180 and 180 degrees"
            f"(clockwise) relative to the starting position at 0 degree.\n"
            f"Here, -180 and +180 degree would mean turning around and go in the complete opposite way compared to "
            f"the previous frame.\n"
            f"For displacement, give your answer values in meter unit.\n"
            f"Finally, present your response in json format:\n"
            f"```json {{"
            f"\"delta_heading\": [degree1, ... degree{clip_length-1}],\n"
            f"\"displacement\": [displacement1, ... displacement{clip_length-1}]\n"
            f"}}```\n"
            f"Make sure you have {clip_length-1} int or float elements in both the direction list and displacement list.\n"
            f"The delta headings and displacements can't be all the same\n\n"
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
                        os.makedirs(f"{out_dir}/batches/batch_{batches[i]}", exist_ok=True)
                        if os.path.exists(f"{out_dir}/batches/batch_{batches[i]}/Q&A.md"):
                            os.remove(f"{out_dir}/batches/batch_{batches[i]}/Q&A.md")
                        ans = answer[i]
                        with open(f"{out_dir}/batches/batch_{batches[i]}/Q&A.md", "a", encoding="utf-8") as file:
                            file.write(f"\nused frames {batches[i]}\n\n")
                            file.write(f"{question}")
                            file.write(f"{ans}")
                        with open(f"{out_dir}/batches/batch_{batches[i]}/Q&A.json", 'w') as f:
                            json.dump({question: ans}, f, indent=4)

        ''' ================== save questions and answers in md and json ================== '''



if __name__ == "__main__":
    main()
