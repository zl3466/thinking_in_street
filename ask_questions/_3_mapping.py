import sys
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *


def check_uploaded_file(file_path):
    creation_time = os.path.getctime(file_path)
    current_time = time.time()

    # Check if the file is older than 2 days (2 * 24 * 60 * 60 seconds)
    if (current_time - creation_time) > (2 * 86400):
        os.remove(file_path)
        print(f"The images have been uploaded but already expired, removing existing filename records: {file_path}")
    else:
        print(f"The images have been uploaded and are still valid on Gemini: {file_path}")


idx2letter = ["A", "B", "C", "D"]


def main():
    data_dir = "../data/long_route_random_single"
    map_list = os.listdir(data_dir)

    map_name = "map2_soho"
    # for map_name in map_list:
    print("====================================")
    print(f"Processing Map {map_name}")

    map_root_dir = f"{data_dir}/{map_name}"
    out_dir = f"{data_dir}/{map_name}/bev_map"
    if not os.path.exists(map_root_dir):
        print(f"Directory not found: {map_root_dir}")
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    question_list = [
        f"\nThe images I uploaded are from a dash cam video footage from a vehicle. They cover most of the streets "
        f"within an area.\n"
        f"Suppose I now have a 10*10 grid that represents a BEV map of this area. "
        f"Based on the spatial information you can perceive from the footage, fill in grids with 10 landmarks you "
        f"can most confidently identify from the footage such that the BEV map correctly represents the location of "
        f"the landmarks. \n"
        f"In addition to landmarks, also identify the streets and roads you see in connecting the landmarks in the "
        f"footage and lay them on the grid as well. Note that streets and roads shouldn't be single grid; rather, "
        f"they should be represented by consecutive grids as they actually are located spatially. "
        f"The street blocks do not count into the 10 landmarks. For each street you can identify, give me all the "
        f"blocks that it occupies.\n"
        f"Make sure the blocks for the same road should be connected to each other.\n"
        f" Give your answer in json format:\n "
        f"```json {{"
        f"\"landmark\": [{{landmark_name, [r, c]}}, {{landmark_name, [r, c]}}, ...],\n"
        f"\"street\": {{street_name: [[r, c], [r, c], [r, c], ...], street_name: [[r, c], [r, c], [r, c], ...]}}\n"
        f"}}```\n"
        f"where landmark_name is the name of a landmark, "
        f"street_name is the name of a street, "
        f"and (r, c) is the coordinate of the grid with r being row index and c being column index.\n"
        f"Each landmark should have one single coordinate, and each street should have a list of coordinates."
    ]

    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    results, _ = analyze_street_view(map_root_dir, question_list, out_dir=map_root_dir)

    ''' ================== save questions and answers in md and json ================== '''
    ''' the json could be used later for auto eval? '''
    if results:
        print("\nAnalysis Summary:")
        if os.path.exists(f"{out_dir}/Q&A.md"):
            os.remove(f"{out_dir}/Q&A.md")

        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")
            with open(f"{out_dir}/Q&A.md", "a", encoding="utf-8") as file:
                file.write(f"{question}")
                file.write(f"{answer}")

        with open(f"{out_dir}/Q&A.json", 'w') as f:
            answers = list(results.values())
            json.dump(answers, f, indent=4)


if __name__ == "__main__":
    main()
