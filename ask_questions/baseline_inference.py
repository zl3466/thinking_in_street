import json
import os
import re
from model.gemini import GeminiModel
from model.gpt4v import GPT4VModel
from model.claude import ClaudeModel
from config import LLM_PROVIDER

def get_frame_number(filename):
    """Extract frame number from filename"""
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else 0

def get_model():
    """Get the appropriate model based on config"""
    if LLM_PROVIDER == "gemini":
        return GeminiModel()
    elif LLM_PROVIDER == "openai":
        return GPT4VModel()
    elif LLM_PROVIDER == "anthropic":
        return ClaudeModel()
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def analyze_street_view(image_directory, question_list, out_dir):
    """Analyze street view images in a directory and generate spatial data"""
    # Get and sort image paths from frames directory
    frames_dir = os.path.join(image_directory, "frames")
    route_json = f"{image_directory}/route_data.json"
    route_info = json.load(open(route_json))

    # frames = route_info["frames"]
    # coord_dir = {}
    # coord = {}
    # direction = {}
    # for frame in frames:
    #     frame_idx = frame["filename"].split("_")[1]
    #     coord_dir[frame_idx] = {"coordinate": frame["coordinates"], "direction": frame["heading"]}
    #     coord[frame_idx] = frame["coordinates"]
    #     direction[frame_idx] = frame["heading"]
    #
    # sorted_coord_dir = dict(sorted(coord_dir.items(), reverse=True))
    # sorted_coord = dict(sorted(coord.items(), reverse=True))
    # sorted_direction = dict(sorted(direction.items(), reverse=True))
    # # print(sorted_coordinates)

    if os.path.exists(frames_dir):
        image_directory = frames_dir
    
    # Get all image paths
    image_paths = []
    for image_file in os.listdir(image_directory):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_directory, image_file)
            image_paths.append(full_path)
    
    if not image_paths:
        print(f"No images found in {image_directory}!")
        return None
        
    # Sort by frame number in reverse order
    image_paths = sorted(image_paths, key=lambda x: get_frame_number(os.path.basename(x)), reverse=True)
    print(f"\nFound {len(image_paths)} images in {image_directory}")


    try:
        # Initialize model based on config
        model = get_model()
        results = model.analyze_images(image_paths, question_list, out_dir)
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

def main():
    data_dir = "../data/long_route_random_single"
    question_dir = f"../ask_questions/multi_stop"
    map_list = os.listdir(data_dir)

    # for map_name in map_list:
    #     img_dir = f"{data_dir}/{map_name}"
    #     if not os.path.exists(img_dir):
    #         print(f"Directory not found: {img_dir}")
    #         return
    #     question_list = json.load(open(f"{question_dir}/{map_name}"))
    #     results = analyze_street_view(img_dir, question_list)
    #     if results:
    #         print("\nAnalysis Summary:")
    #         for question, answer in results.items():
    #             print(f"\n{question}")
    #             print(f"{answer}")

    map_name = "map1_downtown_bk"
    map_root_dir = f"{data_dir}/{map_name}"
    if not os.path.exists(map_root_dir):
        print(f"Directory not found: {map_root_dir}")
        return
    question_list = json.load(open(f"{question_dir}/{map_name}.json"))
    results = analyze_street_view(map_root_dir, question_list, out_dir=map_root_dir)
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")
    with open(f"{data_dir}/{map_name}/Q&A.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
