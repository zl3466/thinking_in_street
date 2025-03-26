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


def analyze_street_view(image_directory):
    """Analyze street view images in a directory and generate spatial data"""
    # Get and sort image paths from frames directory
    out_dir = image_directory
    frames_dir = os.path.join(image_directory, "frames/cat")
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

        questions = [
            "Can you see a chase bank, a jordan retail shop, and a brooklyn cancer center in this area?",

            "As you can find a chase bank, a jordan retail shop, and a brooklyn cancer center in this area, "
            "I need to visit all of these three places. "
            "My uploaded images should have covered all the streets within this area."
            "pay attention to the same buildings appearing in different images to get a sense of how the streets are organized."
            "can you give me a description of the route that connects all these three places? "
            "e.g. walk toward which direction, turn to which direction, etc.",

            "As you can find a chase bank, a jordan retail shop, and a brooklyn cancer center in this area, "
            "I need to visit all of these three places."
            "My uploaded images should have covered all the streets within this area."
            "pay attention to the same buildings appearing in different images to get a sense of how the streets are organized."
            "can you give me a description of the route that connects all these three places? "
            "e.g. walk toward which direction, turn to which direction, etc."
            "What would be the optimal (fastest) order of visiting these three places? what would be the route for that?"

            "can you draw me a map of how the streets covered by my uploaded images look like from a bird's eye view?"
        ]

        results = {}
        for question in questions:
            response = model.analyze_images(image_paths, question, out_dir=out_dir)
            # response = model.analyze_images_long(image_paths, question)

            if response:
                print(f"\nQ: {question}")
                print(f"A: {response}")
                results[question] = response

        return results

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


def main():
    image_directory = "../data/long_grid/map_2_downtownbk_test"
    if not os.path.exists(image_directory):
        print(f"Directory not found: {image_directory}")
        return

    results = analyze_street_view(image_directory)
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")


if __name__ == "__main__":
    main()
