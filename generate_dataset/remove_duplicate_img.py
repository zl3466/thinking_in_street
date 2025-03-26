import os
import shutil
import imagehash
from PIL import Image
from collections import defaultdict, deque
import json

# Define paths
source_folder = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/data/long_route_random_single/map7_columbia/frames"  # Folder containing images
duplicate_folder = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/data/long_route_random_single/map7_columbia/frames_duplicate"  # Folder to move duplicates

# Define paths
json_file_path = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/data/long_route_random_single/map7_columbia/route_data.json"

# Ensure duplicate folder exists
os.makedirs(duplicate_folder, exist_ok=True)

# Load JSON data
with open(json_file_path, "r") as f:
    route_data = json.load(f)

# Extract frames
frames = route_data["frames"]

# Queue to track last 3 hashes
hash_queue = deque(maxlen=3)
updated_frames = []

# Step 1: Identify and move duplicates
for frame in frames:
    filename = frame["filename"]
    file_path = os.path.join(source_folder, filename)

    if not os.path.exists(file_path):
        print(f"Warning: {filename} not found. Skipping.")
        continue

    try:
        # Compute image hash
        img_hash = imagehash.average_hash(Image.open(file_path))

        # Check last 3 frames for duplicates
        if img_hash in hash_queue:
            # Move duplicate
            dest_path = os.path.join(duplicate_folder, filename)
            shutil.move(file_path, dest_path)
            # print(f"Moved duplicate: {file_path} → {dest_path}")

        else:
            # Store hash and keep frame
            hash_queue.append(img_hash)
            updated_frames.append(frame)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# # Step 2: Reverse the frames in the JSON (reversing the entire frame dict)
print(updated_frames[0:3])
updated_frames = updated_frames[::-1]
# print(updated_frames[0:3])

# Step 3: Generate new filenames **without conflicts**
existing_filenames = set()
temp_mapping = {}

for index, frame in enumerate(updated_frames):
    parts = frame["filename"].split("_")

    new_frame_number = f"{index:05d}"
    new_filename = f"frame_{new_frame_number}_{parts[2]}_{parts[3]}_{parts[4]}"

    source_path = os.path.join(source_folder, frame["filename"])
    final_path = os.path.join(source_folder, new_filename)

    # Rename to final name
    if os.path.exists(source_path):
        os.rename(source_path, final_path)
        print(f'renaming {frame["filename"]} to {new_filename}')

    # Update JSON filename
    frame["filename"] = new_filename
    updated_frames[index] = frame


# Step 6: Save updated JSON
updated_frames = updated_frames[::-1]
route_data["frames"] = updated_frames
with open(json_file_path, "w") as f:
    json.dump(route_data, f, indent=2)

print("Duplicate removal, JSON update, renumbering, and reverse sorting complete.")