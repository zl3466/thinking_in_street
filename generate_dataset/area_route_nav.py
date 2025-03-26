import os
import math
import requests
import polyline
from dotenv import load_dotenv
import hashlib
from PIL import Image
import io
import json
from pathlib import Path
import cv2
from tqdm import tqdm

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# START_LOCATION = "Nami Nori Williamsburg, 236 N 12th St, Brooklyn, NY 11211"
# END_LOCATION = "Rule of Thirds, 171 Banker St, Brooklyn, NY 11222"

map1_intersection_list = [[40.725777, -74.000906],
                          [40.725392, -74.000103],
                          [40.725017, -73.999332],
                          [40.724648, -73.998594],
                          [40.724246, -73.997795],
                          [40.724542, -74.001930],
                          [40.724171, -74.001130],
                          [40.723806, -74.000375],
                          [40.723438, -73.999605],
                          [40.723044, -73.998824],
                          [40.723315, -74.002947],
                          [40.722945, -74.002185],
                          [40.722576, -74.001427],
                          [40.722195, -74.000661],
                          [40.721799, -73.999873],
                          [40.722392, -74.003756],
                          [40.722014, -74.002984],
                          [40.721644, -74.002217],
                          [40.721266, -74.001450],
                          [40.720844, -74.000653]]

map2_intersection_list = [[40.693677, -73.984253],
                          [40.693658, -73.983309],
                          [40.693600, -73.981758],
                          [40.692144, -73.984361],
                          [40.692100, -73.983398],
                          [40.692057, -73.982537],
                          [40.692042, -73.981790],
                          [40.691292, -73.982113],
                          [40.690301, -73.984431],
                          [40.689953, -73.983564],
                          [40.689721, -73.982939],
                          [40.689972, -73.981489],
                          [40.688936, -73.980965]]

map_list_dict = {"map_1_soho": map1_intersection_list,
                 "map_2_downtownbk": map2_intersection_list}

map_order_dict = {"map_1_soho": [4, 0, 15, 19, 14, 10, 5, 9, 4, 3, 18, 19, 4, 2, 17, 19, 4, 1, 16],
                 "map_2_downtownbk": [12, 2, 0, 8, 9, 4, 3, 8, 11, 10, 12, 1, 5, 7, 11]}

map_question_dict = {"map_2_downtownbk": "jordan"}

# Base directories
BASE_DATA_DIR = "C:/Users/ROG_ZL/Documents/github/Thinking_in_Street/data/long_route"
BASE_VIDEO_DIR = "C:/Users/ROG_ZL/Documents/github/Thinking_in_Street/videos/long_route"


def get_routes_from_intersection(intersection_list, intersection_order, map_name):
    all_positions_and_headings = []
    way_points = []

    for i in range(1, len(intersection_order) - 1):
        idx = intersection_order[i]
        location = ", ".join(map(str, intersection_list[idx]))
        way_points.append(location)
        # 1. Get route polyline
        # print(start, end)
    start = ", ".join(map(str, intersection_list[intersection_order[0]]))
    end = ", ".join(map(str, intersection_list[intersection_order[-1]]))
    print(f"route waypoints: {start} | {way_points} | {end}")
    overview_poly = get_directions(start, end, GOOGLE_MAPS_API_KEY, way_points=way_points)

    # Create map visualization (optional)
    create_map_visualization(overview_poly, output_file=f"{BASE_DATA_DIR}/{map_name}/route/route_map_{map_name}.html")

    # 2. Decode polyline
    decoded_pts = polyline.decode(overview_poly)

    # 3. Interpolate additional points
    sampled_points = []
    for i in range(len(decoded_pts) - 1):
        interpolated = interpolate_points(
            decoded_pts[i][0], decoded_pts[i][1],
            decoded_pts[i + 1][0], decoded_pts[i + 1][1]
        )
        sampled_points.extend(interpolated[:-1])
    sampled_points.append(decoded_pts[-1])

    # 4. Compute headings
    headings = []
    for i in range(len(sampled_points) - 1):
        lat1, lng1 = sampled_points[i]
        lat2, lng2 = sampled_points[i + 1]
        b = compute_bearing(lat1, lng1, lat2, lng2)
        headings.append(b)

    # 5. Build list of positions and headings with movement tracking

    for i in range(len(sampled_points) - 1):
        lat_cur, lng_cur = sampled_points[i]
        old_heading = headings[i]

        # Add straight movement
        all_positions_and_headings.append((lat_cur, lng_cur, old_heading, "forward"))

        # If there's a turn coming
        if i < len(headings) - 1:
            new_heading = headings[i + 1]
            turn_direction = determine_turn_direction(old_heading, new_heading)
            if turn_direction != "forward":
                turn_seq = generate_turn_headings(old_heading, new_heading, step=12)
                lat_turn, lng_turn = sampled_points[i + 1]
                for h in turn_seq[1:]:
                    all_positions_and_headings.append((lat_turn, lng_turn, h, turn_direction))

    # for i in range(len(intersection_order) - 1):
    #     start_idx = intersection_order[i]
    #     end_idx = intersection_order[i+1]
    #     start = ", ".join(map(str, intersection_list[start_idx]))
    #     end = ", ".join(map(str, intersection_list[end_idx]))
    #
    #     # 1. Get route polyline
    #     # print(start, end)
    #     overview_poly = get_directions(start, end, GOOGLE_MAPS_API_KEY)
    #
    #     # Create map visualization (optional)
    #     create_map_visualization(overview_poly, output_file=f"{BASE_DATA_DIR}/{map_name}/route/route_map_{i}.html")
    #
    #     # 2. Decode polyline
    #     decoded_pts = polyline.decode(overview_poly)
    #
    #     # 3. Interpolate additional points
    #     sampled_points = []
    #     for i in range(len(decoded_pts) - 1):
    #         interpolated = interpolate_points(
    #             decoded_pts[i][0], decoded_pts[i][1],
    #             decoded_pts[i + 1][0], decoded_pts[i + 1][1]
    #         )
    #         sampled_points.extend(interpolated[:-1])
    #     sampled_points.append(decoded_pts[-1])
    #
    #     # 4. Compute headings
    #     headings = []
    #     for i in range(len(sampled_points) - 1):
    #         lat1, lng1 = sampled_points[i]
    #         lat2, lng2 = sampled_points[i + 1]
    #         b = compute_bearing(lat1, lng1, lat2, lng2)
    #         headings.append(b)
    #
    #     # 5. Build list of positions and headings with movement tracking
    #
    #     for i in range(len(sampled_points) - 1):
    #         lat_cur, lng_cur = sampled_points[i]
    #         old_heading = headings[i]
    #
    #         # Add straight movement
    #         all_positions_and_headings.append((lat_cur, lng_cur, old_heading, "forward"))
    #
    #         # If there's a turn coming
    #         if i < len(headings) - 1:
    #             new_heading = headings[i + 1]
    #             turn_direction = determine_turn_direction(old_heading, new_heading)
    #             if turn_direction != "forward":
    #                 turn_seq = generate_turn_headings(old_heading, new_heading, step=12)
    #                 lat_turn, lng_turn = sampled_points[i + 1]
    #                 for h in turn_seq[1:]:
    #                     all_positions_and_headings.append((lat_turn, lng_turn, h, turn_direction))
    return all_positions_and_headings


def get_location_folder_name(location):
    """Extract first few words from location for folder name"""
    words = location.split()[:3]  # Take first 3 words
    return '_'.join(words).lower().replace(',', '')


def determine_turn_direction(old_heading, new_heading):
    """Determine if the turn is left or right"""
    diff = ((new_heading - old_heading + 180) % 360) - 180
    if abs(diff) < 10:  # threshold for forward movement
        return "forward"
    return "right" if diff > 0 else "left"


def get_directions(origin, destination, api_key, way_points=None):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    if way_points is not None:
        params = {
            "origin": origin,
            "destination": destination,
            "waypoints": "|".join(way_points),
            "key": api_key
        }
    else:
        params = {
            "origin": origin,
            "destination": destination,
            "key": api_key
        }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Directions API error: {data.get('error_message', data['status'])}")
    route = data["routes"][0]
    return route["overview_polyline"]["points"]


def compute_bearing(lat1, lng1, lat2, lng2):
    lat1_r, lng1_r = map(math.radians, [lat1, lng1])
    lat2_r, lng2_r = map(math.radians, [lat2, lng2])
    d_lng = lng2_r - lng1_r
    x = math.cos(lat2_r) * math.sin(d_lng)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lng))
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def interpolate_points(lat1, lng1, lat2, lng2, step=0.0001):
    """Interpolate points between two coordinates using a fixed step size."""
    points = []
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    distance = math.sqrt(dlat ** 2 + dlng ** 2)
    num_steps = max(1, int(distance / step))

    for i in range(num_steps + 1):
        fraction = i / num_steps
        lat = lat1 + fraction * dlat
        lng = lng1 + fraction * dlng
        points.append((lat, lng))
    return points


def generate_turn_headings(old_heading, new_heading, step=30):
    """Generate a series of intermediate heading angles between old_heading and new_heading."""
    turn_frames = []
    old_h = old_heading % 360
    new_h = new_heading % 360

    # Calculate the shortest turning direction
    diff = ((new_h - old_h + 180) % 360) - 180
    abs_diff = abs(diff)

    # Number of intermediate steps
    num_steps = int(abs_diff / step)

    for i in range(num_steps + 1):
        fraction = i / num_steps if num_steps > 0 else 1
        current_h = (old_h + diff * fraction) % 360
        turn_frames.append(current_h)

    # Ensure the final heading is included
    if not turn_frames or abs(turn_frames[-1] - new_h) > 0.1:
        turn_frames.append(new_h)

    return turn_frames


def generate_video_from_img(folder_dir, out_dir, frame_rate=5, reverse=True):
    if not os.path.exists(folder_dir) or len(os.listdir(folder_dir)) == 0:
        print(f"input image folder {folder_dir} is empty or does not exist")
        return False
    img_list = os.listdir(folder_dir)

    img_list = sorted(img_list, reverse=(not reverse))
    sample_img = cv2.imread(f"{folder_dir}/{img_list[0]}")
    sample_img = cv2.resize(sample_img, dsize=(sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    video_width = sample_img.shape[1]
    video_height = sample_img.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out0 = cv2.VideoWriter(out_dir, fourcc, frame_rate, (video_width, video_height))

    for i in tqdm(range(len(img_list)), desc=f"{out_dir}"):
        img_filename = img_list[i]
        img_dir = f"{folder_dir}/{img_filename}"
        img = cv2.imread(img_dir)
        img = cv2.resize(img, dsize=(sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        out0.write(img)
    return True


def generate_video(frames_dir, output_file="output.mp4", framerate=10, frame_duration=1):
    """
    Generate a video from a directory of frames using ffmpeg.
    
    Args:
        frames_dir: Directory containing the frame images
        output_file: Output video file path
        framerate: Frames per second in the output video
        frame_duration: How many times each frame should be repeated (in seconds)
    """
    import subprocess
    import glob

    frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if not frames:
        print("No frames found in directory:", frames_dir)
        return False

    # Sort frames and then reverse them
    frames.sort(reverse=True)

    # Create a temporary file with the list of frames
    with open("frames_list.txt", "w") as f:
        for frame in frames:
            # Repeat each frame multiple times based on duration
            for _ in range(int(frame_duration * framerate)):
                f.write(f"file '{frame}'\n")

    generate_video_from_img(frames_dir, output_file)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", "frames_list.txt",
            "-framerate", str(framerate),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_file
        ]
        print(f"trying to generate video to {output_file}...\n {cmd}")
        subprocess.run(cmd, check=True)
        print(f"Video generated successfully: {output_file}")

        # Clean up
        os.remove("frames_list.txt")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
        if os.path.exists("frames_list.txt"):
            os.remove("frames_list.txt")
        return False


def get_image_hash(image_data):
    """Generate a hash of the image content."""
    return hashlib.md5(image_data).hexdigest()


def fetch_street_view_image(lat, lng, heading, api_key,
                            size="640x400", pitch=0, fov=120,
                            folder="frames", seen_hashes=None,
                            frame_index=0):
    """
    Fetch and save a Street View image, labeled with a reversed frame_index.
    This ensures that when frames are sorted in descending order, the earliest
    route frames appear first in the final video.
    """
    if seen_hashes is None:
        seen_hashes = set()

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Check if Street View is available
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    metadata_params = {
        "location": f"{lat},{lng}",
        "radius": 10,
        "source": "outdoor",
        "key": api_key
    }

    metadata_response = requests.get(metadata_url, params=metadata_params)
    if (metadata_response.status_code != 200 or
            metadata_response.json().get("status") != "OK"):
        print(f"No Street View imagery at location: {lat}, {lng}")
        return None

    # If imagery is available, fetch it
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "source": "outdoor",
        "key": api_key
    }

    response = requests.get(base_url, params=params, stream=True)
    if response.status_code == 200:
        image_data = response.content
        image_hash = get_image_hash(image_data)

        # Check duplicates
        if image_hash in seen_hashes:
            print(f"Skipping duplicate image at: {lat}, {lng}, heading={heading}")
            return None

        seen_hashes.add(image_hash)

        # Name the file using reverse index
        # Example: frame_0099_40.74844_-73.98401_0.0.jpg
        # The leading zeros in index ensure correct alphabetical sorting if needed
        filename = (f"{folder}/frame_{frame_index:05d}_"
                    f"{lat:.5f}_{lng:.5f}_{heading:.1f}.jpg")

        with open(filename, 'wb') as f:
            f.write(image_data)
        return filename
    else:
        print(f"Failed to fetch Street View: {response.status_code}")
        return None


def create_map_visualization(overview_poly, output_file="route_map.html"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Visualization</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&libraries=geometry"></script>
        <style>
            #map {{
                height: 600px;
                width: 100%;
                margin: 20px auto;
            }}
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Route Visualization</h2>
        <div id="map"></div>
        <script>
            function initMap() {{
                const map = new google.maps.Map(document.getElementById('map'), {{
                    zoom: 14,
                    center: {{ lat: 40.7589, lng: -73.9851 }},
                    mapTypeId: google.maps.MapTypeId.ROADMAP
                }});

                const path = '{overview_poly}';
                const decodedPath = google.maps.geometry.encoding.decodePath(path);
                
                const routePath = new google.maps.Polyline({{
                    path: decodedPath,
                    geodesic: true,
                    strokeColor: '#FF0000',
                    strokeOpacity: 1.0,
                    strokeWeight: 4
                }});

                routePath.setMap(map);

                // Markers for start and end
                const startMarker = new google.maps.Marker({{
                    position: decodedPath[0],
                    map: map,
                    title: 'Start',
                    label: 'S'
                }});

                const endMarker = new google.maps.Marker({{
                    position: decodedPath[decodedPath.length - 1],
                    map: map,
                    title: 'End',
                    label: 'E'
                }});

                // Fit map bounds
                const bounds = new google.maps.LatLngBounds();
                decodedPath.forEach(point => bounds.extend(point));
                map.fitBounds(bounds);
            }}
            window.onload = initMap;
        </script>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Map visualization created: {output_file}")


def main():
    # Create base directories if they don't exist
    ''' show available maps '''
    # print(map_order_dict.keys())
    map_name = "map_2_downtownbk"
    map_intersection_order = map_order_dict[map_name]
    map_intersection_list = map_list_dict[map_name]

    location_folder = get_location_folder_name(map_name)
    data_dir = os.path.join(BASE_DATA_DIR, location_folder)
    frames_dir = os.path.join(data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(BASE_VIDEO_DIR, exist_ok=True)

    # Initialize data storage
    route_data = {
        "map": map_name,
        "frames": [],
        "movements": []
    }

    all_positions_and_headings = get_routes_from_intersection(map_intersection_list, map_intersection_order,
                                                              map_name=map_name)

    # 6. Fetch Street View images
    frame_files = []
    seen_hashes = set()
    total_frames = len(all_positions_and_headings)

    for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                   total=len(all_positions_and_headings)):
        reverse_idx = total_frames - 1 - idx
        fname = fetch_street_view_image(
            lat, lng, heading,
            api_key=GOOGLE_MAPS_API_KEY,
            seen_hashes=seen_hashes,
            folder=frames_dir,
            frame_index=reverse_idx
        )
        if fname:
            frame_files.append(fname)
            # Store frame data with the movement from all_positions_and_headings
            route_data["frames"].append({
                "filename": os.path.basename(fname),
                "coordinates": {"lat": lat, "lng": lng},
                "heading": heading,
                "movement": movement  # This now correctly uses the movement we determined earlier
            })

    print("Downloaded frames:", len(frame_files))

    # Save route data to JSON
    route_data_file = os.path.join(data_dir, "route_data.json")
    with open(route_data_file, 'w') as f:
        json.dump(route_data, f, indent=2)

    # Generate video
    if frame_files:
        video_file = os.path.join(BASE_VIDEO_DIR, f"{location_folder}.mp4")
        success = generate_video_from_img(frames_dir, video_file, frame_rate=5, reverse=False)
        # success = generate_video(frames_dir, output_file=video_file, framerate=5)
        if success:
            print(f"Video generation complete. Check {video_file}")
        else:
            print("Failed to generate video")


if __name__ == "__main__":
    main()
