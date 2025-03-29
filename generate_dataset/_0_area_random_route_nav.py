from collections import deque
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

seen_hashes = deque(maxlen=6)

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
street_view_resolution = "640x400"

map_dict = {
    "New York": {
        "map1_downtown_bk_new": [[40.689981, -73.983547],
                             [40.691480, -73.987365],
                             [40.694969, -73.987166],
                             [40.694847, -73.984001],
                             [40.689973, -73.981591]],
        "map2_soho": [[40.725781, -74.000906],
                      [40.724238, -73.997790],
                      [40.721791, -73.999872],
                      [40.722941, -74.002176]],
        "map3_time_square": [[40.759789, -73.987923],
                             [40.757398, -73.982247],
                             [40.753548, -73.985047],
                             [40.755979, -73.990729]],
        "map4_williamsburg": [[40.723086, -73.958733],
                              [40.718807, -73.952433],
                              [40.717360, -73.954000],
                              [40.721422, -73.960540]],
        "map5_chinatown": [[40.719086, -73.996479],
                           [40.717375, -73.991236],
                           [40.715121, -73.992582],
                           [40.716862, -73.997730]],
        "map6_east_village": [[40.730019, -73.983533],
                              [40.727108, -73.976643],
                              [40.722283, -73.980152],
                              [40.725184, -73.987054]],
        "map7_columbia": [[40.812647, -73.963045],
                          [40.810089, -73.957187],
                          [40.803999, -73.960341],
                          [40.806979, -73.967411]],
        "map8_flushing": [[40.760266, -73.835000],
                          [40.754724, -73.833031],
                          [40.758193, -73.823212],
                          [40.762980, -73.825329]],
        "map9_5th_ave": [[40.761754, -73.979080],
                         [40.759035, -73.972485],
                         [40.755255, -73.975237],
                         [40.758044, -73.981795]],
        "map10_wall_st": [[40.709314, -74.011930],
                          [40.705239, -74.004732],
                          [40.703235, -74.007993],
                          [40.704799, -74.014982]]

    }
}


# Base directories


# BASE_VIDEO_DIR = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/videos/long_route_random"

def area2grid(api_key, lat_lng_corners, grid_size, use_portion=0.2, max_num_waypoint=20):
    lat = lat_lng_corners[:, 0]
    lng = lat_lng_corners[:, 1]
    min_lat, max_lat = min(lat), max(lat)
    min_lng, max_lng = min(lng), max(lng)

    # Approximate meters to degrees conversion (varies with latitude)
    lat_step = geodesic(meters=grid_size).destination((min_lat, min_lng), 0)[0] - min_lat
    lng_step = geodesic(meters=grid_size).destination((min_lat, min_lng), 90)[1] - min_lng

    # Generate lat/lng grid
    lat_points = np.arange(min_lat, max_lat, lat_step)
    lng_points = np.arange(min_lng, max_lng, lng_step)

    # Create grid points
    grid_points = [(lat, lng) for lat in lat_points for lng in lng_points]
    # Check if Street View is available
    useful_grid_pts = []
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    for coord in tqdm(grid_points, desc=f"checking sampled grid points to keep only those available for streetview"):
        lat = coord[0]
        lng = coord[1]
        metadata_params = {
            "location": f"{lat},{lng}",
            "radius": 10,
            "source": "outdoor",
            "key": api_key
        }

        metadata_response = requests.get(metadata_url, params=metadata_params)
        if (metadata_response.status_code != 200 or
                metadata_response.json().get("status") != "OK"):
            # print(f"No Street View imagery at location: {lat}, {lng}")
            continue
        else:
            useful_grid_pts.append((lat, lng))

    # Google Maps allow max num of waypoints to be 25. we use 20 by default just to be conservative
    num_pts_wanted = int(len(useful_grid_pts) * use_portion)
    num_pts_wanted = min(max_num_waypoint, num_pts_wanted)
    print(f"{len(useful_grid_pts)} out of {len(grid_points)} grid points do have street view images, "
          f"using {num_pts_wanted} of it")
    route_pts = random.choices(useful_grid_pts, k=num_pts_wanted)

    return route_pts


def get_routes_from_pts(location_list, route_file_out="route_map.html", route_mode="driving"):
    all_positions_and_headings = []
    way_points = []

    for i in range(1, len(location_list) - 1):
        location = ", ".join(map(str, location_list[i]))
        way_points.append(location)
        # 1. Get route polyline
        # print(start, end)
    start = ", ".join(map(str, location_list[0]))
    end = ", ".join(map(str, location_list[-1]))
    print(f"route start & end: {start} | {end}, with total {len(location_list) - 2} way points in between")
    # print(f"route waypoints: {start} | {way_points} | {end}")

    overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY, mode=route_mode,
                                                       way_points=way_points)

    ''' ============= Create map visualization ============= '''
    # Decode polyline
    decoded_pts = polyline.decode(overview_poly)
    route_map = folium.Map(location=decoded_pts[0], zoom_start=7)

    # Add polyline to the map
    folium.PolyLine(decoded_pts, color="blue", weight=5, opacity=0.7).add_to(route_map)

    # save the map
    route_map.save(route_file_out)

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

    return all_positions_and_headings, distance, duration


def get_routes_from_locations(location_list, route_file_output, route_mode="driving"):
    '''

    :param location_list: list of well formatted string locations that can be used directly by GoogleMaps for navigation
    :param map_name:
    :param route_file_output: e.g. f"{BASE_DATA_DIR}/{map_name}/route_map.html"
    :return:
    '''
    all_positions_and_headings = []
    way_points = []

    for i in range(1, len(location_list) - 1):
        location = location_list[i]
        way_points.append(location)

    start = location_list[0]
    end = location_list[-1]
    print(f"route start & end: {start} | {end}, with total {len(location_list) - 2} way points in between")
    # print(f"route waypoints: {start} | {way_points} | {end}")

    overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY, mode=route_mode,
                                                       way_points=way_points)

    ''' ============= Create map visualization ============= '''
    # Decode polyline
    decoded_pts = polyline.decode(overview_poly)
    route_map = folium.Map(location=decoded_pts[0], zoom_start=7)

    # Add polyline to the map
    folium.PolyLine(decoded_pts, color="blue", weight=5, opacity=0.7).add_to(route_map)

    # save the map
    route_map.save(route_file_output)

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

    return all_positions_and_headings, distance, duration


def determine_turn_direction(old_heading, new_heading):
    """Determine if the turn is left or right"""
    diff = ((new_heading - old_heading + 180) % 360) - 180
    if abs(diff) < 10:  # threshold for forward movement
        return "forward"
    return "right" if diff > 0 else "left"


def get_directions(origin, destination, api_key, mode="driving", way_points=None):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    if way_points is not None:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "waypoints": "|".join(way_points),
            "key": api_key
        }
    else:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": api_key
        }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Directions API error: {data.get('error_message', data['status'])}")
    route = data["routes"][0]
    distance = route["legs"][0]["distance"]["text"]
    duration = route["legs"][0]["duration"]["text"]
    return route["overview_polyline"]["points"], distance, duration


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


def get_image_hash(image_data):
    """Generate a hash of the image content."""
    return hashlib.md5(image_data).hexdigest()


def fetch_street_view_image(lat, lng, heading, api_key,
                            size=street_view_resolution, pitch=0, fov=120,
                            folder="frames", seen_hashes=None,
                            frame_index=0, surround_view=False):
    """
    Fetch and save a Street View image, labeled with a reversed frame_index.
    This ensures that when frames are sorted in descending order, the earliest
    route frames appear first in the final video.
    """

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
        print(f"No Street View imagery at location: {lat}, {lng}, {metadata_response.json().get('error_message', metadata_response.json()['status'])}")
        return None

    # If imagery is available, fetch it
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    img_list = []
    if surround_view:
        heading_list = [heading - 120, heading, heading + 120]
        if not os.path.exists(f"{folder}/frames_surround"):
            os.makedirs(f"{folder}/frames_surround")
    else:
        heading_list = [heading]

    heading_name_list = ["left", "front", "right"]
    os.makedirs(f"{folder}/frames", exist_ok=True)
    for i in range(len(heading_list)):
        heading_i = heading_list[i]
        heading_name = heading_name_list[i]

        if surround_view and heading_name != "front":
            os.makedirs(f"{folder}/frames_{heading_name}", exist_ok=True)

        params = {
            "size": size,
            "location": f"{lat},{lng}",
            "heading": heading_i,
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
            seen_hashes.append(image_hash)

            # Name the file using reverse index
            # Example: frame_0099_40.74844_-73.98401_0.0.jpg
            # The leading zeros in index ensure correct alphabetical sorting if needed
            if surround_view:
                if heading_name == "front":
                    filename = (f"{folder}/frames/frame_{frame_index:05d}_"
                                f"{lat:.5f}_{lng:.5f}_{heading_i:.1f}.jpg")
                else:
                    filename = (f"{folder}/frames_{heading_name}/frame_{frame_index:05d}_"
                                f"{lat:.5f}_{lng:.5f}_{heading_i:.1f}.jpg")
            else:
                filename = (f"{folder}/frames/frame_{frame_index:05d}_"
                            f"{lat:.5f}_{lng:.5f}_{heading_i:.1f}.jpg")
            if surround_view:
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                img_list.append(image)
            with open(filename, 'wb') as f:
                f.write(image_data)
        else:
            print(f"Failed to fetch Street View: {response.status_code}")
            return None
    true_heading_filename = (f"{folder}/frames/frame_{frame_index:05d}_"
                             f"{lat:.5f}_{lng:.5f}_{heading:.1f}.jpg")

    if surround_view:
        true_heading_filename = (f"{folder}/frames_surround/frame_{frame_index:05d}_"
                                 f"{lat:.5f}_{lng:.5f}_{heading:.1f}.jpg")
        surround_view_img = np.concatenate(img_list, axis=1)
        cv2.imwrite(true_heading_filename, surround_view_img)
    return true_heading_filename


def main():
    # Create base directories if they don't exist
    ''' ============= choose map ============= '''
    # print(map_order_dict.keys())
    # map_name = "map1_downtown_bk"
    # map_name = "map6_east_village"
    city_name = "New York"
    for map_name in ["map1_downtown_bk_new"]:
        # TODO: manually specify here if you want to generate surround-view image or singular image
        surround = False

        if surround:
            BASE_DATA_DIR = "../data/long_route_random_surround"
        else:
            BASE_DATA_DIR = "../data/long_route_random_single"

        # for map_name in map_dict.keys():

        area_corner_list = map_dict[city_name][map_name]
        print(f"processing map {map_name}")

        # grid coord params
        grid_size = 30
        data_dir = f"{BASE_DATA_DIR}/{map_name}"
        os.makedirs(data_dir, exist_ok=True)

        ''' ============= get route ============= '''
        area_corner_arr = np.array(area_corner_list)
        area_grid_pts = area2grid(GOOGLE_MAPS_API_KEY, area_corner_arr, grid_size, use_portion=0.2, max_num_waypoint=16)
        all_positions_and_headings, distance, duration = get_routes_from_pts(area_grid_pts,
                                                                             route_file_out=f"{BASE_DATA_DIR}/{map_name}/route_map.html")

        # save used route points
        m = folium.Map(location=area_grid_pts[0], zoom_start=5, tiles="OpenStreetMap")
        for [lat, lng] in area_grid_pts:
            folium.Marker(location=[lat, lng], popup=f"({lat}, {lng})").add_to(m)
        m.save(f"{data_dir}/grid_map.html")

        ''' ============= fetch street view images along route ============= '''
        # Initialize data storage
        route_data = {
            "map": map_name,
            "frames": [],
        }

        frame_files = []
        total_frames = len(all_positions_and_headings)

        for idx, (lat, lng, heading, movement) in tqdm(enumerate(all_positions_and_headings),
                                                       total=len(all_positions_and_headings)):
            reverse_idx = total_frames - 1 - idx
            fname = fetch_street_view_image(
                lat, lng, heading,
                api_key=GOOGLE_MAPS_API_KEY,
                seen_hashes=seen_hashes,
                folder=data_dir,
                frame_index=reverse_idx,
                surround_view=surround
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

        ''' ============= generate street view video ============= '''
        if frame_files:
            video_file = f"{data_dir}/{map_name}.mp4"
            # video_file = os.path.join(BASE_DATA_DIR, f"{location_folder}.mp4")
            if surround:
                success = generate_video_from_img(f"{data_dir}/frames_surround", video_file, frame_rate=10,
                                                  reverse=False)
            else:
                success = generate_video_from_img(f"{data_dir}/frames", video_file, frame_rate=10, reverse=False)

            if success:
                print(f"Video generation complete. Check {video_file}")
            else:
                print("Failed to generate video")

        # save a list of locations that the model can see from the footage for upcoming tasks
        question_list = ["The images I uploaded are from a dash cam video footage from a vehicle. They cover most of "
                         "the streets within an area.\n"
                         " Give me 10 landmarks you can confidently see in this footage"]
        results = analyze_street_view(data_dir, question_list, out_dir=data_dir)
        if results:
            with open(f"{data_dir}/seen_landmarks.json", 'w') as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
