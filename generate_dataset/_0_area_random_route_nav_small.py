from collections import deque
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *
from utils.qwen_utils import *

seen_hashes = deque(maxlen=6)

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
street_view_resolution = "640x400"

map_dict = {
    "New York": {
        # "map1_downtown_bk": ["McDonald's, 395 Flatbush Ave Ext, Brooklyn, NY 11201",
        #                      "TD Bank, 2-4 Flatbush Ave, Brooklyn, NY 11217",
        #                      "30 Flatbush Ave, Brooklyn, NY 11217",
        #                      "Mark Morris Dance Group, 3 Lafayette Ave, Brooklyn, NY 11217",
        #                      "66 Rockwell, 66 Rockwell Pl, Brooklyn, NY 11217",
        #                      "McDonald's, 395 Flatbush Ave Ext, Brooklyn, NY 11201",
        #                      "Popeyes Louisiana Kitchen, 25 Flatbush Ave, Brooklyn, NY 11217"],
        # "map2_soho": ["Bulthaup, 158 Wooster St, New York, NY 10012",
        #               "Wolford Boutique - SoHo, 124 Prince St, New York, NY 10012",
        #               "Polo Ralph Lauren, 109 Prince St, New York, NY 10012",
        #               "RIMOWA, 99 Prince St, New York, NY 10012",
        #               "Versani, 171 Mercer St, New York, NY 10012",
        #               "Shopify NY, 131 Greene St, New York, NY 10012",
        #               "Apple SoHo, 103 Prince St, New York, NY 10012",
        #               "Van Leeuwen Ice Cream, 61 W Houston St, New York, NY 10012",
        #               "D&C Italia, 147 Wooster St, New York, NY 10012"],
        # "map3_time_square": ["M&M'S New York, 1600 Broadway, New York, NY 10019",
        #                      "TKTS Times Square, Broadway at, W 47th St, New York, NY 10036",
        #                      "Hotel Riu Plaza Manhattan Times Square, 145 W 47th St, New York, NY 10036",
        #                      "Sanctuary Hotel New York, 132 W 47th St, New York, NY 10036",
        #                      "20th Century Fox Credit Union, 1211 6th Ave #3, New York, NY 10036",
        #                      "Buck Parson - Chase Home Lending Advisor - NMLS ID 487168, 1230 Ave of the Americas, New York, NY 10036",
        #                      "Fox News, 1211 6th Ave, New York, NY 10036",
        #                      "James Earl Jones Theatre, 138 W 48th St, New York, NY 10036",
        #                      "TKTS Times Square, Broadway at, W 47th St, New York, NY 10036"],
        # "map4_williamsburg": [[40.723086, -73.958733],
        #                       [40.718807, -73.952433],
        #                       [40.717360, -73.954000],
        #                       [40.721422, -73.960540]],
        # "map5_chinatown": ["Chase Bank, 231 Grand St, New York, NY 10013",
        #                    "C.T. Seafood Marts, 249 Grand St, New York, NY 10002",
        #                    "Kung Fu Tea, 73 Chrystie St, New York, NY 10002",
        #                    "Wyndham Garden Chinatown, 93 Bowery, New York, NY 10002",
        #                    "HEYTEA (Grand St), 240 Grand St, New York, NY 10002",
        #                    "Bank of America Financial Center, 88 Bowery, New York, NY 10013"
        #                    "Tofu Tofu, 96 Bowery, New York, NY 10013",
        #                    "Hong Kong Supermarket, 157 Hester St, New York, NY 10013",
        #                    "Heng Xing Grocery, 95 Elizabeth St, New York, NY 10013",
        #                    "Chase Bank, 231 Grand St, New York, NY 10013"],
        # "map6_east_village": [[40.730019, -73.983533],
        #                       [40.727108, -73.976643],
        #                       [40.722283, -73.980152],
        #                       [40.725184, -73.987054]],
        # "map7_columbia": ["Citibank ATM, 101 Altschul Hall, 3009 Broadway Suite 103, New York, NY 10027",
        #                   "Barnard College, 3009 Broadway, New York, NY 10027",
        #                   "Thompson Hall, 553-587 W 120th St, New York, NY 10027",
        #                   "Hartley Pharmacy, 1219 Amsterdam Ave, New York, NY 10027",
        #                   "Morningside Campus, 1150 Amsterdam Ave, New York, NY 10027",
        #                   "Shake Shack Morningside Heights, 2957 Broadway, New York, NY 10025",
        #                   "Alma Mater, 2970 Broadway, New York, NY 10027",
        #                   "Department Of Chemistry, 3000 Broadway, New York, NY 10027"],
        # "map8_flushing": ["Chase Bank, 39-01 Main St, Flushing, NY 11354",
        #                   "Nan Xiang Soup Dumplings - Flushing, 39-16 Prince St #104, Flushing, NY 11354",
        #                   "East West Bank, 135-11 Roosevelt Ave, Flushing, NY 11354",
        #                   "Citi, 3817 Main St, Flushing, NY 11354",
        #                   "Chase Bank, 3724 Main St, Flushing, NY 11354",
        #                   "Bake Culture, 38-04 Prince St, Flushing, NY 11354",
        #                   "135-33-135-49 39th Ave, Flushing, NY 11354",
        #                   "Chase Bank, 39-01 Main St, Flushing, NY 11354"],
        # "map9_5th_ave": [
        #     "Victoria's Secret & PINK by Victoria's Secret, 640 5th Ave, New York, NY 10019",
        #     "Banana Republic, 626 5th Ave, New York, NY 10111",
        #     "Michael Kors, 610 5th Ave, New York, NY 10020",
        #     "FAO Schwarz, 30 Rockefeller Plaza, New York, NY 10111",
        #     "Pebble Bar, 67 W 49th St, New York, NY 10112",
        #     "% Arabica New York 30 Rock, 30 Rockefeller Plaza, New York, NY 10112",
        #     "Banana Republic, 626 5th Ave, New York, NY 10111",
        #     "The LEGO® Store Fifth Avenue, 636 5th Ave, New York, NY 10020",
        #     "American Friends of Covent Garden, 40 W 51st St, New York, NY 10020",
        #     "% Arabica New York 30 Rock, 30 Rockefeller Plaza, New York, NY 10112"],
        # "map10_wall_st": ["Trinity Church, 89 Broadway, New York, NY 10006",
        #                   "ALLCITY Extended Stay, 71 Broadway, New York, NY 10006",
        #                   "MTM, 64 Trinity Pl, New York, NY 10006",
        #                   "Leadership & Public Service High School, 90 Trinity Pl, New York, NY 10006",
        #                   "Citi, 120 Broadway, New York, NY 10271",
        #                   "TD Bank, 2 Wall St, New York, NY 10005",
        #                   "Whole Foods Market Floral, 66 Broadway, New York, NY 10005"],
        # "map11_wtc": ["PNC Bank, Fulton Bldg, 200 Broadway, New York, NY 10038",
        #               "Double Check by John Seward Johnson II, New York, NY 10006",
        #               "Chopt Creative Salad Co., 1 Liberty St, New York, NY 10006",
        #               "Burger King, 106 Liberty St, New York, NY 10006",
        #               "Auntie Anne's, 44 Church St, New York, NY 10007",
        #               "Anthropologie, 195 Broadway, New York, NY 10007",
        #               "Convene One Liberty Plaza, 1 Liberty St, New York, NY 10006",
        #               "Le Cafe Coffee, 28 Cortlandt St, New York, NY 10006"],
        # "map12_jay_st": ["Dallas BBQ, 180 Livingston St, Brooklyn, NY 11201",
        #                  "Duane Reade, 386 Fulton St, Brooklyn, NY 11201",
        #                  "Appellate Supreme Court Clerk, 141 Livingston St # 15, Brooklyn, NY 11201",
        #                  "Cardtronics ATM, 422 Fulton St, Brooklyn, NY 11201",
        #                  "adidas Store Brooklyn, 454 Fulton St, Brooklyn, NY 11201",
        #                  "Banana Republic Factory Store, 485 Fulton St, Brooklyn, NY 11201",
        #                  "Raising Cane's Chicken Fingers, 447 Fulton St, Brooklyn, NY 11201",
        #                  "Foot Locker, 408 Fulton St, Brooklyn, NY 11201"],
        # "map13_7th_ave": ["Bar Ameritania, 230 W 54th St, New York, NY 10019",
        #                   "Food City Cafe and pizza, 1691 Broadway, New York, NY 10019",
        #                   "msocial Rooftop, 226 W 52nd St, New York, NY 10019",
        #                   "Tajeen Halal Food, Manhattan Times Square Hotel 790 7th Ave, corner, W 52nd St, New York, NY 10019",
        #                   "FedEx Office Print & Ship Center, 811 7th Ave, New York, NY 10019",
        #                   "Raymond's Coffee, 7th Ave and, W 53rd St, New York, NY 10019",
        #                   "7th Ave Electronics & Luggage, 841 7th Ave, New York, NY 10019",
        #                   "Flûte Champagne Bar, 205 W 54th St, New York, NY 10019",
        #                   "Chop & Go, 1700 Broadway, New York, NY 10019",
        #                   "Manhattan Acupuncture, 159 W 53rd St, New York, NY 10019"]
        # "map14_union_sq": ["Book Light Inc, 1 Union Square W # 201, New York, NY 10003",
        #                    "36 E 14th St, New York, NY 10003",
        #                    "JOE & THE JUICE, 116 University Pl, New York, NY 10003",
        #                    "1799 Ninth Regiment Plaque, 113 University Place, University Pl, New York, NY 10003",
        #                    "Forbidden Planet, 832 Broadway, New York, NY 10003",
        #                    "Ala Local 1, 113 University Pl, New York, NY 10003",
        #                    "Custom Pet Portraits by Gulchik Art, 65 E 13th St, New York, NY 10003",
        #                    "Western Union, 52 E 14th St, New York, NY 10003",
        #                    "Hyde Park Antiques, 836 Broadway, New York, NY 10003"]
        "map15_hells_kitchen": ["Althea’s Hideaway Lounge, Nightclub & Rooftop, 634 W 52nd St, New York, NY 10019",
                                "Lamborghini Manhattan, 711 11th Ave, New York, NY 10019",
                                "Kenneth Cole Productions Corporate Office, 603 W 50th St, New York, NY 10019",
                                "West Side Pie, 684 12th Ave, New York, NY 10019",
                                "Mobil, 718 11th Ave, New York, NY 10019",
                                "Coco and Toto, 730 11th Ave, New York, NY 10019",
                                "Subway, 600 W 52nd St, New York, NY 10019",
                                "Manhattan Jeep, 711 11th Ave, New York, NY 10019",
                                "FreshDirect, 630 W 52nd St, New York, NY 10036",
                                "11th Avenue Tennants HDFC, 564 W 52nd St, New York, NY 10019"]
    },
    "Paris": {
        "map0_arc_de_triomphe": [[48.875081, 2.293869],
                                 [48.874604, 2.296964],
                                 [48.872456, 2.296170],
                                 [48.873140, 2.292995]]
    }
}


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
    # if len(useful_grid_pts) > 20:
    #     num_pts_wanted = int(len(useful_grid_pts) * use_portion)
    # else:
    #     num_pts_wanted= len(useful_grid_pts)
    # num_pts_wanted = min(max_num_waypoint, num_pts_wanted)
    num_pts_wanted = int(len(useful_grid_pts) * use_portion)
    num_pts_wanted = min(max_num_waypoint, num_pts_wanted)
    print(f"{len(useful_grid_pts)} out of {len(grid_points)} grid points do have street view images, "
          f"using {num_pts_wanted} of it")

    route_pts = random.choices(useful_grid_pts, k=max_num_waypoint)
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


def get_routes_from_locations(location_list, route_file_out, route_mode="driving"):
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
        print(
            f"No Street View imagery at location: {lat}, {lng}, {metadata_response.json().get('error_message', metadata_response.json()['status'])}")
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
    # city_name = "Paris"
    for map_name in list(map_dict[city_name].keys()):
        # TODO: manually specify here if you want to generate surround-view image or singular image
        surround = False

        if surround:
            BASE_DATA_DIR = "../dataset/google_streetview_surround"
        else:
            BASE_DATA_DIR = "../dataset/google_streetview"

        # for map_name in map_dict.keys():

        area_corner_list = map_dict[city_name][map_name]
        print(f"processing map {map_name}")

        # grid coord params
        grid_size = 25
        data_dir = f"{BASE_DATA_DIR}/{map_name}"
        os.makedirs(data_dir, exist_ok=True)

        ''' ============= get route ============= '''
        area_corner_arr = np.array(area_corner_list)
        # area_grid_pts = area2grid(GOOGLE_MAPS_API_KEY, area_corner_arr, grid_size, use_portion=0.2, max_num_waypoint=5)
        # all_positions_and_headings, distance, duration = get_routes_from_pts(area_grid_pts,
        #                                                                      route_file_out=f"{BASE_DATA_DIR}/{map_name}/route_map.html",
        #                                                                      route_mode="walking")
        all_positions_and_headings, distance, duration = get_routes_from_locations(area_corner_list,
                                                                                   route_file_out=f"{BASE_DATA_DIR}/{map_name}/route_map.html",
                                                                                   route_mode="walking")

        # save used route points
        # m = folium.Map(location=area_grid_pts[0], zoom_start=5, tiles="OpenStreetMap")
        # for [lat, lng] in area_grid_pts:
        #     folium.Marker(location=[lat, lng], popup=f"({lat}, {lng})").add_to(m)
        # m.save(f"{data_dir}/grid_map.html")

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
        question_list = ["These are frames of a video. They cover all streets within an area.\n"
                         "Give me 10 specific landmarks you see in this video"]
        results = analyze_street_view(data_dir, question_list, out_dir=data_dir)
        if results:
            with open(f"{data_dir}/seen_landmarks.json", 'w') as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
