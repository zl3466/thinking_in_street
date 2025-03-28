from _0_area_random_route_nav import *

roud_data_dir = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/data/long_route_random_single/map1_downtown_bk/route_data.json"

route_data = json.load(open(roud_data_dir))
decoded_pts = []
for frame in route_data["frames"]:
    lat = frame["coordinates"]["lat"]
    lng = frame["coordinates"]["lng"]
    decoded_pts.append((lat, lng))


route_map = folium.Map(location=decoded_pts[0], zoom_start=7)

# Add polyline to the map
folium.PolyLine(decoded_pts, color="blue", weight=5, opacity=0.7).add_to(route_map)

# save the map
route_map.save(f"{roud_data_dir}/../route_map.html")

