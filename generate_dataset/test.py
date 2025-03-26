import os
from tqdm import tqdm
from google import genai
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
img_path = "C:/Users/ROG_ZL/Documents/github/thingking_in_street_new/data/long_grid/map_2_downtownbk_test/frames/cat"
img_list = os.listdir(img_path)
upload_filelist = []
upload_filename_list = []
for img_filename in tqdm(img_list):
    path = f"{img_path}/{img_filename}"
    myfile = client.files.upload(file=path)
    upload_filelist.append(myfile)
    # uploaded_file_name = myfile.name
    # upload_filename_list.append(uploaded_file_name)



# for img_filename in tqdm(img_list):
#     myfile = client.files.get(name=img_filename)
#     upload_filelist.append(myfile)
result = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=[
        upload_filelist,
        "given the uploaded set of surround view images that covers the street view of an enclosed area, "
        "can you see a Target the market?"]
)
print(f"{result.text=}")