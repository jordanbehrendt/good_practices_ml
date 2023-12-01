import leafmap
import csv
import ast
from PIL import Image, ImageSequence
import requests
import os

def get_aerial_images():
    # Bounding Boxes taken from 'natural earth data http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'

    file = open('bounding_boxes.csv')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    bounding_boxes = []
    for row in csvreader:
        bounding_boxes.append(row)

    country_filenames = []
    for country in bounding_boxes:
        bbox = [float(country[1]), float(country[2]), float(country[3]), float(country[4])]
        gdf = leafmap.oam_search(
            bbox=bbox, limit=2, return_gdf=True
        )
        if gdf is not None:
            # print(f'{country[0]}: Found {len(gdf)} images')
            images = gdf['thumbnail'].tolist()
            if len(images) != 0:
                os.makedirs("./open_aerial_map/{}/".format(country[0]))
                for i in range(0,len(images)):
                    img_data = requests.get(images[i]).content
                    with open("./open_aerial_map/{}/aerial-{}-{}.png".format(country[0],country[0],i), 'wb') as handler:
                        handler.write(img_data)

if __name__ == "__main__":
    get_aerial_images()
