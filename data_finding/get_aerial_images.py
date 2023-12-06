import leafmap
import csv
import argparse
import yaml
import ast
from PIL import Image, ImageSequence
import requests
import os

def get_aerial_images(REPO_PATH):
    # Bounding Boxes taken from 'natural earth data http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'
    file = open('{}/data_finding/bounding_boxes.csv'.format(REPO_PATH))
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
                os.makedirs("{}/data_finding/open_aerial_map/{}/".format(REPO_PATH, country[0]))
                for i in range(0,len(images)):
                    img_data = requests.get(images[i]).content
                    with open("{}/data_finding/open_aerial_map/{}/aerial-{}-{}.png".format(REPO_PATH, country[0],country[0],i), 'wb') as handler:
                        handler.write(img_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        get_aerial_images(REPO_PATH)
