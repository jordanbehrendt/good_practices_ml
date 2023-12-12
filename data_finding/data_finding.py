import leafmap
import os
import requests
import argparse
import yaml
import csv
import json
import pandas
import operator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Options for the selenium chrome webdriver
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-popup-blocking')
options.add_argument('--disable-notifications')

def get_travel_images(REPO_PATH):
    """Scrapes images from bigfoto.com using the paths saved in the file regions.json
    Saves the files into the folder {REPO_PATH}/data/bigfoto

    Args:
        REPO_PATH (str): Path to project folder.

    Returns:
        None
    """
    regions_file = open("{}/data_finding/regions.json".format(REPO_PATH))
    regions_json = json.load(regions_file)
    regions = regions_json["regions"]
    for region in regions:
        for country in region['array']:
            for link_path in country['links']:
                link = 'https://bigfoto.com/{}/{}/'.format(region['region'],link_path)
                result = {}
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.get(link)
                result.update(link=link)
                try:
                    elems = driver.find_elements(By.XPATH, "//li[contains(@class,'blocks-gallery-item')]//figure//a")
                    links = [elem.get_attribute('href') for elem in elems]
                    for i in range(0,len(links)):
                        img_data = requests.get(links[i]).content
                        folder_path = "{}/data/bigfoto/{}".format(REPO_PATH, country['name'])
                        if not os.path.isdir(folder_path):
                            os.makedirs(folder_path)
                        with open("{}/data/bigfoto/{}/{}-{}.png".format(REPO_PATH, country['name'],link_path.replace('/','-'),i), 'wb') as handler:
                            handler.write(img_data)
                except TimeoutException:
                    print("Loading of result page took too much time!")
                    driver.quit()

def find_interior_centroid_boxes(REPO_PATH):
    """Find largest interior box for each country, based on polygon data taken from natural earth (https://github.com/martynafford/natural-earth-geojson/blob/master/110m/cultural/ne_110m_admin_0_countries.json)
    and centroids taken from (https://github.com/gavinr/world-countries-centroids/blob/master/dist/countries.csv)
    Saves the internal boxes into the file internal_boxes.csv
    
    Args:
        REPO_PATH (str): Path to project folder.

    Returns:
        None
    """
    polygon_file = open("{}/data_finding/ne_110m_admin_0_countries.json".format(REPO_PATH))
    polygons_json = json.load(polygon_file)
    countries_polygons = polygons_json["features"]

    country_panda = pandas.read_csv('{}/data_finding/country_list.csv'.format(REPO_PATH))
    country_centroids = pandas.read_csv('{}/data_finding/countries_centroids.csv'.format(REPO_PATH))

    interior_boxes = []
    for index, row in country_panda.iterrows():
        centroid_object = country_centroids[country_centroids['ISO'] == row['Alpha2Code']]
        if not centroid_object.empty:
            longitude = float(centroid_object['longitude'].values[0])
            latitude = float(centroid_object['latitude'].values[0])
            polygon_object = []
            for country_pol in countries_polygons:
                if (country_pol["properties"]["ISO_A2"] == row['Alpha2Code']):
                    polygon_object = country_pol["geometry"]["coordinates"]
            inner_box = [longitude - 10, latitude - 10, longitude + 10, latitude + 10]
            polygon_exist = False
            if (len(polygon_object) > 0):
                polygon_exist = True
                for el in polygon_object[0]:
                    if len(el) > 2:
                        for nested_el in el:
                            if len(nested_el) == 2:
                                if (((nested_el[0] > inner_box[0]) and (nested_el[0] < inner_box[2])) and ((nested_el[1] > inner_box[1]) and (nested_el[1] < inner_box[3]))):
                                    if (nested_el[0] < longitude):
                                        inner_box[0] = nested_el[0]
                                    else:
                                        inner_box[2] = nested_el[0]
                                    if (nested_el[1] < latitude):
                                        inner_box[1] = nested_el[1]
                                    else:
                                        inner_box[3] = nested_el[1]
                    else:
                        if (((el[0] > inner_box[0]) and (el[0] < inner_box[2])) and ((el[1] > inner_box[1]) and (el[1] < inner_box[3]))):
                            if (el[0] < longitude):
                                inner_box[0] = el[0]
                            else:
                                inner_box[2] = el[0]
                            if (el[1] < latitude):
                                inner_box[1] = el[1]
                            else:
                                inner_box[3] = el[1]
            if polygon_exist:
                interior_boxes.append({'Country': row['Country'], 'Alpha2Code': row['Alpha2Code'], 'x1': inner_box[0], 'y1': inner_box[1], 'x2': inner_box[2], 'y2': inner_box[3]})            
    interior_panda = pandas.DataFrame(interior_boxes)
    interior_panda.to_csv("{}/data_finding/interior_boxes.csv".format(REPO_PATH))


def get_aerial_images(REPO_PATH):
    """Finds images from openaerialmap.org using bounding boxes saved in bounding_boxes.csv
    Saves the files into the folder {REPO_PATH}/data/open_aerial_map

    Args:
        REPO_PATH (str): Path to project folder.

    Returns:
        None
    """
    interior_boxes = pandas.read_csv('{}/data_finding/interior_boxes.csv'.format(REPO_PATH))

    for index, row in interior_boxes.iterrows():
        bbox = [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
        gdf = leafmap.oam_search(
            bbox=bbox, limit=2, return_gdf=True
        )
        if gdf is not None:
            images = gdf['thumbnail'].tolist()
            if len(images) != 0:
                os.makedirs("{}/data/open_aerial_map/{}/".format(REPO_PATH, row['Country']))
                for i in range(0,len(images)):
                    img_data = requests.get(images[i]).content
                    with open("{}/data/open_aerial_map/{}/aerial-{}-{}.png".format(REPO_PATH, row['Country'],row['Country'],i), 'wb') as handler:
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
        # get_travel_images(REPO_PATH)
        get_aerial_images(REPO_PATH)
        # find_interior_centroid_boxes(REPO_PATH)