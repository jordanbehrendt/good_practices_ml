import leafmap
import os
import requests
import argparse
import yaml
import csv
import json
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
                

def get_aerial_images(REPO_PATH):
    """Finds images from openaerialmap.org using bounding boxes saved in bounding_boxes.csv
    Saves the files into the folder {REPO_PATH}/data/open_aerial_map

    Args:
        REPO_PATH (str): Path to project folder.

    Returns:
        None
    """
    # Bounding Boxes taken from 'natural earth data http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'
    file = open('{}/data_finding/bounding_boxes.csv'.format(REPO_PATH))
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    bounding_boxes = []
    for row in csvreader:
        bounding_boxes.append(row)

    for country in bounding_boxes:
        bbox = [float(country[1]), float(country[2]), float(country[3]), float(country[4])]
        gdf = leafmap.oam_search(
            bbox=bbox, limit=2, return_gdf=True
        )
        if gdf is not None:
            images = gdf['thumbnail'].tolist()
            if len(images) != 0:
                os.makedirs("{}/data/open_aerial_map/{}/".format(REPO_PATH, country[0]))
                for i in range(0,len(images)):
                    img_data = requests.get(images[i]).content
                    with open("{}/data/open_aerial_map/{}/aerial-{}-{}.png".format(REPO_PATH, country[0],country[0],i), 'wb') as handler:
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
        get_travel_images(REPO_PATH)
        get_aerial_images(REPO_PATH)