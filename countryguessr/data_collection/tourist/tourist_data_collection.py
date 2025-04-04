# -*- coding: utf-8 -*-
"""
tourist.tourist_data_collection
tourist------------------------

Script to scrape TOURIST datset images.
"""
# Imports
# Built-in
import os
import json
import random
import argparse

# Local

# 3r-party
import yaml
import requests
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

# Set random seed for random selection of images
random.seed(9)


def get_tourist_images(REPO_PATH, DATA_PATH):
    """Scrapes images from bigfoto.com using the paths saved in the
    file regions.json
    Saves the files into the folder {DATA_PATH}/tourist

    Args:
        REPO_PATH (str): Path to project folder.
        DATA_PATH (str): Path to data folder.

    Returns:
        None
    """
    regions_file = open(
        f"{REPO_PATH}/data_collection/tourist/"
        "url_paths_for_tourist_collection.json"
    )
    regions_json = json.load(regions_file)
    regions = regions_json["regions"]
    for region in regions:
        for country in region['array']:
            for link_path in country['links']:
                link = f'https://bigfoto.com/{region["region"]}/{link_path}/'
                result = {}
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=options
                )
                driver.get(link)
                result.update(link=link)
                try:
                    elems = driver.find_elements(
                        By.XPATH,
                        "//li[contains(@class,'blocks-gallery-item')]"
                        "//figure//a"
                    )
                    links = [elem.get_attribute('href') for elem in elems]
                    for i in range(0, len(links)):
                        img_data = requests.get(links[i]).content
                        folder_path = f"{DATA_PATH}/tourist/{country['name']}"
                        if not os.path.isdir(folder_path):
                            os.makedirs(folder_path)
                        with open(
                            f"{DATA_PATH}/tourist/{country['name']}/"
                            f"{link_path.replace('/','-')}-{i}.png",
                            'wb'
                        ) as handler:
                            handler.write(img_data)
                except TimeoutException:
                    print("Loading of result page took too much time!")
                    driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument(
        '--yaml_path',
        metavar='str',
        required=True,
        help='The path to the yaml file with the stored paths'
    )
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path']
        REPO_PATH = paths['repo_path']
        get_tourist_images(REPO_PATH, DATA_PATH)
