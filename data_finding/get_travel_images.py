import os
import requests
import argparse
import yaml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-popup-blocking')
options.add_argument('--disable-notifications')

regions = [ 
    {'region': 'africa', 
    'array': [
        {'name': 'Mauritius', 'links': ['mauritius']},
        {'name': 'Tunisia', 'links': ['tunisia']},
        {'name': 'Egypt', 'links': ['egypt']},
        {'name': 'Ghana', 'links': ['ghana']},
        {'name': 'Morocco', 'links': ['morocco']},
        {'name': 'South Africa', 'links': ['south-africa']},
    ]},
    {'region': 'america', 
    'array': [
        {'name': 'Canada', 'links': ['canada']},
        {'name': 'Brazil', 'links': ['rio-de-janeiro']}
    ]},
    {'region': 'asia', 
    'array': [
        {'name': 'Indonesia', 'links': ['bali']},
        {'name': 'Hong Kong', 'links': ['hongkong']},
        {'name': 'China', 'links': ['beijing']},
        {'name': 'Israel', 'links': ['israel']},
        {'name': 'Laos', 'links': ['laos']},
        {'name': 'Malaysia', 'links': ['malaysia']},
        {'name': 'Singapore', 'links': ['singapore']},
        {'name': 'South Korea', 'links': ['seoul']},
        {'name': 'Turkey', 'links': ['turkey']},
        {'name': 'Thailand', 'links': ['bangkok']},
        {'name': 'Vietnam', 'links': ['vietnam']},
        {'name': 'Uzbekistan', 'links': ['uzbekistan']},
    ]},
    {'region': 'pacific', 
    'array': [
        {'name': 'Australia', 'links': ['australia']},
        {'name': 'New Zealand', 'links': ['new-zealand']},
    ]},
    {'region': 'europe',
    'array': [
        {'name': 'Croatia', 'links': ['croatia']},
        {'name': 'Czech Republic', 'links': ['prague']},
        {'name': 'Austria', 'links': ['austria/graz', 'austria/vienna', 'austria/salzburg']},
        {'name': 'Bulgaria', 'links': ['bulgaria/sofia', 'bulgaria/plovdiv']},
        {'name': 'Denmark', 'links': ['copenhagen']},
        {'name': 'France', 'links': ['paris']},
        {'name': 'Germany', 'links': ['germany/berlin', 'germany/munich', 'germany/bonn', 'germany/neuswan']},
        {'name': 'Netherlands', 'links': ['netherlands/other-2', 'netherlands/amsterdam', 'netherlands/rotterdam', 'netherlands/denhaag', 'netherlands/utrecht']},
        {'name': 'Poland', 'links': ['poland']},
        {'name': 'Switzerland', 'links': ['switzerland/zurich']},
        {'name': 'United Kingdom', 'links': ['london']},
    ]},
]

def get_travel_images(REPO_PATH):
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
                        folder_path = "{}/data_finding/bigfoto/{}".format(REPO_PATH, country['name'])
                        if not os.path.isdir(folder_path):
                            os.makedirs(folder_path)
                        with open("{}/data_finding/bigfoto/{}/{}-{}.png".format(REPO_PATH, country['name'],link_path.replace('/','-'),i), 'wb') as handler:
                            handler.write(img_data)
                except TimeoutException:
                    print("Loading of result page took too much time!")
                    driver.quit()

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