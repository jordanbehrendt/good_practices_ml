import os
import requests
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

def get_travel_images():
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
                        folder_path = "./bigfoto/{}".format(country['name'])
                        if not os.path.isdir(folder_path):
                            os.makedirs(folder_path)
                        with open("./bigfoto/{}/{}-{}.png".format(country['name'],link_path.replace('/','-'),i), 'wb') as handler:
                            handler.write(img_data)
                except TimeoutException:
                    print("Loading of result page took too much time!")
                    driver.quit()

if __name__ == "__main__":
    get_travel_images()