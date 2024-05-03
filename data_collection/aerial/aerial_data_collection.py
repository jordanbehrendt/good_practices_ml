import leafmap
import os
import requests
import argparse
import yaml
import csv
import json
import pandas
import operator
import math
import random
import ast
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
import shapely.wkt

# Options for the selenium chrome webdriver
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-popup-blocking')
options.add_argument('--disable-notifications')

#Set random seed for random selection of images
random.seed(9)

def find_interior_boxes(REPO_PATH):
    """Find a large interior box for each country, based on polygon data taken from natural earth (https://github.com/martynafford/natural-earth-geojson/blob/master/110m/cultural/ne_110m_admin_0_countries.json)
    Saves the internal boxes into the file interior_boxes.csv
    
    Args:
        REPO_PATH (str): Path to project folder.

    Returns:
        None
    """
    polygon_file = open("{}/data_collection/aerial/ne_110m_admin_0_countries.json".format(REPO_PATH))
    polygons_json = json.load(polygon_file)
    countries_polygons = polygons_json["features"]

    country_panda = pandas.read_csv('{}/country_list/country_list_region_and_continent.csv'.format(REPO_PATH))
    interior_boxes = []
    for index, row in country_panda.iterrows():
        polygon_object = []
        for country_pol in countries_polygons:
            if (country_pol["properties"]["ISO_A2"] == row['Alpha2Code']):
                polygon_object = country_pol["geometry"]["coordinates"]
        coords = []
        nested_coords = []
        if (len(polygon_object) > 0):
            for el in polygon_object:
                if len(el) > 1:
                    points = []
                    for point in el:
                        points.append((float(point[0]), float(point[1])))
                    coords.append((Polygon(points)))
                else:
                    nested = []
                    for nested_point in el[0]:
                        nested.append((float(nested_point[0]), float(nested_point[1])))
                    nested_coords.append(Polygon(nested))
        if (len(coords) > 0):
            result = coords[0]
            box = Polygon(get_rectangle(result, result.centroid))
            count = 0
            while not result.covers(box) and count < 10000:
                box = box.buffer(-0.005, join_style="mitre")
                count += 1
            if result.covers(box):
                xx,yy = box.exterior.coords.xy
                interior_boxes.append({'Country': row['Country'], 'Alpha2Code': row['Alpha2Code'], 'box': [min(xx), max(yy), max(xx), min(yy)], 'boxes': ''})
            else:
                print(row['Country'])            
        elif (len(nested_coords) > 0):
            result = nested_coords
            boxes = []
            for el in result:
                box = Polygon(get_rectangle(el, el.centroid))
                count = 0
                while not el.covers(box) and count < 10000:
                    box = box.buffer(-0.005, join_style="mitre")
                    count += 1
                if el.covers(box):
                    xx,yy = box.exterior.coords.xy
                    boxes.append([min(xx), max(yy), max(xx), min(yy)])
                else:
                    print(row['Country'])
            interior_boxes.append({'Country': row['Country'], 'Alpha2Code': row['Alpha2Code'], 'box': '', 'boxes': boxes})   
        else:
            print(row['Country'])         
    interior_panda = pandas.DataFrame(interior_boxes)
    interior_panda.to_csv("{}/data_collection/aerial/interior_boxes.csv".format(REPO_PATH))


def get_rectangle(polygon, center):
    """Creates rectangle based on the given polygon and the center point.
    First measures the shortest distances to the boundaries in the x and y directions
    then creates and returns a rectangle defined by these smallest distances

    Args:
        polygon (Polygon): The country polygon provided by natural earth 
        center (Point): The center of the polygon

    Returns:
        bbox (List): The largest possible rectangle (probably not interior) created by going as far as possible in x and y directions
    """
    distances = []
    multipoints = []
    points = []
    boundary = polygon.boundary
    multipoints.append(boundary.intersection(LineString([(center.x + 500, center.y),(center.x - 500, center.y)]), grid_size=0.00000000000001))
    multipoints.append(boundary.intersection(LineString([(center.x, center.y + 500),(center.x, center.y - 500)]), grid_size=0.00000000000001))
    for i in range(0,len(multipoints)):
        intersections = [p for p in multipoints[i].geoms]
        intersections_distances = []
        for j in range(0, len(intersections)):
            intersections_distances.append(intersections[j].distance(center))
        distances.append(min(intersections_distances))
        points.append(intersections[intersections_distances.index(min(intersections_distances))])
    dif_x = points[0].x - center.x
    dif_y = points[1].y - center.y
    bbox = [[center.x - dif_x, center.y + dif_y], [center.x + dif_x, center.y + dif_y], [center.x + dif_x, center.y - dif_y], [center.x - dif_x, center.y + dif_y]]
    return bbox

def get_aerial_images(REPO_PATH, DATA_PATH):
    """Finds images from openaerialmap.org using interior_boxes saved in interior_boxes.csv and bounding boxes (for a few island or peninsular countries) saved in island_bounding_boxes.csv
    Saves the files into the folder {DATA_PATH}/aerial

    Args:
        REPO_PATH (str): Path to project folder.
        DATA_PATH (str): Path to data folder.

    Returns:
        None
    """
    interior_boxes = pandas.read_csv('{}/data_collection/aerial/interior_boxes.csv'.format(REPO_PATH))

    for index, row in interior_boxes.iterrows():
        count = 0
        if not pandas.isnull(row["box"]):
            new_count = get_images_from_bbox(ast.literal_eval(row["box"]), count, row["Country"], DATA_PATH)
            count = new_count
        elif not pandas.isnull(row["boxes"]):
            loaded_array = ast.literal_eval(row["boxes"])
            for el in loaded_array:
                new_count = get_images_from_bbox(el, count, row["Country"], DATA_PATH)   
                count = new_count

    island_bounding_boxes = pandas.read_csv('{}/data_collection/aerial/island_bounding_boxes.csv'.format(REPO_PATH))
    for index, row in island_bounding_boxes.iterrows():
        get_images_from_bbox([row['x1'], row['y1'], row['x2'], row['y2']], 0, "islandbbox-{}".format(row["Country"]), DATA_PATH) 


def get_images_from_bbox(bbox, count, country, DATA_PATH):
    """Downloads images within the given bounding box from openaerialmap using the leafmap module

    Args:
        bbox (List): The bounding box that provides the area within which images can be found
        count (int): The current amount of images already found
        country (str): the name of the country (prefaced with islandbbox- for island countries) to provide the folder and path for saving of the images
        DATA_PATH (str): Path to data folder.

    Returns:
        count (int): the updated amount of images already found
    """
    gdf = leafmap.oam_search(
        bbox=bbox, limit=100, gsd_to=0.05, order_by='gsd', sort='asc'
    )
    if gdf is not None:
        images = gdf['thumbnail'].tolist()
        if len(images) != 0:
            random.shuffle(images)
            only_country = country.split('-')[-1]
            if not os.path.isdir("{}/aerial/{}/".format(DATA_PATH, only_country)):
                os.makedirs("{}/aerial/{}/".format(DATA_PATH, only_country))
            for i in range(0,min(20,len(images))):
                    img_data = requests.get(images[i]).content
                    with open("{}/aerial/{}/aerial-{}-{}.png".format(DATA_PATH, only_country,country,count), 'wb') as handler:
                        handler.write(img_data)
                    count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        # find_interior_boxes(REPO_PATH)
        get_aerial_images(REPO_PATH)