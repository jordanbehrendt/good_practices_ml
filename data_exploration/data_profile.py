# Standard library imports
import argparse
import math
from typing import Union, Dict

# Third-party library imports
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ydata_profiling import ProfileReport

# Local or intra-package imports
from scripts import load_geoguessr_data
import os


def log(x: Union[int, float]) -> Union[int, float]:
    """
    Calculates the logarithm of a given value and returns it (minimum 1)

    Args:
        x (Union[int, float]): Value that should be logarithmized

    Returns:
        Union[int, float]: Logarithmic value (minimum 1)
    """
    return max(1, math.log(x))

def world_heat_map(image_distribution_path: str, output_dir: str) -> None:    
    """
    Generates a world heat map based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the plot will be saved.

    Returns:
        None: Saves the world heat map plot at the specified output directory.
    """
    # Reading the world map data 
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # Reading image distribution data from CSV
    image_distribution = pd.read_csv(image_distribution_path)

    # Applying logarithm to the 'count' column in image distribution
    count_column = image_distribution["count"]
    new_count_column = count_column.apply(log)
    image_distribution["count"] = new_count_column

    # Merging world and image distribution dataframes
    world['join'] = 1
    image_distribution['join'] = 1
    data_frame_full = world.merge(image_distribution, on='join').drop('join', axis=1)
    image_distribution.drop('join', axis=1, inplace=True)

    # Checking for matches between 'name' and 'label' columns
    data_frame_full['match'] = data_frame_full.apply(lambda x: x["name"].find(x["label"]), axis=1).ge(0)

    # Filtering the dataframe based on matches and plotting the world map
    df = data_frame_full[data_frame_full['match']]
    df.plot(column='count', legend=True)

    # Saving the world heat map at the specified output path
    plt.savefig(os.path.join(output_dir, world_heat_map.jpg))

def data_profile(DATA_PATH: str, REPO_PATH: str) -> None:
    """
    Generates a profile report, image distribution CSV, and world heat map based on Geoguessr data.

    Args:
        DATA_PATH (str): File path of the Geoguessr data.
        REPO_PATH (str): Root path of the repository.

    Returns:
        None: Saves the profile report, image distribution CSV, and world heat map.
    """
    # Loading Geoguessr data and creating a profile report
    geoguessr_df = load_geoguessr_data.load_data(DATA_PATH=DATA_PATH, size_constraints=True)
    profile = ProfileReport(geoguessr_df, title='Geoguessr Profile Report')

    # Saving the profile report as HTML
    profile_html_path = f'{REPO_PATH}/data_exploration/geoguessr_profile.html'
    profile.to_file(profile_html_path)

    # Generating image distribution and saving as CSV
    image_distribution = geoguessr_df['label'].value_counts()
    image_distribution_path = f'{REPO_PATH}/data_exploration/image_distribution.csv'
    image_distribution.to_csv(image_distribution_path)

    # Generating and saving world heat map based on image distribution
    world_heat_map(image_distribution_path, output_dir=f'{REPO_PATH}/data_exploration/')

def main(user: str, yaml_path: str) -> None:
    """
    Create Geoguessr Data Profile based on user and YAML file paths.

    Args:
        user (str): The user of the gpml group.
        yaml_path (str): The path to the YAML file with the stored paths.

    Returns:
        None: Generates Geoguessr data profile report, image distribution CSV, and world heat map.
    """
    with open(yaml_path) as file:
        paths: Dict[str, Dict[str, str]] = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][user]
        REPO_PATH = paths['repo_path'][user]
        data_profile(DATA_PATH, REPO_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Geoguessr Data Profile')
    parser.add_argument('--user', metavar='str', required=True, help='the user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='the path to the yaml file with the stored paths')
    args = parser.parse_args()
    main(args.user, args.yaml_path)
