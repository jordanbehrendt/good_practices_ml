# Standard library imports
import argparse
import math
from typing import Union, Dict

# Third-party library imports
import geopandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import yaml
#from ydata_profiling import ProfileReport

# Local or intra-package imports
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current directory
utils_dir = os.path.join(current_dir, '..', 'utils')

# Add the utils directory to sys.path
sys.path.append(utils_dir)
import sys
sys.path.append('/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/utils')


import load_dataset, pdf

def line_graph(image_distribution_path: str, output_dir: str, logarithmic: bool) -> None:
    """
    Generate and save a line graph based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the graph will be saved.
        logarithmic (bool): Flag indicating whether to use logarithmic scale.

    Returns:
        None: Saves the line graph at the specified output directory.
    """
    # Read the image distribution data
    image_distribution = pd.read_csv(image_distribution_path)

    # Create and customize the line graph
    plt.figure(figsize=(8, 6))
    if logarithmic:
        plt.plot(image_distribution['label'], image_distribution['count'].apply(log), linestyle='-', color='b')
        graph_name = 'line_graph_log.jpg'
        ylabel = 'Log Total Images'
        title = 'Image Distribution (Logarithmic)'
    else:
        plt.plot(image_distribution['label'], image_distribution['count'], linestyle='-', color='b')
        graph_name = 'line_graph.jpg'
        ylabel = 'Total Images'
        title = 'Image Distribution'

    # Label axes and set the title
    plt.xlabel('Country')
    plt.ylabel(ylabel)
    plt.title(title)

    # Hide x-axis tick labels
    empty_labels = [''] * len(image_distribution['label'])
    plt.xticks(range(len(image_distribution['label'])), empty_labels, fontsize='small')

    # Ensuring y-ticks are integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Display and save the plot
    plt.grid(axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, graph_name))
    plt.close()

def bar_graph(image_distribution_path: str, output_dir: str, logarithmic: bool) -> None:
    """
    Generate and save a vertical bar graph based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the graph will be saved.
        logarithmic (bool): Flag indicating whether to use logarithmic scale.

    Returns:
        None: Saves the bar graph at the specified output directory.
    """
    # Read the image distribution data
    image_distribution = pd.read_csv(image_distribution_path)

    # Determine the number of labels
    num_labels = len(image_distribution['label'])

    # Adjust the figure height dynamically based on the number of labels
    # Assuming 0.3 inches per label, minimum 8 inches, max 20 inches
    figure_height = max(8, min(20, 0.3 * num_labels))

    # Create and customize the bar graph
    plt.figure(figsize=(10, figure_height))
    if logarithmic:
        plt.barh(image_distribution['label'], image_distribution['count'].apply(log), color='b')
        graph_name = 'bar_graph_log.jpg'
        xlabel = 'Log Total Images'
        title = 'Image Distribution (Logarithmic)'
    else:
        plt.barh(image_distribution['label'], image_distribution['count'], color='b')
        graph_name = 'bar_graph.jpg'
        xlabel = 'Total Images'
        title = 'Image Distribution'

    # Label axes and set the title
    plt.ylabel('Country')
    plt.xlabel(xlabel)
    plt.title(title)

    # Ensuring x-ticks are integers if not logarithmic
    if not logarithmic:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Display and save the plot
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, graph_name))
    plt.close()

def log(x: Union[int, float]) -> Union[int, float]:
    """
    Calculates the logarithm of a given value and returns it (minimum 1)

    Args:
        x (Union[int, float]): Value that should be logarithmized

    Returns:
        Union[int, float]: Logarithmic value (minimum 1)
    """
    return max(1, math.log(x, 10))

def world_heat_map(image_distribution_path: str, output_dir: str, logarithmic: bool) -> None:
    """
    Generates a world heat map based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the plot will be saved.
        logarithmic (bool): Determines whether to apply logarithm to the data.

    Returns:
        None: Saves the world heat map plot at the specified output directory.
    """
    # Reading the world map data 
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # Reading image distribution data from CSV
    image_distribution = pd.read_csv(image_distribution_path)

    # Applying logarithm to the 'count' column in image distribution
    if logarithmic:
        image_distribution['count'] = image_distribution['count'].apply(log)
        graph_name = 'world_heat_map_log.jpg'
        title = 'Logarithmic World Heat Map'
        legend_title = 'Logarithmic Count of Images per Country'
    else:
        graph_name = 'world_heat_map.jpg'
        title = 'World Heat Map'
        legend_title = 'Count of Images per Country'
        
    # Merging world and image distribution dataframes
    world['join'] = 1
    image_distribution['join'] = 1
    data_frame_full = world.merge(image_distribution, on='join').drop('join', axis=1)
    image_distribution.drop('join', axis=1, inplace=True)

    # Checking for matches between 'name' and 'label' columns
    data_frame_full['match'] = data_frame_full.apply(lambda x: x["name"].find(x["label"]), axis=1).ge(0)

    # Filtering the dataframe based on matches and plotting the world map
    df = data_frame_full[data_frame_full['match']]
    ax = df.plot(column='count', legend=True, legend_kwds={'label': legend_title})

    # Saving the world heat map at the specified output path
    plt.title(title)
    plt.savefig(os.path.join(output_dir, graph_name))
    plt.close()

def data_profile(dataset_dir: str, REPO_PATH: str, dataset_name: str) -> None:
    """
    Generates a profile report, image distribution CSV, world heat map, and line graph based on a dataset.

    Args:
        dataset_dir (str): File path of the dataset.
        repo_path (str): Root path of the repository.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    """
    output_dir = os.path.join(REPO_PATH, 'data_collection/data_exploration', dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading dataset and creating a profile report
    df = load_dataset.load_data(DATA_PATH=dataset_dir, size_constraints=False)
    #profile = ProfileReport(df, title=f'{dataset_name} Profile Report')
    #profile.to_file(os.path.join(output_dir, 'profile.html'))

    # Generating image distribution and saving as CSV
    image_distribution = df['label'].value_counts()
    image_distribution.to_csv(os.path.join(output_dir, 'image_distribution.csv'))

    # Generating and saving world heat map and line graph based on image distribution (default and logarithmic) and save it in one pdf
    world_heat_map(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=True)
    world_heat_map(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=False)
    line_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=True)
    line_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=False)
    bar_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=True)
    bar_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=False)
    pdf.create_merged_pdf(output_dir, 'image_distribution')

def create_dataset_profile(user: str, yaml_path: str, dataset_dir: str, dataset_name: str) -> None:
    """
    Create a dataset profile including a report, image distribution CSV, and world heat map.

    Args:
        user (str): The user of the gpml group.
        yaml_path (str): The path to the YAML file with the stored paths.
        dataset_dir (str): The path to directory with the dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    with open(yaml_path) as file:
        paths: Dict[str, Dict[str, str]] = yaml.safe_load(file)
        repo_path = paths['repo_path'][user]
        data_profile(dataset_dir, repo_path, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Dataset Profile')
    parser.add_argument('--user', metavar='str', required=True, help='the user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='the path to the yaml file with the stored paths')
    parser.add_argument('--dataset_dir', metavar='str', required=True, help='the path to directory with the dataset')
    parser.add_argument('--dataset_name', metavar='str', required=True, help='the name of the dataset')
    args = parser.parse_args()
    create_dataset_profile(args.user, args.yaml_path, args.dataset_dir, args.dataset_name)
