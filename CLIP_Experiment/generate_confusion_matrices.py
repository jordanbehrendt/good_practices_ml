import pandas as pd
import os
from typing import List
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deepsig import aso
import sys
sys.path.append('.')
import argparse
import yaml
import ast
from sklearn.metrics import confusion_matrix

def load_data(REPO_PATH, dataset_name, seed):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to repo folder.
        dataset_name (str): unique dataset name from {geoguessr, aerial, tourist}
        seed (int): random seed which defines the index of the repeated k-fold experiment
    """    
    # Directory containing CSV files
    directory = f'{REPO_PATH}/CLIP_Experiment/clip_results/seed_{seed}/image_from_prompt/{dataset_name}'

    # Get a list of all filenames in each directory
    file_list = [file for file in os.listdir(directory)]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through the files, read them as DataFrames, and append to the list
    for file in file_list:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

def create_confusion_matrices(REPO_PATH, seed):
    """
    Creates and visualizes the country and regional confusion matrices for each dataset.

    Args:
        REPO_PATH (str): path of the repo folder.
        seed (int): random seed which defines the index of the repeated k-fold experiment

    Returns:
        None
    """
    country_list = pd.read_csv(f'{REPO_PATH}/country_list/country_list_region_and_continent.csv')
    regional_ordering_index = [8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101, 105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140, 146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163, 166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192, 20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204, 78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72, 196, 13, 50, 55, 59, 62, 65, 69,
                                    86, 88, 92, 94, 113, 115, 142, 168, 172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147, 157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30, 138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57, 71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22, 37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21, 47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10, 18, 36, 49, 63, 110, 114, 118, 141]
    create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'geoguessr', seed)
    create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'aerial', seed)
    create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'tourist', seed)

def create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, dataset_name, seed):
    """
    Create and 

    Args:
        REPO_PATH (str): path of the repo folder.
        country_list (DataFrame): Data Frame of Countries and Regions
        regional_ordering_index (List): List to order the regions so that continents are grouped together
        dataset_name (str): unique dataset name from {geoguessr, aerial, tourist} 
        seed (int): random seed which defines the index of the repeated k-fold experiment
    """            
    # constant for classes
    classes = country_list['Country']
    np_classes = np.array(classes)

    # Create true_countries and predicted_countries lists
    dataset_df = load_data(REPO_PATH, dataset_name, seed)
    country_labels = dataset_df["label"]
    true_countries = []
    for elem in country_labels:
        true_countries.append(country_list.index[country_list['Country'] == elem].tolist()[0])
    dataset_df['Probs-Array'] = dataset_df['All-Probs'].apply(lambda x: ast.literal_eval(x))
    dataset_df['Predicted labels'] = dataset_df['Probs-Array'].apply(lambda x: np.argmax(np.array(x)))
    predicted_countries = dataset_df['Predicted labels']

    # Build country confusion matrix
    cf_matrix = confusion_matrix(true_countries, predicted_countries, labels=range(0, 211))
    ordered_index = np.argsort(-cf_matrix.diagonal())
    ordered_matrix = cf_matrix[ordered_index][:, ordered_index]

    regionally_ordered_matrix = cf_matrix[regional_ordering_index][:,regional_ordering_index]

    ordered_classes = np_classes[ordered_index]
    regionally_ordered_classes = np_classes[regional_ordering_index]

    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    ordered_df_cm = pd.DataFrame(
        ordered_matrix, index=ordered_classes, columns=ordered_classes)
    regionally_ordered_df_cm = pd.DataFrame(
        regionally_ordered_matrix, index=regionally_ordered_classes, columns=regionally_ordered_classes)

    np_regions = np.sort(np.array(list(set(country_list['Intermediate Region Name']))))

    # Build region confusion matrix
    true_regions = []
    predicted_regions = []
    for i in range(0, len(true_countries)):
        true_regions.append(ast.literal_eval(country_list.iloc[true_countries[i]]["One Hot Region"]).index(1))
        predicted_regions.append(ast.literal_eval(country_list.iloc[predicted_countries[i]]["One Hot Region"]).index(1))

    regions_cf_matrix = confusion_matrix(
        true_regions, predicted_regions, labels=range(0, len(np_regions)))
    regions_ordered_index = np.argsort(-regions_cf_matrix.diagonal())
    regions_ordered_matrix = regions_cf_matrix[regions_ordered_index][:,regions_ordered_index]
    ordered_regions = np_regions[regions_ordered_index]

    regions_df_cm = pd.DataFrame(regions_cf_matrix, index=np_regions, columns=np_regions)
    regions_ordered_df_cm = pd.DataFrame(regions_ordered_matrix, index=ordered_regions, columns=ordered_regions)


    if not os.path.exists(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/'):
        os.makedirs(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/')

    plt.figure(1, figsize=(120, 70))
    figure = sns.heatmap(df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
    figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/simple_confusion_matrix.png')
    plt.figure(2, figsize=(120, 70))
    ordered_figure = sns.heatmap(ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
    ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/ordered_confusion_matrix.png')
    plt.figure(3, figsize=(120, 70))
    regionally_ordered_figure = sns.heatmap(regionally_ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
    regionally_ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regionally_ordered_confusion_matrix.png')
    plt.figure(4, figsize=(120, 70))
    regions_figure = sns.heatmap(regions_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
    regions_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regions_confusion_matrix.png')
    plt.figure(5, figsize=(120, 70))
    regions_ordered_figure = sns.heatmap(regions_ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
    regions_ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regions_ordered_confusion_matrix.png')
    return



if __name__ == "__main__":
    """
    Create and compute confusion matrices with seed = 4808 (the choice of seed is irrelevant)
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        create_confusion_matrices(REPO_PATH, 4808)