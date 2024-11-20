import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils.confusion_matrix import create_and_save_confusion_matrices
import argparse
import yaml
import ast

def load_data(REPO_PATH, dataset_name, seed):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to repo folder.
        dataset_name (str): unique dataset name from {geoguessr, aerial, tourist}
        seed (int): random seed which defines the index of the repeated k-fold experiment
    """    
    # Directory containing CSV files
    directory = f'{REPO_PATH}/CLIP_Experiment/clip_results/seed_{seed}/extended_prompt/{dataset_name}'

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

def generate_confusion_matrices(REPO_PATH, seed):
    """
    Creates and visualizes the country and regional confusion matrices for each dataset.

    Args:
        REPO_PATH (str): path of the repo folder.
        seed (int): random seed which defines the index of the repeated k-fold experiment

    Returns:
        None
    """
    country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv')
    create_labels_and_call_confusion_matrices(REPO_PATH, country_list, 'geoguessr', seed)
    create_labels_and_call_confusion_matrices(REPO_PATH, country_list, 'aerial', seed)
    create_labels_and_call_confusion_matrices(REPO_PATH, country_list, 'tourist', seed)

def create_labels_and_call_confusion_matrices(REPO_PATH, country_list, dataset_name, seed):
    """
    Create the true and predicted labels for a dataset and then call the confusion matrices function.

    Args:
        REPO_PATH (str): path of the repo folder.
        country_list (DataFrame): Data Frame of Countries and Regions
        dataset_name (str): unique dataset name from {geoguessr, aerial, tourist} 
        seed (int): random seed which defines the index of the repeated k-fold experiment
    """            

    # Create true_countries and predicted_countries lists
    dataset_df = load_data(REPO_PATH, dataset_name, seed)
    country_labels = dataset_df["label"]
    true_countries = []
    for elem in country_labels:
        true_countries.append(country_list.index[country_list['Country'] == elem].tolist()[0])
    dataset_df['Probs-Array'] = dataset_df['All-Probs'].apply(lambda x: ast.literal_eval(x))
    dataset_df['Predicted labels'] = dataset_df['Probs-Array'].apply(lambda x: np.argmax(np.array(x)))
    predicted_countries = dataset_df['Predicted labels']

    SAVE_FIGURES_PATH = f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}'

    create_and_save_confusion_matrices(REPO_PATH,SAVE_FIGURES_PATH,true_countries,predicted_countries,True)
    create_and_save_confusion_matrices(REPO_PATH,SAVE_FIGURES_PATH,true_countries,predicted_countries,False)
    return



if __name__ == "__main__":
    """
    Create and compute confusion matrices with seed = 4808 (the choice of seed is irrelevant)
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path']
        generate_confusion_matrices(REPO_PATH, 4808)