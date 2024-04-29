import sys
sys.path.append('.')

import models
from models import model_tester
import scripts
from scripts import load_dataset
import clip
import torch
import csv
import pandas as pd
import argparse
import yaml

def run_experiments(DATA_PATH: str, REPO_PATH: str):
    seed = 1234
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    datasets = []

    geoguessr = load_dataset.load_data(f'{DATA_PATH}/compressed_dataset/', 0, 5000, False, False, seed)
    geoguessr = geoguessr.head(int(len(geoguessr)*0.2))
    geoguessr = load_dataset.ImageDataset_from_df(geoguessr, preprocessor, name= "geoguessr")
    datasets.append(geoguessr)
    # tourist = load_dataset.load_data(f'{DATA_PATH}/tourist/', 0, 5000, False, False, seed)
    # tourist = load_dataset.ImageDataset_from_df(tourist, preprocessor, name= "tourist")
    # datasets.append(tourist)
    # aerialmap = load_dataset.load_data(f'{DATA_PATH}/open_aerial_map/', 0, 5000, False, False, seed)
    # aerialmap = load_dataset.ImageDataset_from_df(aerialmap, preprocessor, name= "aerial")
    # datasets.append(aerialmap)

    default_prompt = lambda x: f"{x}"
    image_from_prompt = lambda x: f"This image shows the country {x}"

    batch_sizes = []

    geoguessr_batch_size = 430
    batch_sizes.append(geoguessr_batch_size)
    # tourist_batch_size = 115
    # batch_sizes.append(tourist_batch_size)
    # aerialmap_batch_size = 14
    # batch_sizes.append(aerialmap_batch_size)

    country_list = pd.read_csv(f'{REPO_PATH}/data_finding/country_list.csv')["Country"].to_list()

    folder_path = f'{REPO_PATH}/CLIP_Experiment'
    model_name = 'clip_results'

    default_prompt_name = 'default_prompt'
    image_prompt_name = 'image_from_prompt'

    for i in range(0,len(datasets)):
        test = model_tester.ModelTester(datasets[i], model, [default_prompt, image_from_prompt], batch_sizes[i], country_list, seed, folder_path, model_name, [default_prompt_name, image_prompt_name] , '')
        test.run_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        run_experiments(DATA_PATH, REPO_PATH)