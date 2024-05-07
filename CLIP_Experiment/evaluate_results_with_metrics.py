import pandas as pd
import sys
sys.path.append('.')
from utils import geo_metrics
import argparse
import yaml
import os

def calculate_accuracies(REPO_PATH, seed, dataset, prompt, metric) -> list:

    experiment_dir = f"{REPO_PATH}/CLIP_Experiment/clip_results/seed_{seed}/{prompt}/{dataset}"
    return geo_metrics.calculate_experiment_metric(REPO_PATH, experiment_dir, metric)


def calculate_metrics(REPO_PATH):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to the repo folder.
    """

    seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]
    datasets = ['geoguessr', 'tourist', 'aerial']
    prompts = ['default_prompt', 'image_from_prompt']
    metrics = ['country_acc', 'region_acc', 'mixed']
    for metric in metrics:
        for prompt in prompts:
            for dataset in datasets:
                df = pd.DataFrame()
                for seed in seeds:
                    result_list = calculate_accuracies(REPO_PATH, seed, dataset, prompt, metric)
                    df[f'{seed}'] = result_list
                output_dir = f'{REPO_PATH}/CLIP_Experiment/result_accuracy/{metric}/{prompt}'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                df.to_csv(f'{output_dir}/{dataset}.csv')

if __name__ == "__main__":
    """
    Calculate batch accuracies with the 3 different metrics
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        calculate_metrics(REPO_PATH)

