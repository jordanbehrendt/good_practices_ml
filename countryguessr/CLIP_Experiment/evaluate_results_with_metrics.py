# -*- coding: utf-8 -*-
"""
CLIP_Experiment.evaluate_results_with_metrics
---------------------------------------------

Script to compute evaluation metrics for the different combinations of seed,
prompt type and dataset.

:copyright: Cognitive Systems Lab, 2025
"""
# Imports
# Built-in
import os
import argparse

# Local
from countryguessr.utils import geo_metrics

# 3r-party
import yaml
import pandas as pd


def calculate_accuracies(REPO_PATH, seed, dataset, prompt, metric) -> list:

    experiment_dir = f"{REPO_PATH}/CLIP_Experiment/clip_results/seed_{seed}/{prompt}/{dataset}"  # noqa: E501
    return geo_metrics.calculate_experiment_metric(
        REPO_PATH, experiment_dir, metric
    )


def calculate_metrics(REPO_PATH):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to the repo folder.
    """

    seeds = [4808, 4947, 5723, 3838, 5836, 3947, 8956, 5402, 1215, 8980]
    datasets = ['geoguessr', 'tourist', 'aerial']
    prompts = ['default_prompt', 'extended_prompt']
    # metrics = ['country_acc', 'region_acc']
    for prompt in prompts:
        for dataset in datasets:
            # for metric in metrics:
            #     df = pd.DataFrame()
            #     for seed in seeds:
            #         result_list = calculate_accuracies(
            #             REPO_PATH, seed, dataset, prompt, metric
            #         )
            #         df[f'{seed}'] = result_list
            #     output_dir = f'{REPO_PATH}/CLIP_Experiment/result_accuracy/{metric}/{prompt}'  # noqa E501
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     df.to_csv(f'{output_dir}/{dataset}.csv')
            pre_df = pd.DataFrame()
            rec_df = pd.DataFrame()
            f1_df = pd.DataFrame()
            for seed in seeds:
                result_list = calculate_accuracies(
                    REPO_PATH, seed, dataset, prompt, 'mixed'
                )
                pre_df[f'{seed}'] = result_list[:, 0]
                rec_df[f'{seed}'] = result_list[:, 1]
                f1_df[f'{seed}'] = result_list[:, 2]
                pre_output_dir = f'{REPO_PATH}/CLIP_Experiment/result_accuracy/Mixed_Precision/{prompt}'  # noqa: E501
                if not os.path.exists(pre_output_dir):
                    os.makedirs(pre_output_dir)
                pre_df.to_csv(f'{pre_output_dir}/{dataset}.csv')
                rec_output_dir = f'{REPO_PATH}/CLIP_Experiment/result_accuracy/Mixed_Recall/{prompt}'  # noqa: E501
                if not os.path.exists(rec_output_dir):
                    os.makedirs(rec_output_dir)
                rec_df.to_csv(f'{rec_output_dir}/{dataset}.csv')
                f1_output_dir = f'{REPO_PATH}/CLIP_Experiment/result_accuracy/Mixed_F1/{prompt}'  # noqa: E501
                if not os.path.exists(f1_output_dir):
                    os.makedirs(f1_output_dir)
                f1_df.to_csv(f'{f1_output_dir}/{dataset}.csv')


if __name__ == "__main__":
    """
    Calculate batch accuracies with the 3 different metrics
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument(
        '--yaml_path',
        metavar='str',
        required=True,
        help='The path to the yaml file with the stored paths'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        required=False,
        help='Enable debug mode',
        default=False
    )
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path']
        calculate_metrics(REPO_PATH)
