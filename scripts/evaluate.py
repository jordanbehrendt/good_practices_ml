import pandas as pd
import os
import argparse
import yaml
from sklearn import metrics
from typing import List
from collections.abc import Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import numpy as np
from deepsig import aso

def box_plot(df: pd.DataFrame, metric: str, experiment_name: str, experiment_dir: str):
    sns.boxplot(data=df, showmeans=True)
    plt.title(f"Boxplot of {metric} for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel(f"{metric}")
    plt.savefig(os.path.join(experiment_dir, f"{experiment_name}_{metric}_boxplot.png"))
    plt.close()

def violin_plot(df: pd.DataFrame, metric: str, experiment_name: str, experiment_dir: str):
    sns.violinplot(data = df)
    plt.title(f"Violinplot of {metric} for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel(f"{metric}")
    plt.savefig(os.path.join(experiment_dir, f"{experiment_name}_{metric}_violinplot.png"))
    plt.close()

def ttest(accs, metric: str, seed: int):
    p_values = np.zeros((3,3))
    num_datasets = len(accs)
    for i in range(num_datasets):
        for j in range(num_datasets):
            p_values[i][j] = stats.ttest_ind(accs[i], accs[j], random_state=seed).pvalue
    print(f'{metric} {p_values}')

def aso_test(accs, metric: str, seed: int):
    min_eps_values = np.zeros((3,3))
    num_datasets = len(accs)
    for i in range(num_datasets):
        for j in range(num_datasets):
            min_eps_values[i][j] = aso(accs[i], accs[j], seed=seed)
    print(f'{metric} {min_eps_values}')

def evaluate(REPO_PATH: str, experiment_name: str):
    seed = 1234

    unsd_regions = pd.read_csv(os.path.join(REPO_PATH, 'data', 'UNSD_Methodology.csv'))[['Intermediate Region Name','Country or Area']]
    experiment_dir = os.path.join(REPO_PATH, "Experiments", experiment_name)
    experiment_accuracies = []
    experiment_region_accuracies = []
    dataset_names = []
    for dir in os.listdir(experiment_dir):

        dataset_dir = os.path.join(experiment_dir, dir)
        if os.path.isdir(dataset_dir):
            dataset_accuracies = []
            dataset_region_accuracies = []
            dataset_names.append(dir)
            for batch_file in os.listdir(dataset_dir):
                if not os.path.isdir(os.path.join(dataset_dir, batch_file)):
                    batch_df = pd.read_csv(os.path.join(dataset_dir, batch_file))
                    if dir == 'street_prompt':
                        batch_df['Predicted labels'] = batch_df['Predicted labels'].str.replace('A google streetview image from This image shows the country ', '')
                    elif dir == 'elab_prompt':
                        batch_df['Predicted labels'] = batch_df['Predicted labels'].str.replace('This image shows the country ', '')
                    batch_df = pd.merge(batch_df, unsd_regions, left_on='Predicted labels', right_on='Country or Area', how='inner')
                    batch_df = batch_df.rename(columns={'Intermediate Region Name': 'predicted_region'})
                    del batch_df['Country or Area']
                    batch_df = pd.merge(batch_df, unsd_regions, left_on='label', right_on='Country or Area', how='inner')
                    batch_df = batch_df.rename(columns={'Intermediate Region Name': 'reference_region'})
                    del batch_df['Country or Area']
                    dataset_accuracies.append(metrics.accuracy_score(batch_df["label"].to_list(), batch_df["Predicted labels"].to_list()))
                    dataset_region_accuracies.append(metrics.accuracy_score(batch_df["reference_region"].to_list(), batch_df["predicted_region"].to_list()))
            experiment_accuracies.append(dataset_accuracies)
            experiment_region_accuracies.append(dataset_region_accuracies)
    
    exp_dict = {}
    for dataset_name, exp_acc in zip(dataset_names, experiment_accuracies):
        exp_dict[dataset_name] = exp_acc
    exp_df = pd.DataFrame(exp_dict)



    exp_region_dict = {}
    for dataset_name, exp_region_acc in zip(dataset_names, experiment_region_accuracies):
        exp_region_dict[dataset_name] = exp_region_acc
    exp_region_df = pd.DataFrame(exp_region_dict)
    region_metric = 'Region Accuracy'
    box_plot(exp_region_df, region_metric, experiment_name, experiment_dir)
    violin_plot(exp_region_df, region_metric, experiment_name, experiment_dir)
    ttest(experiment_region_accuracies, region_metric, seed)
    aso_test(experiment_region_accuracies, region_metric, seed)

    def model_performance(probs: List[float], labels: List[str], possible_captions: List):
        """Saves the conifdence of the best prediction with the predicted an correct label on a batch.

        Args:
            probs (List[float]): The probabilities for each class of each sample in the batch.
            labels (List[str]): The groud trouth labels for each sample in the batch.
            possible_captions (List): The list of all possible captions (labels).

        Returns:
            pd.DataFrame: A named DataFrame, containing the
        """
        max_index = probs.argmax(axis=1)  # Finding the index of the maximum probability for each sample
        max_probabilities = probs[range(probs.shape[0]), max_index]
        predicted_label = possible_captions[max_index]

        performance_data = pd.DataFrame({
            'Probabilities': max_probabilities,
            'predicted labels': predicted_label,
            'label' : labels
        })
        return performance_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        evaluate(REPO_PATH, "prompt_compare")