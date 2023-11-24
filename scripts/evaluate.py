import pandas as pd
import os
import argparse
import yaml
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import numpy as np
from deepsig import aso

def evaluate(REPO_PATH: str, experiment_name: str):
    seed = 1234

    experiment_dir = os.path.join(REPO_PATH, "Experiments", experiment_name)
    experiment_accuracies = []
    dataset_names = []
    for dir in os.listdir(experiment_dir):

        dataset_dir = os.path.join(experiment_dir, dir)
        if os.path.isdir(dataset_dir):
            dataset_accuracies = []
            dataset_names.append(dir)
            for batch_file in os.listdir(dataset_dir):
                batch_df = pd.read_csv(os.path.join(dataset_dir, batch_file))
                dataset_accuracies.append(metrics.accuracy_score(batch_df["label"].to_list(), batch_df["predicted labels"].to_list()))
            experiment_accuracies.append(dataset_accuracies)
    
    exp_dict = {}
    for dataset_name, exp_acc in zip(dataset_names, experiment_accuracies):
        exp_dict[dataset_name] = exp_acc

    exp_df = pd.DataFrame(exp_dict)
    sns.boxplot(data=exp_df, showmeans=True)
    plt.title(f"Boxplot of Accuracies for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(experiment_dir, f"{experiment_name}_boxplot.png"))

    plt.close()

    sns.violinplot(data = exp_df)
    plt.title(f"Violinplot of Accuracies for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(experiment_dir, f"{experiment_name}_violinplot.png"))

    p_values = np.zeros((3,3))
    num_datasets = len(experiment_accuracies)
    for i in range(num_datasets):
        for j in range(num_datasets):
            p_values[i][j] = stats.ttest_ind(experiment_accuracies[i], experiment_accuracies[j], random_state=seed).pvalue

    print(p_values)

    min_eps_values = np.zeros((3,3))
    for i in range(num_datasets):
        for j in range(num_datasets):
            min_eps_values[i][j] = aso(experiment_accuracies[i], experiment_accuracies[j], seed=seed)

    print(min_eps_values)



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