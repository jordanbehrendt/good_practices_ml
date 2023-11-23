import pandas as pd
import os
import argparse
import yaml
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(REPO_PATH: str, experiment_name: str):
    experiment_dir = os.path.join(REPO_PATH, "Experiments", experiment_name)
    experiment_accuracies = []
    dataset_names = []
    for dir in os.listdir(experiment_dir):
        dataset_names.append(dir)
        dataset_dir = os.path.join(experiment_dir, dir)
        dataset_accuracies = []
        for batch_file in os.listdir(dataset_dir):
            batch_df = pd.read_csv(os.path.join(dataset_dir, batch_file))
            dataset_accuracies.append(metrics.accuracy_score(batch_df["label"].to_list(), batch_df["predicted labels"].to_list()))
        experiment_accuracies.append(dataset_accuracies)
    
    exp_dict = {}
    for dataset_name, exp_acc in zip(dataset_names, experiment_accuracies):
        exp_dict[dataset_name] = exp_acc

    exp_df = pd.DataFrame(exp_dict)
    sns.boxplot(data=exp_df)
    plt.title(f"Boxplot of Accuracies for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(experiment_dir, f"{experiment_name}_boxplot.png"))
        



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