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
import pdf
import geo_metrics
import argparse
import yaml

def generate_plot(df: pd.DataFrame, plot_type: str, metric: str, experiment_name: str, experiment_dir: str) -> None:
    """
    Generate and save a plot based on the given parameters.

    Args:
        df (pd.DataFrame): DataFrame containing data for the plot.
        plot_type (str): Type of plot ('box' or 'violin').
        metric (str): Metric for the plot.
        experiment_name (str): Name of the experiment.
        experiment_dir (str): Directory to save the plot.
    """
    if plot_type == 'box':
        sns.boxplot(data=df, showmeans=True)
    elif plot_type == 'violin':
        sns.violinplot(data=df)

    plt.title(f"{plot_type.capitalize()} plot of {metric} for experiment {experiment_name}")
    plt.xlabel("Model")
    plt.ylabel(f"{metric}")
    
    plot_filename = f"{experiment_name}_{metric}_{plot_type}plot.jpg"
    plt.savefig(os.path.join(experiment_dir, plot_filename))
    plt.close()

def compute_test_values(accs: dict, metric: str, test_type: str, seed: int, experiment_name: str, experiment_dir: str, paired_ttest: bool = False) -> None:
    """
    Compute statistical test values and save them to a CSV file.

    Args:
        accs (dict): Dictionary containing accuracy values for models.
        metric (str): Metric for computation.
        test_type (str): Type of statistical test ('ttest' or 'aso_test').
        seed (int): Seed for random number generation.
        experiment_name (str): Name of the experiment.
        experiment_dir (str): Directory to save the test results CSV file.
        paired_ttest (bool): Whether batches are paired.
    """
    model_names = list(accs.keys())
    accs = list(accs.values())
    num_datasets = len(accs)
    result_values = np.zeros((num_datasets, num_datasets))

    for i in range(num_datasets):
        for j in range(num_datasets):
            if test_type == 'ttest':
                if paired_ttest:
                    result_values[i][j] = stats.ttest_rel(accs[i], accs[j]).pvalue
                else:
                    result_values[i][j] = stats.ttest_ind(accs[i], accs[j], random_state=seed).pvalue
            elif test_type == 'aso_test':
                result_values[i][j] = aso(accs[i], accs[j], seed=seed)
            else:
                raise ValueError("Invalid computation type. Choose 'ttest' or 'aso_test'.")

    test_df = pd.DataFrame(result_values, columns=model_names, index=model_names)
    test_df.to_csv(os.path.join(experiment_dir, f'{experiment_name}_{test_type}.csv'))

def create_metric_csv(metrics: pd.DataFrame, metric: str, experiment_name: str, experiment_dir: str) -> None:
    """
    Save metrics DataFrame to a CSV file and calculate statistics (median, mean, std deviation) for each model.

    Args:
        metrics (pd.DataFrame): DataFrame with accuracy scores for multiple models.
        metric (str): Name of the metric.
        experiment_name (str): Name of the experiment.
        experiment_dir (str): Directory to save the CSV files.
    """
    # Save metrics DataFrame to a CSV file
    metrics.to_csv(os.path.join(experiment_dir, f'{experiment_name}_{metric}.csv'), index=False)

    # Calculate median, mean, and standard deviation for each model
    stats = {
        'Model': [],
        'Median': [],
        'Mean': [],
        'Standard Deviation': []
    }

    for column in metrics.columns:
        stats['Model'].append(column)
        stats['Median'].append(np.median(metrics[column]))
        stats['Mean'].append(np.mean(metrics[column]))
        stats['Standard Deviation'].append(np.std(metrics[column]))

    # Convert the stats dictionary to a DataFrame
    stats_df = pd.DataFrame(stats)

    # Save the statistics DataFrame to a CSV file
    stats_df.to_csv(os.path.join(experiment_dir, f'{experiment_name}_{metric}_stats.csv'), index=False)

def create_evaluation_report(accs: dict, metric: str, seed: int, paired_ttest: bool, output_dir: str, comparison_name: str) -> None:
    """
    Create an evaluation report containing plots and test results.

    Args:
        accs (dict): Dictionary containing experiment names and corresponding metric values.
        metric (str): Metric for evaluation.
        seed (int): Seed for random number generation.
        paired_ttest (bool): Whether batches are paired.
        output_dir (str): Output directory to save the report.
        comparison_name (str): Custom name for comparison.
    """
    accs_df = pd.DataFrame(accs)
    create_metric_csv(accs_df, metric, comparison_name, output_dir)
    generate_plot(accs_df, 'box', metric, comparison_name, output_dir)
    generate_plot(accs_df, 'violin', metric, comparison_name, output_dir)
    compute_test_values(accs, metric, 'ttest', seed, comparison_name, output_dir, paired_ttest)
    compute_test_values(accs, metric, 'aso_test', seed, comparison_name, output_dir)
    pdf.create_merged_pdf(output_dir, comparison_name)

def main(repo_path: str, exp_dirs: List[str], exp_names: List[str], metric: str, paired_ttest: bool, comparison_name: str) -> None:
    """
    Main function to perform evaluation.

    Args:
        repo_path (str): Path to repository.
        exp_dirs (List[str]): List of experiment directories.
        exp_names (List[str]): List of experiment names.
        metric (str): Metric for evaluation ('country_acc', 'region_acc' or 'mixed).
        paired_ttest (bool): Whether batches are paired.
        comparison_name (str): Custom name for comparison.
    """
    if len(exp_dirs) != len(exp_names):
        raise Exception(f'Length of experiment directories ({len(exp_dirs)}) is not the same as length of experiment names ({len(exp_names)}).')
    if metric not in ['country_acc', 'region_acc']:
        raise Exception(f'The metric {metric} is not known. Instead use either country_acc or region_acc')
    output_dir = os.path.join(repo_path, 'Evaluation', comparison_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seed = 1234
    metric_dict = geo_metrics.calculate_experiment_metric(repo_path, exp_dirs, exp_names, metric)
    create_evaluation_report(metric_dict, metric, seed, paired_ttest, output_dir, comparison_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('--experiment_dirs', nargs='+', metavar='str', required=True, help='At least two paths to the directories with the experiments that should be compared')
    parser.add_argument('--experiment_names', nargs='+', metavar='str', required=True, help='At least two names of the experiments that should be compared')
    parser.add_argument('--metric', metavar='str', required=True, help='Metric of the evaluation (country_acc, region_acc, mixed)')
    parser.add_argument('--paired_ttest', metavar='str', required=True, help='Information if the training batches are paired')
    parser.add_argument('--comparison_name', metavar='str', required=True, help='Custom name of the comparison')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        main(REPO_PATH, args.experiment_dirs, args.experiment_names, args.metric, bool(args.paired_ttest), args.comparison_name)
