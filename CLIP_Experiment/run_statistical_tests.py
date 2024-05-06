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
import math

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

    plt.ylabel(f"{metric} metric")
    
    plot_filename = f"{experiment_name}_{metric}_{plot_type}plot.jpg"
    plt.savefig(os.path.join(experiment_dir, plot_filename))
    plt.close()

def compute_paired_k_fold_cross_validation_t_test(results_1:list,results_2:list,k:int,r:int):
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
    if len(results_1) != len(results_2):
        raise Exception(f'The results are not the same length.')
    if len(results_1) != r*k:
        print(len(results_1))
        raise Exception(f'The results do not coincide with the r and k values.')
    differences = [x_i - y_i for x_i, y_i in zip(results_1, results_2)] 
    mean = sum(differences) / (k*r)
    variance = 0
    for count in range(0, len(differences)):
        variance += (differences[count] - mean)**2
    variance *= 1/(k*r - 1)
    t = mean / math.sqrt(variance/(k*r - 1))
    return t


# def compute_test_values(accs: dict, metric: str, test_type: str, seed: int, experiment_name: str, experiment_dir: str, paired_ttest: bool = False) -> None:
#     """
#     Compute statistical test values and save them to a CSV file.

#     Args:
#         accs (dict): Dictionary containing accuracy values for models.
#         metric (str): Metric for computation.
#         test_type (str): Type of statistical test ('ttest' or 'aso_test').
#         seed (int): Seed for random number generation.
#         experiment_name (str): Name of the experiment.
#         experiment_dir (str): Directory to save the test results CSV file.
#         paired_ttest (bool): Whether batches are paired.
#     """
#     model_names = list(accs.keys())
#     accs = list(accs.values())
#     num_datasets = len(accs)
#     result_values = np.zeros((num_datasets, num_datasets))

#     for i in range(num_datasets):
#         for j in range(num_datasets):
#             if test_type == 'ttest':
#                 if paired_ttest:
#                     result_values[i][j] = stats.ttest_rel(accs[i], accs[j]).pvalue
#                 else:
#                     result_values[i][j] = stats.ttest_ind(accs[i], accs[j], random_state=seed).pvalue
#             elif test_type == 'aso_test':
#                 result_values[i][j] = aso(accs[i], accs[j], seed=seed)
#             else:
#                 raise ValueError("Invalid computation type. Choose 'ttest' or 'aso_test'.")

#     test_df = pd.DataFrame(result_values, columns=model_names, index=model_names)
#     test_df.to_csv(os.path.join(experiment_dir, f'{experiment_name}_{test_type}.csv'))

def run_analysis_of_dataset_and_prompts(output_dir: str,comparison_df: pd.DataFrame) -> None:
    """
    Save metrics DataFrame to a CSV file and calculate statistics (median, mean, std deviation) for each model.

    Args:
        metrics (pd.DataFrame): DataFrame with accuracy scores for multiple models.
        metric (str): Name of the metric.
        experiment_name (str): Name of the experiment.
        experiment_dir (str): Directory to save the CSV files.
    """

    # Calculate median, mean, standard deviation and p-value for each model
    stats = {
        'Model': [],
        'Median': [],
        'Mean': [],
        'Standard Deviation': [],
        't-test-value': []
    }

    for prompt in comparison_df.columns:
        stats['Model'].append(prompt)
        stats['Median'].append(np.median(comparison_df[prompt]))
        stats['Mean'].append(np.mean(comparison_df[prompt]))
        stats['Standard Deviation'].append(np.std(comparison_df[prompt]))
        stats['t-test-value'].append(compute_paired_k_fold_cross_validation_t_test(comparison_df['default_prompt'],comparison_df['image_from_prompt'],20, 10))

    # Convert the stats dictionary to a DataFrame
    stats_df = pd.DataFrame(stats)

    # Save the statistics DataFrame to a CSV file
    stats_df.to_csv(f'{output_dir}/stats.csv', index=False)

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
    run_analysis_of_dataset_and_prompts(accs_df, metric, comparison_name, output_dir)
    generate_plot(accs_df, 'box', metric, comparison_name, output_dir)
    generate_plot(accs_df, 'violin', metric, comparison_name, output_dir)
    # compute_test_values(accs, metric, 'ttest', seed, comparison_name, output_dir, paired_ttest)
    # compute_test_values(accs, metric, 'aso_test', seed, comparison_name, output_dir)
    # pdf.create_merged_pdf(output_dir, comparison_name)

# def main(REPO_PATH: str, exp_dirs: List[str], exp_names: List[str], metric: str, paired_ttest: bool, comparison_name: str, seed: int) -> None:
#     """mixed
#     Main function to perform evaluation.

#     Args:
#         REPO_PATH (str): Path to repository.
#         exp_dirs (List[str]): List of experiment directories.
#         exp_names (List[str]): List of experiment names.
#         metric (str): Metric for evaluation ('country_acc', 'region_acc' or 'mixed).
#         paired_ttest (bool): Whether batches are paired.
#         comparison_name (str): Custom name for comparison.
#     """
#     if len(exp_dirs) != len(exp_names):
#         raise Exception(f'Length of experiment directories ({len(exp_dirs)}) is not the same as length of experiment names ({len(exp_names)}).')
#     if metric not in ['country_acc', 'region_acc', 'mixed']:
#         raise Exception(f'The metric {metric} is not known. Instead use either country_acc, region_acc or mixed')
#     output_dir = os.path.join(REPO_PATH, 'CLIP_Experiment/analysis/', comparison_name)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     metric_dict = geo_metrics.calculate_experiment_metric(REPO_PATH, exp_dirs, exp_names, metric)
#     create_evaluation_report(metric_dict, metric, seed, paired_ttest, output_dir, comparison_name)

def run_analysis_of_experiments(REPO_PATH, metric):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to the repo folder.
    """
    seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]

    datasets = ['geoguessr', 'aerial', 'tourist']
    for dataset in datasets:
        default_experiment = pd.read_csv(f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/default_prompt/{dataset}.csv")
        image_from_experiment = pd.read_csv(f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/image_from_prompt/{dataset}.csv")
        default_list = []
        image_from_list = []
        for seed in seeds:
            default_list += list(default_experiment[f'{seed}'])
            image_from_list += list(image_from_experiment[f'{seed}'])
        comparison_df = pd.DataFrame({'default_prompt': default_list, 'image_from_prompt': image_from_list})
        output_dir = f'{REPO_PATH}/CLIP_Experiment/statistical_tests/{metric}/{dataset}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        run_analysis_of_dataset_and_prompts(output_dir,comparison_df)

    # experiment_dirs = [
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/default_prompt/geoguessr",
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/default_prompt/tourist",
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/default_prompt/aerial",
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/image_from_prompt/geoguessr",
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/image_from_prompt/tourist",
    #     f"{REPO_PATH}/CLIP_Experiment/metrics/{metric}/image_from_prompt/aerial",
    # ]

    # # Run comparison of datasets for image_from_prompt
    # main(REPO_PATH, [experiment_dirs[3], experiment_dirs[4], experiment_dirs[5]], ['geoguessr', 'tourist', 'aerial'], 'mixed', False, 'datasets_image_prompt_mixed', seed)
    # main(REPO_PATH, [experiment_dirs[0], experiment_dirs[1], experiment_dirs[2]], ['geoguessr', 'tourist', 'aerial'], 'mixed', False, 'datasets_default_prompt_mixed', seed)

if __name__ == "__main__":
    """
    Run statistical analysis and save confusion matrices
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        run_analysis_of_experiments(REPO_PATH, 'mixed')
