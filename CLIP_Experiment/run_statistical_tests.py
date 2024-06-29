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

def generate_plot(df: pd.DataFrame, plot_type: str, metric: str, output_dir: str) -> None:
    """
    Generate and save a plot based on the given parameters.

    Args:
        df (pd.DataFrame): DataFrame containing data for the plot.
        plot_type (str): Type of plot ('box' or 'violin').
        metric (str): Metric for the plot.
        output_dir (str): Directory to save the plot.
    """
    if plot_type == 'box':
        sns.boxplot(data=df, showmeans=True)
    elif plot_type == 'violin':
        sns.violinplot(data=df)

    plt.ylabel(f"{metric} metric")
    
    plot_filename = f"{plot_type}_plot.jpg"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

def compute_paired_k_fold_cross_validation_t_test(results_1:list,results_2:list,k:int,r:int):
    """
    Compute paired k fold cross validation t-test in accordance with the method described in https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3

    Args:
        results_1 (list): List of accuracies for each batch in first experiment
        results_2 (list): List of accuracies for each batch in second experiment
        k (int): Number of cross validation folds
        r (int): Number of random cross-validation repetitions
    """
    num_experiments = len(results_1)
    if num_experiments != len(results_2):
        raise Exception(f'The results are not the same length.')
    if num_experiments != r*k:
        raise Exception(f'The results do not coincide with the r and k values.')
    differences = [x_i - y_i for x_i, y_i in zip(results_1, results_2)] 
    mean = sum(differences) / num_experiments
    variance = 0
    for count in range(0, num_experiments):
        variance += (differences[count] - mean)**2
    variance *= 1/(num_experiments - 1)
    t = mean / math.sqrt(variance/(num_experiments + 1/(k-1)))

    p = stats.t.sf(abs(t), num_experiments-1)
    df= num_experiments-1
    return t,p,df


def run_analysis_of_dataset_and_prompts(output_dir: str,comparison_df: pd.DataFrame, metric: str, paired_ttest: bool) -> None:
    """
    Save metrics DataFrame to a CSV file and calculate statistics (median, mean, std deviation) for each model.

    Args:
        output_dir (str): Directory to save the CSV files.
        comparison_df (str): Dataframe to be analyzed
        metric (str): Name of the metric.
        paired_ttest (bool): Whether the ttest should be a paired test or not
    """

    # Calculate median, mean, standard deviation and p-value for each model
    analysis = {
        'Model': [],
        'Median': [],
        'Mean': [],
        'Standard Deviation': [],
    }

    for prompt in comparison_df.columns:
        analysis['Model'].append(prompt)
        analysis['Median'].append(np.median(comparison_df[prompt]))
        analysis['Mean'].append(np.mean(comparison_df[prompt]))
        analysis['Standard Deviation'].append(np.std(comparison_df[prompt]))

    num_datasets = len(comparison_df.columns)

    for i in range(num_datasets):
        t_values = []
        p_values = []
        df_values = []
        for j in range(num_datasets):              
            if i > j:
                if paired_ttest:
                    t,p,df = compute_paired_k_fold_cross_validation_t_test(comparison_df.iloc[:,i].tolist(),comparison_df.iloc[:,j].tolist(),20,10)
                    t_values.append(t)
                    p_values.append(p)
                    df_values.append(df)
                else:
                    results = stats.ttest_ind(comparison_df.iloc[:,i].tolist(),comparison_df.iloc[:,j].tolist())
                    t_values.append(results.statistic)
                    p_values.append(results.pvalue)
                    df_values.append(len(comparison_df.iloc[:,i].tolist()) - 1)
            else:
                t_values.append(0.0)
                p_values.append(0.0)               
                df_values.append(0.0) 

        analysis[f'{comparison_df.columns[i]}_t_values'] = t_values
        analysis[f'{comparison_df.columns[i]}_p_values'] = p_values
        analysis[f'{comparison_df.columns[i]}_df_values'] = df_values

    # Convert the analysis dictionary to a DataFrame
    analysis_df = pd.DataFrame(analysis)

    generate_plot(comparison_df, 'box', metric, output_dir)
    generate_plot(comparison_df, 'violin', metric, output_dir)

    # Save the statistics DataFrame to a CSV file
    analysis_df.to_csv(f'{output_dir}/analysis.csv', index=False)


def run_analysis_of_experiments(REPO_PATH, metric):
    """
    Generate and save a plot based on the given parameters.

    Args:
        REPO_PATH (str): path to the repo folder.
        metric (str): name of metric used
    """
    seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]

    datasets = ['geoguessr', 'aerial', 'tourist']
    default_df = pd.DataFrame()
    extended_prompt_df = pd.DataFrame()
    for dataset in datasets:
        default_experiment = pd.read_csv(f"{REPO_PATH}/CLIP_Experiment/result_accuracy/{metric}/default_prompt/{dataset}.csv")
        extended_prompt_experiment = pd.read_csv(f"{REPO_PATH}/CLIP_Experiment/result_accuracy/{metric}/extended_prompt/{dataset}.csv")
        default_list = []
        extended_prompt_list = []
        for seed in seeds:
            default_list += list(default_experiment[f'{seed}'])
            extended_prompt_list += list(extended_prompt_experiment[f'{seed}'])
        comparison_df = pd.DataFrame({'default_prompt': default_list, 'extended_prompt': extended_prompt_list})
        default_df[f'{dataset}'] = default_list
        extended_prompt_df[f'{dataset}'] = extended_prompt_list
        output_dir = f'{REPO_PATH}/CLIP_Experiment/statistical_tests/{metric}/{dataset}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        run_analysis_of_dataset_and_prompts(output_dir,comparison_df, metric, True)
    
    default_output_dir = f'{REPO_PATH}/CLIP_Experiment/statistical_tests/{metric}/default_prompt/'
    extended_prompt_output_dir = f'{REPO_PATH}/CLIP_Experiment/statistical_tests/{metric}/extended_prompt/'
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    if not os.path.exists(extended_prompt_output_dir):
        os.makedirs(extended_prompt_output_dir)
    run_analysis_of_dataset_and_prompts(default_output_dir,default_df, metric, False)
    run_analysis_of_dataset_and_prompts(extended_prompt_output_dir,extended_prompt_df, metric, False)

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
        run_analysis_of_experiments(REPO_PATH, 'Mixed_F1')
