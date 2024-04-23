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
from scripts import pdf, geo_metrics
import argparse
import yaml
import ast
from sklearn.metrics import confusion_matrix

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

    # plt.title(f"{plot_type.capitalize()} plot of {metric} for experiment {experiment_name}")
    # plt.xlabel("Model")
    plt.ylabel(f"{metric} metric")
    
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
    if metric not in ['country_acc', 'region_acc', 'mixed']:
        raise Exception(f'The metric {metric} is not known. Instead use either country_acc, region_acc or mixed')
    output_dir = os.path.join(repo_path, 'CLIP_Experiment/analysis/', comparison_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seed = 1234
    metric_dict = geo_metrics.calculate_experiment_metric(repo_path, exp_dirs, exp_names, metric)
    create_evaluation_report(metric_dict, metric, seed, paired_ttest, output_dir, comparison_name)

def create_confusion_matrices(REPO_PATH):
        """
        Creates and visualizes the confusion matrix for country and region predictions.

        Args:
            true_countries (list): List of true country labels.
            predicted_countries (list): List of predicted country labels.

        Returns:
            None
        """
        country_list = pd.read_csv(f'{REPO_PATH}/data_finding/country_list_region_and_continent.csv')
        regional_ordering_index = [8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101, 105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140, 146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163, 166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192, 20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204, 78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72, 196, 13, 50, 55, 59, 62, 65, 69,
                                        86, 88, 92, 94, 113, 115, 142, 168, 172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147, 157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30, 138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57, 71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22, 37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21, 47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10, 18, 36, 49, 63, 110, 114, 118, 141]
        create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'geoguessr')
        create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'aerial')
        create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, 'tourist')

def create_confusion_matrix(REPO_PATH, country_list, regional_ordering_index, dataset_name):
              
        # constant for classes
        classes = country_list['Country']
        np_classes = np.array(classes)

        # Create true_countries and predicted_countries lists
        dataset_df = load_data(REPO_PATH, dataset_name)
        country_labels = dataset_df["label"]
        true_countries = []
        for elem in country_labels:
            true_countries.append(country_list.index[country_list['Country'] == elem].tolist()[0])
        dataset_df['Probs-Array'] = dataset_df['All-Probs'].apply(lambda x: ast.literal_eval(x))
        dataset_df['Predicted labels'] = dataset_df['Probs-Array'].apply(lambda x: np.argmax(np.array(x)))
        predicted_countries = dataset_df['Predicted labels']

        # Build country confusion matrix
        cf_matrix = confusion_matrix(true_countries, predicted_countries, labels=range(0, 211))
        ordered_index = np.argsort(-cf_matrix.diagonal())
        ordered_matrix = cf_matrix[ordered_index][:, ordered_index]

        regionally_ordered_matrix = cf_matrix[regional_ordering_index][:,regional_ordering_index]

        ordered_classes = np_classes[ordered_index]
        regionally_ordered_classes = np_classes[regional_ordering_index]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
        ordered_df_cm = pd.DataFrame(
            ordered_matrix, index=ordered_classes, columns=ordered_classes)
        regionally_ordered_df_cm = pd.DataFrame(
            regionally_ordered_matrix, index=regionally_ordered_classes, columns=regionally_ordered_classes)

        np_regions = np.sort(np.array(list(set(country_list['Intermediate Region Name']))))

        # Build region confusion matrix
        true_regions = []
        predicted_regions = []
        for i in range(0, len(true_countries)):
            true_regions.append(ast.literal_eval(country_list.iloc[true_countries[i]]["One Hot Region"]).index(1))
            predicted_regions.append(ast.literal_eval(country_list.iloc[predicted_countries[i]]["One Hot Region"]).index(1))

        regions_cf_matrix = confusion_matrix(
            true_regions, predicted_regions, labels=range(0, len(np_regions)))
        regions_ordered_index = np.argsort(-regions_cf_matrix.diagonal())
        regions_ordered_matrix = regions_cf_matrix[regions_ordered_index][:,
                                                                          regions_ordered_index]

        ordered_regions = np_regions[regions_ordered_index]

        regions_df_cm = pd.DataFrame(regions_cf_matrix, index=np_regions, columns=np_regions)
        regions_ordered_df_cm = pd.DataFrame(regions_ordered_matrix, index=ordered_regions, columns=ordered_regions)


        if not os.path.exists(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/'):
            os.makedirs(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/')

        plt.figure(1, figsize=(120, 70))
        figure = sns.heatmap(df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
        figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/simple_confusion_matrix.png')
        plt.figure(2, figsize=(120, 70))
        ordered_figure = sns.heatmap(ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
        ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/ordered_confusion_matrix.png')
        plt.figure(3, figsize=(120, 70))
        regionally_ordered_figure = sns.heatmap(regionally_ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
        regionally_ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regionally_ordered_confusion_matrix.png')
        plt.figure(4, figsize=(120, 70))
        regions_figure = sns.heatmap(regions_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
        regions_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regions_confusion_matrix.png')
        plt.figure(5, figsize=(120, 70))
        regions_ordered_figure = sns.heatmap(regions_ordered_df_cm, cmap=sns.cubehelix_palette(as_cmap=True)).get_figure()
        regions_ordered_figure.savefig(f'{REPO_PATH}/CLIP_Experiment/confusion_matrices/{dataset_name}/regions_ordered_confusion_matrix.png')
        return

def load_data(REPO_PATH, dataset_name):
    # Directory containing CSV files
    directory = f'{REPO_PATH}/CLIP_Experiment/clip_results/image_from_prompt/{dataset_name}'

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

def run_analysis_of_experiments(REPO_PATH):
    experiment_dirs = [
        f"{REPO_PATH}/CLIP_Experiment/clip_results/default_prompt/geoguessr",
        f"{REPO_PATH}/CLIP_Experiment/clip_results/default_prompt/tourist",
        f"{REPO_PATH}/CLIP_Experiment/clip_results/default_prompt/aerial",
        f"{REPO_PATH}/CLIP_Experiment/clip_results/image_from_prompt/geoguessr",
        f"{REPO_PATH}/CLIP_Experiment/clip_results/image_from_prompt/tourist",
        f"{REPO_PATH}/CLIP_Experiment/clip_results/image_from_prompt/aerial",
    ]

    # Run comparison of prompts for each dataset
    main(REPO_PATH, [experiment_dirs[0], experiment_dirs[3]], ['default prompt', 'extended prompt'], 'mixed', True, 'geoguessr_prompts_mixed')
    main(REPO_PATH, [experiment_dirs[1], experiment_dirs[4]], ['default prompt', 'extended prompt'], 'mixed', True, 'tourist_prompts_mixed')
    main(REPO_PATH, [experiment_dirs[2], experiment_dirs[5]], ['default prompt', 'extended prompt'], 'mixed', True, 'aerial_prompts_mixed')

    # # Run comparison of datasets for image_from_prompt
    main(REPO_PATH, [experiment_dirs[3], experiment_dirs[4], experiment_dirs[5]], ['geoguessr', 'tourist', 'aerial'], 'mixed', False, 'datasets_image_prompt_mixed')
    main(REPO_PATH, [experiment_dirs[0], experiment_dirs[1], experiment_dirs[2]], ['geoguessr', 'tourist', 'aerial'], 'mixed', False, 'datasets_default_prompt_mixed')

    create_confusion_matrices(REPO_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        run_analysis_of_experiments(REPO_PATH)
