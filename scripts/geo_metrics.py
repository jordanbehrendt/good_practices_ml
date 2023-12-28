from typing import List
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import ast
import sys
sys.path.append('.')

def calculate_country_accuracy(batch_df: pd.DataFrame) -> float:
    """
    Calculate accuracy score based on country labels.

    Args:
        batch_df (pd.DataFrame): DataFrame containing 'label' and 'Predicted labels'.

    Returns:
        float: Accuracy score based on country labels.
    """
    return metrics.accuracy_score(batch_df["label"].tolist(), batch_df["Predicted labels"].tolist())

def calculate_region_accuracy(repo_path: str, batch_df: pd.DataFrame) -> float:
    """
    Calculate accuracy score based on region labels.

    Args:
        repo_path (str): Path to repository.
        batch_df (pd.DataFrame): DataFrame containing 'label' and 'Predicted labels'.

    Returns:
        float: Accuracy score based on region labels.
    """
    unsd_regions = pd.read_csv(os.path.join(repo_path, 'data', 'UNSD_Methodology.csv'))[['Intermediate Region Name','Country or Area']]
    merged_df = pd.merge(batch_df, unsd_regions, left_on='Predicted labels', right_on='Country or Area', how='inner')
    merged_df = merged_df.rename(columns={'Intermediate Region Name': 'predicted_region'})
    del merged_df['Country or Area']
    merged_df = pd.merge(merged_df, unsd_regions, left_on='label', right_on='Country or Area', how='inner')
    merged_df = merged_df.rename(columns={'Intermediate Region Name': 'reference_region'})
    del merged_df['Country or Area']
    return metrics.accuracy_score(merged_df["reference_region"].tolist(), merged_df["predicted_region"].tolist())

def calculate_metric(repo_path: str, batch_df: pd.DataFrame, metric_name: str) -> float:
    """
    Calculate specified metric.

    Args:
        repo_path (str): Path to repository.
        batch_df (pd.DataFrame): DataFrame containing 'label' and 'Predicted labels'.
        metric_name (str): Name of the metric to be calculated ('country_acc', 'region_acc' or 'mixed').

    Returns:
        float: Calculated metric value.
    """
    country_list = pd.read_csv('/home/kieran/Documents/Uni/WiSe23-24/Good_Practices_of_Machine_Learning/good_practices_ml/data_finding/country_list.csv')
    batch_df['Probs-Array'] = batch_df['All-Probs'].apply(lambda x: ast.literal_eval(x))
    batch_df['Predicted labels'] = batch_df['Probs-Array'].apply(lambda x: country_list['Country'].iloc[np.argmax(np.array(x))])

    if metric_name == 'country_acc':
        return calculate_country_accuracy(batch_df)
    elif metric_name == 'region_acc':
        return calculate_region_accuracy(repo_path, batch_df)
    elif metric_name == 'mixed':
        return calculate_country_accuracy(batch_df) * 0.5 + calculate_region_accuracy(repo_path, batch_df) * 0.5
    else:
        raise ValueError(f"The metric {metric_name} is not known.")
    
def calculate_experiment_metric(repo_path: str, exp_dirs: List[str], exp_names: List[str], metric_name: str) -> dict:
    """
    Calculate metric for each experiment directory.

    Args:
        repo_path (str): Path to repository.
        exp_dirs (List[str]): List of experiment directories.
        exp_names (List[str]): List of experiment names.
        metric_name (str): Name of the metric to be calculated ('country_acc', 'region_acc' or 'mixed').

    Returns:
        dict: Dictionary with experiment names as keys and corresponding metric values as lists.
    """
    metrics = []
    for name, dir_path in zip(exp_names, exp_dirs):
        exp_metric = []
        for batch_file in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, batch_file)):
                batch_df = pd.read_csv(os.path.join(dir_path, batch_file))
                exp_metric.append(calculate_metric(repo_path, batch_df, metric_name))
        metrics.append(exp_metric)

    metric_dict = {exp_name: metric for exp_name, metric in zip(exp_names, metrics)}
    return metric_dict
