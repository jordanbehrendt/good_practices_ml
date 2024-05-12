import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from tensorflow.python.summary.summary_iterator import summary_iterator


def corrected_repeated_kFold_cv_test(data1, data2, n1, n2, alpha):
    """
    Perform corrected repeated k-fold cross-validation test to evaluate the replicability 
    of significance tests for comparing learning algorithms.
    This implments the test as suggested in the paper 
    "A corrected repeated k-fold cross-validation test for replicability in psychophysiology"
    by Bouckaert et al. (2004).

    Parameters:
    data1 (array-like): The first dataset.
    data2 (array-like): The second dataset.
    n1 (int): The number of training samples in each fold.
    n2 (int): The number of test samples ind each fold.
    alpha (float): The significance level.

    Returns:
    tuple: A tuple containing the degrees of freedom, t-statistic, critical value and the p-value.
    """
    n = len(data1)
    if n != len(data2):
        raise ValueError("The datasets must have the same length.")
    # estimate the mean
    m = 1 / n * sum([data1[i] - data2[i] for i in range(n)])
    # estimate the standard deviation
    stdv_sq = np.sqrt(
        1 / (n - 1) * sum([(data1[i] - data2[i] - m) ** 2 for i in range(n)])
    )
    # calculate the test statistic
    t = m / np.sqrt((1 / n + n2 / n1) * stdv_sq)
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = stats.t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - stats.t.cdf(abs(t), df)) * 2.0
    return df, t, cv, p


def box_plot_experiments(list_of_df, name, save_path):
    """
    Genreates box plots for all metrics contained in the dataframes.
    Compares these metrics for each dataframe in the list.

    Parameters:
    list_of_df (list): A list of dataframes.
    name (str): The name of the experiment.
    save_path (str): The path to save the plot.

    Returns:
    pd.DataFrame: A concatenated dataframe containing all data with a coloumn tagging the used Loss.
    """

    sns.set_theme(style="whitegrid")

    for i in range(len(list_of_df)):
        list_of_df[i] = list_of_df[i].assign(Experiment=f"L{i+1}")
        list_of_df[i].columns = list_of_df[i].columns.str.split().str[-2:].str.join(" ")

    condf = pd.concat(list_of_df)
    meltdf = condf.melt(id_vars=["Experiment"], var_name="Metric", value_name="Value")
    meltdf["Value"] = meltdf["Value"].apply(lambda x: float(x[0]) if type(x) == list else x) 
    ax = sns.boxplot(
        x="Experiment", y="Value", hue="Metric", data=meltdf, showfliers=False
    )
    ax.set_title(name)
    lgd = plt.legend(bbox_to_anchor=(0.9, 0.95), loc=2, borderaxespad=0.0)
    plt.savefig(
        save_path + f"{name}-boxplot.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    return condf


def read_event_for_different_seeds(log_dir):
    """
    Read event files for different seeds and extract validation and test data.

    Parameters:
        log_dir (str): The directory containing the event files of all seeds.

    Returns:
        tuple: A tuple containing four pandas DataFrames:
            - region_columns_val: DataFrame containing validation metrics for region columns.
            - country_columns_val: DataFrame containing validation metrics for non-region columns.
            - region_columns_test: DataFrame containing test metrics for region columns.
            - country_columns_test: DataFrame containing test metrics for non-region columns.
            - other_columns: DataFrame containing other 
    """
    # Get a list of file paths that match the pattern in log_dir
    log_files = glob.glob(log_dir + "/*")
    validation_columns = pd.DataFrame([])
    test_columns = pd.DataFrame([])
    other_columns = pd.DataFrame([])
    # Iterate over the file paths to read each event file
    for file_path in log_files:
        # create a summary iterator for evenet file
        sm_iter = summary_iterator(file_path)
        # create a buffer to store the validation and test data
        validation_buffer = {}
        test_buffer = {}
        other_buffer = {}
        # Iterate over the event file values
        for e in sm_iter:
            for v in e.summary.value:
                tag = v.tag
                # filter tags that saved images or matrices
                if "Metrics" not in tag and "Matrix" not in tag:
                    # save loss and number of ignored classes/regions
                    if 'Loss' in tag:
                        if tag not in validation_buffer.keys():
                            other_buffer[tag] = []
                        other_buffer[tag].append([v.simple_value])
                    else:
                        if "Validation" in tag:
                            if tag not in validation_buffer.keys():
                                validation_buffer[tag] = []
                            if  "text_summary" in tag:
                                validation_buffer[tag].append([v.tensor.string_val])
                            else:
                                validation_buffer[tag].append([v.simple_value])
                        elif "Test" in tag:
                            if tag not in test_buffer.keys():
                                test_buffer[tag] = []
                            if  "text_summary" in tag:
                                test_buffer[tag].append([v.tensor.string_val])
                            else:
                                test_buffer[tag].append([v.simple_value])
        # Add the Values of the Last epoch to the validation data for each seed 
        # (validation has 10 folds, test data only 1 value)
        validation_columns = pd.concat(
            [
                validation_columns,
                pd.DataFrame(
                    {key: values[-10:] for key, values in validation_buffer.items()}
                ),
            ],
            ignore_index=True,
        )

        test_columns = pd.concat(
            [
                test_columns,
                pd.DataFrame({key: values[-1:] for key, values in test_buffer.items()}),
            ],
            ignore_index=True,
        )

        other_columns = pd.concat(
            [
                other_columns,
                pd.DataFrame({key: values[-1:] for key, values in other_buffer.items()}),
            ],
            ignore_index=True,
        )

    # Split validation_columns into region_columns and other_columns
    region_columns_val = validation_columns.filter(regex="Region")
    country_columns_val = validation_columns.loc[
        :, ~validation_columns.columns.str.contains("Region")
    ]

    # Split test_columns into region_columns and other_columns
    region_columns_test = test_columns.filter(regex="Region")
    country_columns_test = test_columns.loc[
        :, ~test_columns.columns.str.contains("Region")
    ]
    return (
        region_columns_val,
        country_columns_val,
        region_columns_test,
        country_columns_test,
        other_columns
    )

def event_to_df(log_dir):
    """
    Converts and merges the event files of multiple seeds into a DataFrame for all directories.
    The log_dir should be comtaim multiple folders (e.g. diffrent loss configurations) 
    that each contain event files for all the used random seeds.

    Args:
        log_dir (str): The directory path containing the event log folders.

    Returns:
        tuple: A tuple containing lists of dataframes for different columns.
            The tuple contains the following lists:
            - region_columns_val_list: List of dataframes for region metrics in validation set.
            - country_columns_val_list: List of dataframes for country columns in validation set.
            - region_columns_test_list: List of dataframes for region columns in test set.
            - country_columns_test_list: List of dataframes for country columns in test set.
            - other_coloumns_list: List of dataframes for other columns.

    """
    # Create empty lists to store the dataframes
    region_columns_val_list = []
    country_columns_val_list = []
    region_columns_test_list = []
    country_columns_test_list = []
    other_coloumns_list = []

    # Iterate over the folders in the log directory
    for folder in sorted(os.listdir(log_dir)):
        folder_path = os.path.join(log_dir, folder)
        if os.path.isdir(folder_path):
            # Call the read_event_for_different_seeds function for each folder
            region_columns_val, coutnry_columns_val, region_columns_test, coutnry_columns_test, other_coloumns = read_event_for_different_seeds(folder_path)
            
            # Append the dataframes to the respective lists
            region_columns_val_list.append(region_columns_val)
            country_columns_val_list.append(coutnry_columns_val)
            region_columns_test_list.append(region_columns_test)
            country_columns_test_list.append(coutnry_columns_test)
            other_coloumns_list.append(other_coloumns)

    return region_columns_val_list, country_columns_val_list, region_columns_test_list, country_columns_test_list, other_coloumns_list

if __name__ == "__main__":
    # Specify the path to the TensorBoard log directory
    log_dir = '/media/leon/Samsung_T5/Uni/good_practices_ml/runs/experiment1/mixed_strongly_balanced/'
    save_path = '/media/leon/Samsung_T5/Uni/good_practices_ml/runs/experiment1/mixed_strongly_balanced/'
    # Call the event_to_df function with the log directory 
    region_columns_val_list, coutnry_columns_val_list, region_columns_test_list, country_columns_test_list, other_coloumns_list = event_to_df(log_dir)
    # Call the box_plot_experiments function with the lists of dataframes
    val_region_metrics = box_plot_experiments(region_columns_val_list, 'validation-region', save_path)
    val_country_metrics = box_plot_experiments(coutnry_columns_val_list, 'validation-country', save_path)
    test_region_metric = box_plot_experiments(region_columns_test_list, 'test-region', save_path)
    test_coutnry_metric = box_plot_experiments(country_columns_test_list, 'test-country', save_path)