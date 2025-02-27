# -*- coding: utf-8 -*-
"""
utils.confusion_matrix
----------------------

Module containing methods to create confusion matrices
for country and region level from lists of
true and predicted country labels.
"""
# Imports
# Built-in
import os
import ast

# Local

# 3r-party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _make_confusion_matrix_figure(df, name, save_path, normalize):
    fig, ax = plt.subplots(figsize=(120, 90))
    sns.set_theme(font_scale=8)
    ax = sns.heatmap(
        df,
        cmap=sns.cubehelix_palette(as_cmap=True),
        xticklabels=1,
        yticklabels=1
    )
    ax.tick_params(axis='both', labelsize=15)
    ax.set(xlabel=None, ylabel=None)
    ax.figure.savefig(f'{save_path}/{name}_confusion_matrix.png')
    fig.clf()


def _make_confusion_matrix(true_y, pred_y, normalize, ordering_index, labels):

    cf_matrix = confusion_matrix(
        true_y,
        pred_y,
        labels=range(0, 211),
        normalize='true' if normalize else None
    )

    ordered_index = np.argsort(-cf_matrix.diagonal())
    ordered_matrix = cf_matrix[ordered_index][:, ordered_index]
    regionally_ordered_matrix = cf_matrix[
        ordering_index
    ][
        :, ordering_index
    ]
    ordered_classes = labels[ordered_index]
    super_ordered_classes = labels[ordering_index]

    alphabetical_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels)
    sorted_df_cm = pd.DataFrame(
        ordered_matrix, index=ordered_classes, columns=ordered_classes)
    super_ordered_df_cm = pd.DataFrame(
        regionally_ordered_matrix,
        index=super_ordered_classes,
        columns=super_ordered_classes
    )

    return alphabetical_cm, sorted_df_cm, super_ordered_df_cm


def create_and_save_confusion_matrices(
    REPO_PATH, SAVE_FIGURES_PATH,
    true_countries, predicted_countries,
    normalize=False
):
    """
    Create and save confusion matrices for countries and regions.

    Args:
        REPO_PATH (str): path to repo folder.
        SAVE_FIGURES_PATH (str): path to save the confusion matrices.
        true_countries (list): list of true country labels.
        predicted_countries (list): list of predicted country labels.
        normalize (bool): whether to normalize the confusion matrices or not.

    Returns:
        None
    """

    # Load country list and regional ordering index
    country_list = pd.read_csv(
        f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv'
    )
    regional_ordering_index = [
        8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101,
        105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140,
        146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163,
        166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192,
        20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204,
        78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72,
        196, 13, 50, 55, 59, 62, 65, 69, 86, 88, 92, 94, 113, 115, 142, 168,
        172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147,
        157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30,
        138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57,
        71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22,
        37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21,
        47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10,
        18, 36, 49, 63, 110, 114, 118, 141
    ]
    continent_ordering_index = [
        11, 5, 10, 17, 20, 2, 3, 15, 12, 0, 4, 6, 16, 18,
        21, 7, 13, 19, 22, 1, 8, 9, 14
    ]

    # constant for classes
    classes = np.array(country_list['Country'])

    # Build country confusion matrices
    (
        df_cm, ordered_df_cm, regionally_ordered_df_cm
    ) = _make_confusion_matrix(
        true_countries,
        predicted_countries,
        normalize,
        regional_ordering_index,
        classes
    )

    # Create region labels
    np_regions = np.sort(
        country_list['Intermediate Region Name']
        .unique()
        .tolist()
    )
    true_regions = [
        ast.literal_eval(
            country_list.iloc[country]["One Hot Region"]
        ).index(1)
        for country in true_countries
    ]
    predicted_regions = [
        ast.literal_eval(
            country_list.iloc[country]["One Hot Region"]
        ).index(1)
        for country in predicted_countries
    ]

    # Build region confusion matrices
    (
        regions_df_cm, regions_ordered_df_cm, continent_ordered_df_cm
    ) = _make_confusion_matrix(
        true_regions,
        predicted_regions,
        normalize,
        continent_ordering_index,
        np_regions
    )

    # Save confusion matrices
    if normalize:
        save_path = f'{SAVE_FIGURES_PATH}/normalized'
    else:
        save_path = SAVE_FIGURES_PATH
    os.makedirs(save_path, exist_ok=True)

    dfs = {
        'alphabetical': df_cm,
        'ordered': ordered_df_cm,
        'regional_ordered': regionally_ordered_df_cm,
        'alphabetical_regions': regions_df_cm,
        'ordered_regions': regions_ordered_df_cm,
        'continent_ordered_regions': continent_ordered_df_cm
    }

    for name, df in dfs.items():
        _make_confusion_matrix_figure(df, name, save_path, normalize)
