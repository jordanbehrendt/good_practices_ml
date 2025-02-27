# -*- coding: utf-8 -*-
"""
t-SNE.tsne
----------

Produce t-SNE representations of different datasets
with different granularity levels.
"""
# Imports
# Built-in
import os
import ast
import argparse

# Local

# 3r-party
import torch
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_european_data(REPO_PATH, dataset_name, country_list):
    """Load only european data to run european tSNE analysis

    Args:
        REPO_PATH (str): local path of repository
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
    """
    # Directory containing CSV files
    directory = f'{REPO_PATH}/CLIP_Embeddings/Image/'

    # Get a list of all filenames in each directory
    file_list = [
        file for file in os.listdir(directory)
        if file.startswith(dataset_name)
    ]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through the files, read them as DataFrames,
    # and append to the list
    for file in file_list:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    europe_countries = country_list[(country_list['Continent'] == 'Europe')]
    europe_country_list = europe_countries['Country'].tolist()
    mask = combined_df['label'].isin(europe_country_list)
    europe_combined_df = combined_df[mask]

    # map contries to regions
    combined_df = pd.merge(
        europe_combined_df,
        country_list,
        left_on='label',
        right_on='Country'
    )
    return combined_df


def load_data(REPO_PATH, dataset_name, country_list):
    """Load the data to run tSNE analysis

    Args:
        REPO_PATH (str): local path of repository
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
    """
    # Directory containing CSV files
    directory = f'{REPO_PATH}/CLIP_Embeddings/Image/'

    # Get a list of all filenames in each directory
    file_list = [
        file for file in os.listdir(directory)
        if file.startswith(dataset_name)
    ]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through the files, read them as DataFrames,
    # and append to the list
    for file in file_list:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # map contries to regions
    combined_df = pd.merge(
        combined_df,
        country_list,
        left_on='label',
        right_on='Country'
    )
    return combined_df


def transform_tensor_rep_to_array(x):
    """Transform a string representation of a torch tensor to a numpy array

    Args:
        x (str): string representation of torch tensor
    """
    start = x.find('[[')
    end = x.find(']]') + 2
    x = x[start:end]
    image_embedding = torch.tensor(ast.literal_eval(x))
    image_embedding_values = np.array(
        image_embedding
        .flatten()
        .tolist()
    ).reshape(1, -1)
    return image_embedding_values


def transform_buffer_to_array(x):
    """Transform a buffer string into a numpy list

    Args:
        x (str): buffer string
    """
    model_input_str = eval(x)
    model_input_array = np.frombuffer(model_input_str, dtype=np.float32)
    model_input = model_input_array.tolist()
    return model_input


def save_continent_plot(
    REPO_PATH, tsne_results,
    dataset_name, continents,
    continent_classes,
    include_distances
):
    """Save a plot of the t-SNE results colored by continent

    Args:
        REPO_PATH (str): local path of repository
        tsne_results (Array): Results of the t-SNE analysis
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        continents (List): list of all continents
        include_distances (Boolean): Defines whether the t-SNE
            analysis will be undertaken on the embeddings or
            embeddings appended with distances to prompts
    """
    # Display tSNE results of continents

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['Classes'] = continents

    plt.figure(figsize=(16, 10), clear=True)
    scatterplot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Classes",
        hue_order=continent_classes,
        palette=sns.color_palette("hls", 6),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    if include_distances:
        if not os.path.isdir(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Model_Input/{dataset_name}/World'
        ):
            os.makedirs(
                f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
                f'Model_Input/{dataset_name}/World'
            )
        scatterplot.figure.savefig(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Model_Input/{dataset_name}/World/output.png'
        )
    else:
        if not os.path.isdir(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Embeddings/{dataset_name}/World'
        ):
            os.makedirs(
                f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
                f'Embeddings/{dataset_name}/World'
            )
        scatterplot.figure.savefig(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Embeddings/{dataset_name}/World/output.png'
        )


def save_region_plots_europe(
    REPO_PATH, y, country_list,
    tsne_results, dataset_name,
    region_classes,
    include_distances
):
    """Save plots of the t-SNE results seperated by continent
    and colored by region

    Args:
        REPO_PATH (str): local path of repository
        y (List): List of labels for the t-SNE results
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
        tsne_results (Array): Results of the t-SNE analysis
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        region_classes (List): list of all regions
        include_distances (Boolean): Defines whether the t-SNE analysis
            will be undertaken on the embeddings or embeddings
            appended with distances to prompts
    """
    # Create intermediate region labels
    region_result_array = []
    for elem in y:
        country_row = country_list.loc[country_list['Country'] == elem].iloc[0]
        region_result_array.append(country_row['Intermediate Region Name'])

    # Display tSNE results of intermediate regions within Europe

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['Classes'] = region_result_array

    color_palette = sns.color_palette("hls", len(region_classes)).as_hex()

    plt.figure(figsize=(16, 10), clear=True)
    scatterplot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Classes",
        hue_order=region_classes,
        palette=sns.color_palette(color_palette),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    if include_distances:
        if not os.path.isdir(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Model_Input/{dataset_name}/Europe'
        ):
            os.makedirs(
                f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
                f'Model_Input/{dataset_name}/Europe'
            )
        scatterplot.figure.savefig(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Model_Input/{dataset_name}/Europe/output.png'
        )
    else:
        if not os.path.isdir(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Embeddings/{dataset_name}/Europe'
        ):
            os.makedirs(
                f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
                f'Embeddings/{dataset_name}/Europe'
            )
        scatterplot.figure.savefig(
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'Embeddings/{dataset_name}/Europe/output.png'
        )


def save_region_plots(
    REPO_PATH, y, country_list,
    tsne_results, dataset_name,
    continent_classes,
    include_distances
):
    """Save plots of the t-SNE results seperated by continent and
    colored by region

    Args:
        REPO_PATH (str): local path of repository
        y (List): List of labels for the t-SNE results
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
        tsne_results (Array): Results of the t-SNE analysis
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        continent_classes (List): list of all continents
        include_distances (Boolean): Defines whether the
            t-SNE analysis will be undertaken on the
            embeddings or embeddings appended with distances
            to prompts
    """
    # Create intermediate region labels

    continent_specific_labels = []
    for continent in continent_classes:
        result_array = []
        for elem in y:
            country_row = (
                country_list
                .loc[country_list['Country'] == elem]
                .iloc[0]
            )
            if country_row['Continent'] == continent:
                result_array.append(country_row['Intermediate Region Name'])
            else:
                result_array.append('Other')
        continent_specific_labels.append(result_array)

    # Display tSNE results of intermediate regions within each continent

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    for i in range(0, len(continent_specific_labels)):
        df_subset['Classes'] = continent_specific_labels[i]

        unique_classes = np.unique(continent_specific_labels[i])
        modified_unique = list(np.delete(
            unique_classes,
            np.where(unique_classes == 'Other')
        ))
        color_palette = sns.color_palette("hls", len(modified_unique)).as_hex()
        color_palette.append('#d3d3d3')
        modified_unique.append('Other')

        plt.figure(figsize=(16, 10), clear=True)
        scatterplot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Classes",
            hue_order=modified_unique,
            palette=sns.color_palette(color_palette),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        save_type = 'Model_Input' if include_distances else 'Embeddings'
        save_path = (
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'{save_type}/{dataset_name}/Continent/'
            f'{continent_classes[i]}'
        )

        os.makedirs(save_path, exist_ok=True)
        scatterplot.figure.savefig(os.path.join(
            save_path,
            'output.png'
        ))


def save_country_plots_europe(
    REPO_PATH, y, country_list,
    tsne_results, dataset_name,
    region_classes,
    include_distances
):
    """Save plots of the t-SNE results seperated by region
    and colored by country

    Args:
        REPO_PATH (str): local path of repository
        y (List): List of labels for the t-SNE results
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
        tsne_results (Array): Results of the t-SNE analysis
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        region_classes (List): list of all regions
        include_distances (Boolean): Defines whether the t-SNE
            analysis will be undertaken on the embeddings or
            embeddings appended with distances to prompts.
    """
    # Create country labels

    region_specific_labels = []
    for region in region_classes:
        result_array = []
        for elem in y:
            country_row = (
                country_list
                .loc[country_list['Country'] == elem]
                .iloc[0]
            )
            if country_row['Intermediate Region Name'] == region:
                result_array.append(elem)
            else:
                result_array.append('Other')
        region_specific_labels.append(result_array)

    # Display tSNE results of countries within each intermediate region

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    for i in range(0, len(region_specific_labels)):
        df_subset['Classes'] = region_specific_labels[i]

        unique_classes = np.unique(region_specific_labels[i])
        modified_unique = list(np.delete(
            unique_classes,
            np.where(unique_classes == 'Other')
        ))
        color_palette = sns.color_palette("hls", len(modified_unique)).as_hex()
        color_palette.append('#d3d3d3')
        modified_unique.append('Other')

        plt.figure(figsize=(16, 10), clear=True)
        scatterplot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Classes",
            hue_order=modified_unique,
            palette=sns.color_palette(color_palette),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        save_type = 'Model_Input' if include_distances else 'Embeddings'
        save_path = (
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'{save_type}/{dataset_name}/Europe/'
            f'Region/{region_classes[i]}'
        )
        os.makedirs(save_path, exist_ok=True)
        scatterplot.figure.savefig(os.path.join(save_path, "output.png"))


def save_country_plots(
    REPO_PATH, y, country_list,
    tsne_results, dataset_name,
    region_classes,
    include_distances
):
    """Save plots of the t-SNE results seperated by region
    and colored by country

    Args:
        REPO_PATH (str): local path of repository
        y (List): List of labels for the t-SNE results
        country_list (DataFrame): Data Frame of Countries,
            Regions and Continents
        tsne_results (Array): Results of the t-SNE analysis
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        region_classes (List): list of all regions
        include_distances (Boolean): Defines whether the t-SNE
            analysis will be undertaken on the embeddings or
            embeddings appended with distances to prompts
    """
    # Create country labels

    region_specific_labels = []
    for region in region_classes:
        result_array = []
        for elem in y:
            country_row = (
                country_list
                .loc[country_list['Country'] == elem]
                .iloc[0]
            )
            if country_row['Intermediate Region Name'] == region:
                result_array.append(elem)
            else:
                result_array.append('Other')
        region_specific_labels.append(result_array)

    # Display tSNE results of countries within each intermediate region

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    for i in range(0, len(region_specific_labels)):
        df_subset['Classes'] = region_specific_labels[i]

        unique_classes = np.unique(region_specific_labels[i])
        modified_unique = list(np.delete(
            unique_classes,
            np.where(unique_classes == 'Other')
        ))
        color_palette = sns.color_palette("hls", len(modified_unique)).as_hex()
        color_palette.append('#d3d3d3')
        modified_unique.append('Other')

        plt.figure(figsize=(16, 10), clear=True)
        scatterplot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Classes",
            hue_order=modified_unique,
            palette=sns.color_palette(color_palette),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        save_type = 'Model_Input' if include_distances else 'Embeddings'
        save_path = (
            f'{REPO_PATH}/CLIP_Embeddings/t-SNE/'
            f'{save_type}/{dataset_name}/'
            f'Region/{region_classes[i]}'
        )
        os.makedirs(save_path, exist_ok=True)
        scatterplot.figure.savefig(os.path.join(save_path, "output.png"))


def conduct_tsne_analysis(
    REPO_PATH, dataset_name,
    only_europe, include_distances
):
    """Run t-SNE analysis on a dataset

    Args:
        REPO_PATH (str): local path of repository
        dataset_name (str): unique dataset name from
            {geoguessr, aerial, tourist}
        only_europe (Boolean): Defines whether only european samples
            will be analyzed
        include_distances (Boolean): Defines whether the t-SNE analysis
            will be undertaken on the embeddings or embeddings appended
            with distances to prompts
    """
    country_list = pd.read_csv(
        f'{REPO_PATH}/utils/country_list/'
        'country_list_region_and_continent.csv'
    )

    if (only_europe):
        # Load Data
        combined_df = load_european_data(REPO_PATH, dataset_name, country_list)
    else:
        combined_df = load_data(REPO_PATH, dataset_name, country_list)
    y = combined_df['label'].to_numpy()

    # Get sets of country, region and continent classes
    regions = []
    continents = []
    for elem in y:
        country_row = country_list.loc[country_list['Country'] == elem].iloc[0]
        regions.append(country_row['Intermediate Region Name'])
        continents.append(country_row['Continent'])

    region_classes = np.unique(regions)
    continent_classes = np.unique(continents)

    # Create numpy array for TSNE
    if include_distances:
        X = (
            combined_df["model_input"]
            .apply(transform_buffer_to_array)
            .to_list()
        )
    else:
        X = (
            combined_df["Embeddings"]
            .apply(transform_tensor_rep_to_array)
            .to_list()
        )
    X = np.array(X)
    X = np.squeeze(X)

    # Run TSNE
    tsne = TSNE(n_components=2, verbose=1, init='pca')
    tsne_results = tsne.fit_transform(X)
    if (not only_europe):
        save_continent_plot(
            REPO_PATH, tsne_results,
            dataset_name, continents,
            continent_classes,
            include_distances
        )
        save_region_plots(
            REPO_PATH, y, country_list,
            tsne_results, dataset_name,
            continent_classes,
            include_distances
        )
        save_country_plots(
            REPO_PATH, y,
            country_list, tsne_results,
            dataset_name, region_classes,
            include_distances
        )
    else:
        save_region_plots_europe(
            REPO_PATH, y,
            country_list, tsne_results,
            dataset_name, region_classes,
            include_distances
        )
        save_country_plots_europe(
            REPO_PATH, y,
            country_list, tsne_results,
            dataset_name, region_classes,
            include_distances
        )


if __name__ == "__main__":
    """Runs t-SNE on a dataset. The dataset_name along with the binary
    values of only_europe and include_distances can be modified below
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path',
        metavar='str',
        required=True,
        help='The path to the yaml file with the stored paths'
    )
    parser.add_argument(
        '--dataset_name',
        metavar='str',
        required=False,
        help='Name of dataset to conduct tsne analysis on',
        default='geoguessr'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        required=False,
        help='Enable debug mode',
        default=False
    )
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path']
        conduct_tsne_analysis(
            REPO_PATH,
            dataset_name=args.dataset_name,
            only_europe=False,
            include_distances=True
        )
