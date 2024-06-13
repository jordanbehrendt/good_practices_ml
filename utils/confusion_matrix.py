from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def create_and_save_confusion_matrices(REPO_PATH, SAVE_FIGURES_PATH, true_countries, predicted_countries, normalize=False):
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
    # Flatten the lists if they are nested
    true_countries = [item for sublist in true_countries for item in sublist] if any(isinstance(i, list) for i in true_countries) else true_countries
    predicted_countries = [item for sublist in predicted_countries for item in sublist] if any(isinstance(i, list) for i in predicted_countries) else predicted_countries

    # Load country list and regional ordering index
    country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv')
    regional_ordering_index = [8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101, 105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140, 146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163, 166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192, 20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204, 78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72, 196, 13, 50, 55, 59, 62, 65, 69,
                                    86, 88, 92, 94, 113, 115, 142, 168, 172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147, 157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30, 138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57, 71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22, 37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21, 47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10, 18, 36, 49, 63, 110, 114, 118, 141]
    
    # constant for classes
    classes = country_list['Country']
    np_classes = np.array(classes)

    # Build country confusion matrices
    if normalize:
        cf_matrix = confusion_matrix(true_countries, predicted_countries, labels=range(0, 211), normalize='true')
    else:
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

    # Create region labels
    np_regions = np.sort(np.array(list(set(country_list['Intermediate Region Name']))))
    true_regions = []
    predicted_regions = []
    for i in range(0, len(true_countries)):
        true_regions.append(ast.literal_eval(country_list.iloc[true_countries[i]]["One Hot Region"]).index(1))
        predicted_regions.append(ast.literal_eval(country_list.iloc[predicted_countries[i]]["One Hot Region"]).index(1))

    # Build region confusion matrices
    if normalize:
        regions_cf_matrix = confusion_matrix(true_regions, predicted_regions, labels=range(0, len(np_regions)), normalize='true')
    else:
        regions_cf_matrix = confusion_matrix(true_regions, predicted_regions, labels=range(0, len(np_regions)))
    regions_ordered_index = np.argsort(-regions_cf_matrix.diagonal())
    regions_ordered_matrix = regions_cf_matrix[regions_ordered_index][:,regions_ordered_index]
    ordered_regions = np_regions[regions_ordered_index]

    regions_df_cm = pd.DataFrame(regions_cf_matrix, index=np_regions, columns=np_regions)
    regions_ordered_df_cm = pd.DataFrame(regions_ordered_matrix, index=ordered_regions, columns=ordered_regions)

    # Save confusion matrices
    if normalize:
        if not os.path.exists(f'{SAVE_FIGURES_PATH}/normalized'):
            os.makedirs(f'{SAVE_FIGURES_PATH}/normalized')
    else:
        if not os.path.exists(f'{SAVE_FIGURES_PATH}'):
            os.makedirs(f'{SAVE_FIGURES_PATH}')       

    # Plot and save country confusion matrices
    plot_and_save_heatmap(df_cm, f'{SAVE_FIGURES_PATH}/simple_confusion_matrix.png', normalize)
    plot_and_save_heatmap(ordered_df_cm, f'{SAVE_FIGURES_PATH}/ordered_confusion_matrix.png', normalize)
    plot_and_save_heatmap(regionally_ordered_df_cm, f'{SAVE_FIGURES_PATH}/regionally_ordered_confusion_matrix.png', normalize)

    # Plot and save region confusion matrices
    plot_and_save_heatmap(regions_df_cm, f'{SAVE_FIGURES_PATH}/regions_confusion_matrix.png', normalize)
    plot_and_save_heatmap(regions_ordered_df_cm, f'{SAVE_FIGURES_PATH}/regions_ordered_confusion_matrix.png', normalize)

def plot_and_save_heatmap(dataframe, file_path, normalize):
    plt.figure(figsize=(120, 90))
    sns.set(font_scale=8)
    ax = sns.heatmap(dataframe, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=1, yticklabels=1)
    ax.tick_params(axis='both', labelsize=15)
    ax.set(xlabel=None, ylabel=None)
    ax.figure.savefig(file_path)
    plt.close(ax.figure)

# Example usage
# REPO_PATH = 'path_to_repo'
# SAVE_FIGURES_PATH = 'path_to_save_figures'
# true_countries = [list_of_true_countries]
# predicted_countries = [list_of_predicted_countries]
# create_and_save_confusion_matrices(REPO_PATH, SAVE_FIGURES_PATH, true_countries, predicted_countries, normalize=True)
