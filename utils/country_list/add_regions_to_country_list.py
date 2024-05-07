import pandas as pd
import torch
import argparse
import yaml

def add_continent_and_region(REPO_PATH):
    """
    Add the Continent and Regional Names as columns to the country_list dataframe.

    Args:
        REPO_PATH (str): The path to the repository.
    """
    country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list.csv')
    UNSD = pd.read_csv(f'{REPO_PATH}/utils/country_list/UNSD_Methodology.csv')

    find_continent_name = lambda x: UNSD.loc[UNSD['ISO-alpha2 Code'] == x].iloc[0]['Region Name']
    find_region_name = lambda x: UNSD.loc[UNSD['ISO-alpha2 Code'] == x].iloc[0]['Intermediate Region Name']
    country_list['Continent'] = country_list['Alpha2Code'].apply(find_continent_name)
    country_list['Intermediate Region Name'] = country_list['Alpha2Code'].apply(find_region_name)

    country_list.to_csv('./country_list/country_list_region_and_continent.csv')


def add_one_hot_encodings(REPO_PATH):
    """
    Add One Hot Country Encodings and One Hot Region Encodings as columns to the country_list dataframe.

    Args:
        REPO_PATH (str): The path to the repository.
    """

    country_list = pd.read_csv(f"{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv")
    countries_df = pd.get_dummies(country_list['Country'])
    regions_df = pd.get_dummies(country_list['Intermediate Region Name'], prefix='Region')

    # Convert the one-hot encoded DataFrames to lists of lists
    one_hot_regions_list = regions_df.values.tolist()
    one_hot_regions = [[int(value) for value in row] for row in one_hot_regions_list]
    one_hot_countries_list = countries_df.values.tolist()
    one_hot_countries = [[int(value) for value in row] for row in one_hot_countries_list]

    # Add the lists of one-hot encoded vectors to the original DataFrame as a new column
    country_list['One Hot Region'] = one_hot_regions
    country_list['One Hot Country'] = one_hot_countries
    country_list.to_csv(f"{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv", index=False)


if __name__ == "__main__":
    """Extends the country_list.csv dataframe to include continent, regional and one-hot-encoding information
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument("--yaml_path",  default="", type=str, help="Path to the yaml file")
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        add_continent_and_region(REPO_PATH)
        add_one_hot_encodings(REPO_PATH)