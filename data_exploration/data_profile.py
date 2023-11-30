import sys
from ydata_profiling import ProfileReport
import csv
import yaml
import argparse
import world_heat_map


def data_profile(DATA_PATH: str, REPO_PATH: str):
    sys.path.append(f'{REPO_PATH}scripts')
    import load_geoguessr_data
    geoguessr_df = load_geoguessr_data.load_data(DATA_PATH=DATA_PATH, size_constraints = True)
    profile = ProfileReport(geoguessr_df, title='Geoguessr Profile Report')
    profile.to_file(f'{REPO_PATH}/data_exploration/geoguessr_profile.html')
    image_distribution = geoguessr_df['label'].value_counts()
    image_distribution_path = f'{REPO_PATH}/data_exploration/image_distribution.csv'
    image_distribution.to_csv(image_distribution_path)
    world_heat_map.world_heat_map(image_distribution_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Geoguessr Data Profile')
    parser.add_argument('--user', metavar='str', required=True, help='the user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='the path to the yaml file with the stored paths')
    args = parser.parse_args()
    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        data_profile(DATA_PATH, REPO_PATH)