import sklearn.model_selection
import random
import model_tester as testing
import pandas as pd
import scripts.load_dataset as geo_data
import os
import argparse
import yaml
import clip
import torch
import sys
sys.path.append('.')
# ----------------------------------------------


def run_experiment(DATA_PATH: str, debug: bool, config_path: str):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        min_img = config['min_img']
        max_img = config['max_img']
        batch_size = config['batch_size']
        test_size = config['test_size']
        seed = config['random_seed']
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    sys.path.append(f'{REPO_PATH}scripts')
    geoguessr_df = geo_data.load_data(
        DATA_PATH=DATA_PATH, min_img=min_img, max_img=max_img, size_constraints=True, debug_data=debug)

    # create country list for the clip model to use as possible labels
    country_list = pd.read_csv(os.path.join(
        REPO_PATH, "data_exploration", "image_distribution.csv"))['label'].to_list()

    if debug:
        test = geoguessr_df
        train = geoguessr_df
    else:
        train, test = sklearn.model_selection.train_test_split(
            geoguessr_df, test_size=test_size, random_state=seed, stratify=geoguessr_df["label"], shuffle=True)

    test_dataset = geo_data.ImageDataset_from_df(test, transform=preprocessor)

    standard_tester = testing.ModelTester(test_dataset, model, (lambda x: x), batch_size,
                                          country_list, seed, REPO_PATH, 'pretrained', 'no_prompt', 'experiment0')
    standard_tester.run_test()
    elab_tester = testing.ModelTester(test_dataset, model, (
        lambda x: f"This image shows the country {x}"), batch_size, country_list, seed, REPO_PATH, 'pretrained', 'elab_prompt', 'experiment0')
    elab_tester.run_test()
    streetveiw_tester = testing.ModelTester(test_dataset, model, (
        lambda x: f"A google streetview image from {x}"), batch_size, country_list, seed, REPO_PATH, 'pretrained', 'streetveiw_prompt', 'experiment0')
    streetveiw_tester.run_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    parser.add_argument('--config_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored configs')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        run_experiment(DATA_PATH, args.debug, args.config_path)
