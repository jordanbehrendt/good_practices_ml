# -*- coding: utf-8 -*-
"""
finetuning.run_experiments
--------------------------

Script to run training loop of the model on different
training datasets (list of datasets to use can be
provided as `--training_datasets dataset_1.csv dataset_2.csv ...`)
"""
# Imports
# Built-in
import yaml
import argparse

# Local
import countryguessr.finetuning.model.model_trainer as trainer

# 3r-party


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model training')
    parser.add_argument(
        '--yaml_path',
        metavar='str',
        required=True,
        help='The path to the yaml file with the stored paths'
    )
    parser.add_argument(
        '--training_datasets',
        nargs="*",
        required=False,
        help='List of filenames for the datasets to use.',
        default=None
    )
    parser.add_argument(
        '--balance_loss',
        metavar='bool',
        required=False,
        help='Defines if the loss is balanced or normalized',
        default=False
    )
    args = parser.parse_args()
    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path']

    seeds = [4808, 4947, 5723, 3838, 5836, 3947, 8956, 5402, 1215, 8980]
    for seed in seeds:
        if args.training_datasets == None:
            trainer.create_and_train_model(REPO_PATH, seed, balance_loss_components=args.balance_loss)
        else:
            trainer.create_and_train_model(REPO_PATH, seed, args.training_datasets, balance_loss_components=args.balance_loss)
