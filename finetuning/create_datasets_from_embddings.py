import sys
sys.path.append("./../")
sys.path.append(".")

import random
import numpy as np
import sklearn.model_selection
import PIL
import os
import pandas as pd
import torch
import clip
from utils import load_dataset
import argparse




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)
model.to(device)


def balance_data(
    df: pd.DataFrame, max_images: int = 1000, min_images: int = 10, seed: int = 1234
):
    """
    Balance the data in a DataFrame by randomly sampling a maximum of images from each class, dropping classes below the minimum.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        REPO_PATH (str): The path to the repository.
        max_images (int, optional): The maximum number of images to sample. Defaults to 1000.
        min_images (int, optional): The minimum number of images to sample. Defaults to 10.
        seed (int, optional): The random seed. Defaults to 1234.

    Returns:
        pd.DataFrame: The balanced DataFrame.
    """
    # Group the DataFrame by label
    grouped = df.groupby("label")

    # Create an empty DataFrame to store the balanced data
    balanced_df = pd.DataFrame(columns=df.columns)

    # Iterate over each group
    for _, group_df in grouped:
        # Check if the group has enough images
        if len(group_df) >= min_images:
            # Randomly sample the maximum number of images from the group
            sampled_df = group_df.sample(
                n=min(len(group_df), max_images), random_state=seed
            )
            # Append the sampled data to the balanced DataFrame
            balanced_df = pd.concat([balanced_df, sampled_df], ignore_index=True)

    # Save the balanced DataFrame
    return balanced_df


def create_datasets_from_embddings(
    REPO_PATH,
    seed=1234,
    min_geo_percentage=0.5,
    max_images_weakly=2000,
    max_images_strongly=200,
    min_geo_images=10,
    min_aerial_images=2,
    min_tourist_images=10,
    test_size=0.15,
):
    """
    Create balanced datasets from embeddings for training and testing.

    Args:
        REPO_PATH (str): The path to the repository.
        seed (int, optional): The random seed for reproducibility. Defaults to 1234.
        min_geo_percentage (float, optional): The minimum percentage of geo images to replace with aerial and tourist images. Defaults to 0.5.
        max_images_weakly (int, optional): The maximum number of images for weakly balanced datasets. Defaults to 2000.
        max_images_strongly (int, optional): The maximum number of images for strongly balanced datasets. Defaults to 200.
        min_geo_images (int, optional): The minimum number of geo images. Defaults to 10.
        min_aerial_images (int, optional): The minimum number of aerial images. Defaults to 2.
        min_tourist_images (int, optional): The minimum number of tourist images. Defaults to 10.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.15.
    """

    # set random seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        # read in and balance each dataset
        geo_embed = pd.read_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Image/geoguessr_embeddings.csv")
        )
        aerial_df = pd.read_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Image/aerial_embeddings.csv")
        )
        tourist_df = pd.read_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Image/tourist_embeddings.csv")
        )

        # Print dataset information
        print(f"Datasets read in with seed {seed}")
        print(f"Geo: {len(geo_embed)}")
        print(f"Aerial: {len(aerial_df)}")
        print(f"Tourist: {len(tourist_df)}")

        # Balance the datasets
        balanced_geo_df = balance_data(
            df=geo_embed,
            max_images=max_images_weakly,
            min_images=min_geo_images,
            seed=seed,
        )
        balanced_aerial_df = balance_data(
            df=aerial_df,
            max_images=max_images_weakly,
            min_images=min_aerial_images,
            seed=seed,
        )
        balanced_tourist_df = balance_data(
            df=tourist_df,
            max_images=max_images_weakly,
            min_images=min_tourist_images,
            seed=seed,
        )

        # Split the datasets into train, validation and test sets
        geo_train_and_val, geo_test = sklearn.model_selection.train_test_split(
            balanced_geo_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=balanced_geo_df["label"],
        )
        aerial_train_and_val, aerial_test = sklearn.model_selection.train_test_split(
            balanced_aerial_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=balanced_aerial_df["label"],
        )
        # Make sure that all labels in the training set are also in the test set
        labels_in_aerial_train_not_in_aerial_test = set(aerial_train_and_val['label'].value_counts().keys().to_list()) - set(aerial_test['label'].value_counts().keys().to_list())
        for label in labels_in_aerial_train_not_in_aerial_test:
            entry = aerial_train_and_val[aerial_train_and_val['label'] == label].sample(1)
            aerial_test = pd.concat([aerial_test, entry])
            aerial_train_and_val = aerial_train_and_val.drop(entry.index)

        tourist_train_and_val, tourist_test = sklearn.model_selection.train_test_split(
            balanced_tourist_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=balanced_tourist_df["label"],
        )

        # Concatenate and shuffle all test and zero_shot datasets
        test_data = pd.concat([geo_test, aerial_test, tourist_test]).sample(
            frac=1, random_state=seed
        )

        # Create zero shot datasets
        geo_zero_shot_df = geo_embed[~geo_embed["label"].isin(test_data["label"])]
        aerial_zero_shot_df = aerial_df[
            ~aerial_df["label"].isin(test_data["label"])
        ]
        tourist_zero_shot_df = tourist_df[
            ~tourist_df["label"].isin(test_data["label"])
        ]

        zero_shot_data = pd.concat(
            [geo_zero_shot_df, aerial_zero_shot_df, tourist_zero_shot_df]
        ).sample(frac=1, random_state=seed)

        # Save test and zero shot datasets
        test_data.to_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Testing/known_test_data.csv"),
            index=False,
        )
        zero_shot_data.to_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Testing/zero_shot_test_data.csv"),
            index=False,
        )

        weakly_balanced_geo_df = geo_train_and_val.sample(frac=1, random_state=seed)
        weakly_balanced_geo_df.to_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Training/geo_weakly_balanced.csv"),
            index=False,
        )

        # Get all images from geo_df for the classes that have more than 2000 images
        # and that are not in balanced_geo_df
        geo_large_classes = (
            geo_embed["label"]
            .value_counts()[geo_embed["label"].value_counts() > max_images_weakly]
            .index.tolist()
        )
        geo_additional_images = geo_embed[
            geo_embed["label"].isin(geo_large_classes)
            & ~geo_embed["path"].isin(balanced_geo_df["path"])
        ]

        # Add the additional images to recreate the class imbalance
        unbalanced_geo_df = pd.concat(
            [geo_train_and_val, geo_additional_images]
        ).sample(frac=1, random_state=seed)
        unbalanced_geo_df.to_csv(
            os.path.join(REPO_PATH, "CLIP_Embeddings/Training/geo_unbalanced.csv"),
            index=False,
        )

        # Remove images of classes to have a maximum of 200 images for the strongly balanced dataset
        strongley_balanced_geo_df = balance_data(
            df=unbalanced_geo_df,
            max_images=max_images_strongly,
            min_images=0,
            seed=seed,
        ).sample(frac=1, random_state=seed)
        strongley_balanced_geo_df.to_csv(
            os.path.join(
                REPO_PATH, "CLIP_Embeddings/Training/geo_strongly_balanced.csv"
            ),
            index=False,
        )

        # Replace up to 50% of each label with images of that same label from aerial
        # and tourist data in the weakly balanced set
        mixed_weakly_balanced_df = weakly_balanced_geo_df.copy()
        for label in mixed_weakly_balanced_df["label"].unique():
            label_images = mixed_weakly_balanced_df[
                mixed_weakly_balanced_df["label"] == label
            ]
            aerial_images = aerial_train_and_val[aerial_train_and_val["label"] == label]
            tourist_images = tourist_train_and_val[
                tourist_train_and_val["label"] == label
            ]
            num_replace = min(
                int(len(label_images) * min_geo_percentage),
                len(aerial_images) + len(tourist_images),
            )
            if num_replace == 0:
                continue
            replace_images = pd.concat([aerial_images, tourist_images])
            replace_indices = (
                mixed_weakly_balanced_df[mixed_weakly_balanced_df["label"] == label]
                .sample(num_replace, random_state=seed)
                .index
            )
            cols_to_use = mixed_weakly_balanced_df.columns
            mixed_weakly_balanced_df.loc[replace_indices] = replace_images[cols_to_use].sample(num_replace, random_state=seed).to_numpy()

        # Replace up to 50% of each label with images of that same label from aerial
        # and tourist data in the strongly balanced set
        mixed_strongly_balanced_df = strongley_balanced_geo_df.copy()
        for label in mixed_strongly_balanced_df["label"].unique():
            label_images = mixed_strongly_balanced_df[
                mixed_strongly_balanced_df["label"] == label
            ]
            aerial_images = aerial_train_and_val[aerial_train_and_val["label"] == label]
            tourist_images = tourist_train_and_val[
                tourist_train_and_val["label"] == label
            ]
            num_replace = min(
                int(len(label_images) * min_geo_percentage),
                len(aerial_images) + len(tourist_images),
            )
            if num_replace == 0:
                continue
            replace_images = pd.concat([aerial_images, tourist_images])
            replace_indices = (
                mixed_strongly_balanced_df[mixed_strongly_balanced_df["label"] == label]
                .sample(num_replace, random_state=seed)
                .index
            )
            cols_to_use = mixed_strongly_balanced_df.columns
            mixed_strongly_balanced_df.loc[replace_indices] = replace_images[cols_to_use].sample(num_replace, random_state=seed).to_numpy()

        # Save the mixed datasets
        mixed_weakly_balanced_df.to_csv(
            os.path.join(
                REPO_PATH, "CLIP_Embeddings/Training/mixed_weakly_balanced.csv"
            ),
            index=False,
        )
        mixed_strongly_balanced_df.to_csv(
            os.path.join(
                REPO_PATH, "CLIP_Embeddings/Training/mixed_strongly_balanced.csv"
            ),
            index=False,
        )

        # Print dataset sizes
        print(f"Test data: {len(test_data)}")
        print(f"Zero shot data: {len(zero_shot_data)}")
        print(f"Geo weakly balanced: {len(weakly_balanced_geo_df)}")
        print(f"Geo unbalanced: {len(unbalanced_geo_df)}")
        print(f"Geo strongly balanced: {len(strongley_balanced_geo_df)}")
        print(f"Mixed weakly balanced: {len(mixed_weakly_balanced_df)}")
        print(f"Mixed strongly balanced: {len(mixed_strongly_balanced_df)}")

        # Check how much data from the tourist df is in the mixed weakly dataframe
        tourist_in_mixed_weakly = mixed_weakly_balanced_df[mixed_weakly_balanced_df["path"].isin(tourist_df["path"])]
        tourist_count_in_mixed_weakly = len(tourist_in_mixed_weakly)
        print(f"Number of tourist data in mixed weakly balanced dataset: {tourist_count_in_mixed_weakly}")

        # Check how much data from the aerial df is in the mixed weakly dataframe
        aerial_in_mixed_weakly = mixed_weakly_balanced_df[mixed_weakly_balanced_df["path"].isin(aerial_df["path"])]
        aerial_count_in_mixed_weakly = len(aerial_in_mixed_weakly)
        print(f"Number of aerial data in mixed weakly balanced dataset: {aerial_count_in_mixed_weakly}")

        # Check how much data from the tourist df is in the mixed strongly dataframe
        tourist_in_mixed_strongly = mixed_strongly_balanced_df[mixed_strongly_balanced_df["path"].isin(tourist_df["path"])]
        tourist_count_in_mixed_strongly = len(tourist_in_mixed_strongly)
        print(f"Number of tourist data in mixed strongly balanced dataset: {tourist_count_in_mixed_strongly}")

        # Check how much data from the aerial df is in the mixed strongly dataframe
        aerial_in_mixed_strongly = mixed_strongly_balanced_df[mixed_strongly_balanced_df["path"].isin(aerial_df["path"])]
        aerial_count_in_mixed_strongly = len(aerial_in_mixed_strongly)
        print(f"Number of aerial data in mixed strongly balanced dataset: {aerial_count_in_mixed_strongly}")

        # Check how much data from the tourist df is in the test data
        tourist_in_test = test_data[test_data["path"].isin(tourist_df["path"])]
        tourist_count_in_test = len(tourist_in_test)
        print(f"Number of tourist data in test dataset: {tourist_count_in_test}")

        # Check how much data from the aerial df is in the test data
        aerial_in_test = test_data[test_data["path"].isin(aerial_df["path"])]
        aerial_count_in_test = len(aerial_in_test)
        print(f"Number of aerial data in test dataset: {aerial_count_in_test}")

        # Check how much data from the tourist df is in the zero shot data
        tourist_in_zero_shot = zero_shot_data[zero_shot_data["path"].isin(tourist_df["path"])]
        tourist_count_in_zero_shot = len(tourist_in_zero_shot)
        print(f"Number of tourist data in zero shot dataset: {tourist_count_in_zero_shot}")

        # Check how much data from the aerial df is in the zero shot data
        aerial_in_zero_shot = zero_shot_data[zero_shot_data["path"].isin(aerial_df["path"])]
        aerial_count_in_zero_shot = len(aerial_in_zero_shot)
        print(f"Number of aerial data in zero shot dataset: {aerial_count_in_zero_shot}")

if __name__ == "__main__":
    """Creates the test set and the diffrent train/valdiation sets.
    This script uses the model inputs created using the "generate_image_embeddings.py".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_path",
        default="/media/leon/Samsung_T5/Uni/good_practices_ml/Embeddings/",
        type=str,
        help="Path to the repository",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="The random seed for reproducibility",
    )
    parser.add_argument(
        "--min_geo_percentage",
        default=0.5,
        type=float,
        help="The minimum percentage of geo images to replace with aerial and tourist images",
    )
    parser.add_argument(
        "--max_images_weakly",
        default=2000,
        type=int,
        help="The maximum number of images for weakly balanced datasets",
    )
    parser.add_argument(
        "--max_images_strongly",
        default=200,
        type=int,
        help="The maximum number of images for strongly balanced datasets",
    )
    parser.add_argument(
        "--min_geo_images",
        default=10,
        type=int,
        help="The minimum number of geo images",
    )
    parser.add_argument(
        "--min_aerial_images",
        default=2,
        type=int,
        help="The minimum number of aerial images",
    )
    parser.add_argument(
        "--min_tourist_images",
        default=10,
        type=int,
        help="The minimum number of tourist images",
    )
    parser.add_argument(
        "--test_size",
        default=0.15,
        type=float,
        help="The proportion of the dataset to include in the test split",
    )
    args = parser.parse_args()
    create_datasets_from_embddings(args.repo_path)
