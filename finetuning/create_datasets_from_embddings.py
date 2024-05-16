import sys
sys.path.append("./../")
sys.path.append(".")
import argparse
from utils import load_dataset
import clip
import torch
import pandas as pd
import os
import PIL
import sklearn.model_selection
import numpy as np
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)
model.to(device)


def balance_data(df: pd.DataFrame, max_images: int = 1000, min_images: int = 10, seed: int = 1234):
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
                n=min(len(group_df), max_images), random_state=seed)
            # Append the sampled data to the balanced DataFrame
            balanced_df = pd.concat(
                [balanced_df, sampled_df], ignore_index=True)

    # Save the balanced DataFrame
    return balanced_df


def create_datasets_from_embddings(REPO_PATH, seed=1234):
    # set radom seed
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    
    with torch.no_grad():
        # read in and balance each dataset
        geo_embed = pd.read_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Image/geoguessr_embeddings.csv"))
        aerial_df = pd.read_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Image/aerial_embeddings.csv"))
        tourist_df = pd.read_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Image/tourist_embeddings.csv"))

        # Balance the datasets
        balanced_geo_df = balance_data(
            df=geo_embed, max_images=2000, min_images=10, seed=seed)
        balanced_aerial_df = balance_data(
            df=aerial_df, max_images=2000, min_images=2, seed=seed)
        balanced_tourist_df = balance_data(
            df=tourist_df, max_images=2000, min_images=10, seed=seed)

        # Split the datasets into train, validation and test sets
        geo_train_and_val, geo_test = sklearn.model_selection.train_test_split(
            balanced_geo_df, test_size=0.15, random_state=seed, shuffle=True, stratify=balanced_geo_df["label"])
        aerial_train_and_val, aerial_test = sklearn.model_selection.train_test_split(
            balanced_aerial_df, test_size=0.15, random_state=seed, shuffle=True, stratify=balanced_aerial_df["label"])
        tourist_train_and_val, tourist_test = sklearn.model_selection.train_test_split(
            balanced_tourist_df, test_size=0.15, random_state=seed, shuffle=True, stratify=balanced_tourist_df["label"])

        # Get all images of labels not present in the training and validation sets
        geo_labels_with_few_images = geo_train_and_val["label"].value_counts(
        )[geo_train_and_val["label"].value_counts() <= 10].index.tolist()
        aerial_labels_with_few_images = aerial_train_and_val["label"].value_counts(
        )[aerial_train_and_val["label"].value_counts() <= 1].index.tolist()
        tourist_labels_with_few_images = tourist_train_and_val["label"].value_counts(
        )[tourist_train_and_val["label"].value_counts() <= 10].index.tolist()

        # create zero_shot datasets
        geo_zero_shot_df = geo_train_and_val[geo_train_and_val["label"].isin(
            geo_labels_with_few_images)]
        aerial_zero_shot_df = aerial_train_and_val[aerial_train_and_val["label"].isin(
            aerial_labels_with_few_images)]
        tourist_zero_shot_df = tourist_train_and_val[tourist_train_and_val["label"].isin(
            tourist_labels_with_few_images)]

        # Concatenate and shuffle all test and zero_shot datasets
        test_data = pd.concat([geo_test, aerial_test, tourist_test]).sample(
            frac=1, random_state=seed)
        zero_shot_data = pd.concat(
            [geo_zero_shot_df, aerial_zero_shot_df, tourist_zero_shot_df]).sample(
            frac=1, random_state=seed)
        
        test_data.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Testing/knwon_test_data.csv"), index=False)
        zero_shot_data.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Testing/zero_shot_test_data.csv"), index=False)
        test_data = pd.concat([test_data, zero_shot_data]).sample(
            frac=1, random_state=seed)

        test_data.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Testing/test_data.csv"), index=False)

        weakly_balanced_geo_df = geo_train_and_val.sample(
            frac=1, random_state=seed)
        weakly_balanced_geo_df.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Training/geo_weakly_balanced.csv"), index=False)

        # Get all images from geo_df for the classes that have more than 2000 images and that are not in balanced_geo_df
        geo_large_classes = geo_embed["label"].value_counts(
        )[geo_embed["label"].value_counts() > 2000].index.tolist()
        geo_additional_images = geo_embed[geo_embed["label"].isin(
            geo_large_classes) & ~geo_embed["path"].isin(balanced_geo_df["path"])]

        # add the additional images to recreate the class imbalance
        unbalanced_geo_df = pd.concat(
            [geo_train_and_val, geo_additional_images]).sample(frac=1, random_state=seed)
        unbalanced_geo_df.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Training/geo_unbalanced.csv"), index=False)

        # remove image of classes to have a maximum of 200 images
        strongley_balanced_geo_df = balance_data(
            df=unbalanced_geo_df, max_images=200, min_images=10, seed=seed).sample(frac=1, random_state=seed)
        strongley_balanced_geo_df.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Training/geo_strongly_balanced.csv"), index=False)

        # add aerial and tourist images to create a weakly balanced mixed dataset
        mixed_weakly_balanced_df = pd.concat(
            [weakly_balanced_geo_df, aerial_train_and_val, tourist_train_and_val]).sample(frac=1, random_state=seed)
        mixed_weakly_balanced_df.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Training/mixed_weakly_balanced.csv"), index=False)

        print(len(tourist_train_and_val)/len(weakly_balanced_geo_df))
        tourist_percentage = len(tourist_train_and_val)/len(weakly_balanced_geo_df)
        number_of_strongly_balanced_images = len(strongley_balanced_geo_df)*tourist_percentage
        _, small_tourist_df = sklearn.model_selection.train_test_split(
            tourist_train_and_val, test_size=int(number_of_strongly_balanced_images), random_state=seed, shuffle=True, stratify=tourist_train_and_val["label"])
        # add aerial and tourist images to create a strongly balanced mixed dataset
        print(len(small_tourist_df)/len(strongley_balanced_geo_df))
        mixed_strongly_balanced_df = pd.concat(
            [strongley_balanced_geo_df, aerial_train_and_val, small_tourist_df]).sample(frac=1, random_state=seed)
        mixed_strongly_balanced_df.to_csv(os.path.join(
            REPO_PATH, "CLIP_Embeddings/Training/mixed_strongly_balanced.csv"), index=False)

if __name__ == "__main__":
    """Creates the test set and the diffrent train/valdiation sets.
    This script uses the model inputs created using the "generate_image_embeddings.py".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path",  default="/home/lbrenig/Documents/Uni/GPML/good_practices_ml", type=str, help="Path to the repository")
    args = parser.parse_args()
    create_datasets_from_embddings(args.repo_path)
