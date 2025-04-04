import os
import PIL
import torch
import clip
import pandas as pd
import ast
from torch.utils.data import Dataset
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def filter_min_img_df(df: pd.DataFrame, min_img: int):
    """Filters classes by minimum amount of images

    Args:
        df (pd.DataFrame): dataframe that should be filtered
        min_img (int): minimum number of images

    Returns:
        pd.DataFrame: filtered dataframe
    """
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= min_img].index
    df = df[df['label'].isin(valid_labels)]
    return df


def load_data(DATA_PATH: str, min_img: int = 0, max_img: int = None, size_constraints: bool = False, debug_data: bool = False, random_seed: int = 1234):
    """Loads data in a dataframe form a given folder, with basic filtering.

    Args:
        DATA_PATH (str): Path to folder containing folders of images.
        min_img (int, optional): Minimal number of images accepted into the dataset. Defaults to 0.
        max_img (int, optional): Maximal number of images accepted into the dataset. Defaults to None.
        size_constraints (bool, optional): Remove images of diffrent sizes. Defaults to False.
        debug_data (bool, optional): Reduces dataset size to 100 images. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containg basic infromation on label, img widht/hight, format, path to img.
    """
    random.seed(random_seed)

    list_rows = []
    for folder in os.listdir(DATA_PATH):
        files = os.listdir(os.path.join(DATA_PATH, folder))
        num_images = len(files)
        if num_images < min_img:
            continue
        if (max_img is not None) and (num_images > max_img):
            files = random.sample(files, max_img)
        for file in files:
            with PIL.Image.open(os.path.join(DATA_PATH, folder, file)) as img:
                width, height = img.size
                form = img.format
                temp_dict = {
                    'label': folder,
                    'width': width,
                    'height': height,
                    'format': form,
                    'path': os.path.join(DATA_PATH, folder, file)
                }
                list_rows.append(temp_dict)
    df = pd.DataFrame(list_rows)
    if size_constraints:
        df = df.loc[df['width'] == 1536]
    if min_img > 0:
        df = filter_min_img_df(df, min_img)
    if debug_data:
        df = df.sample(10)
    df = df.sample(frac=1,random_state=random_seed).reset_index(drop=True)
    return df


class ImageDataset_from_df(Dataset):
    def __init__(self, df, transform=None, target_transform=None, name='default_data'):
        self.captions = df["label"].tolist()
        self.images = df["path"].tolist()
        self.target_transform = target_transform
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        if self.target_transform:
            caption = self.target_transform(caption)

        return image, caption
    
class EmbeddingDataset_from_df(Dataset):
    def __init__(self, df, name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = df['label'].tolist()
        self.model_inputs = torch.tensor([np.frombuffer(eval(value),dtype=np.float32) for value in df['model_input']], dtype=torch.float32, device=self.device)
        self.name = name

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        model_input = self.model_inputs[index]
        return model_input, label