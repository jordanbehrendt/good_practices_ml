import os
import PIL
import torch
import clip
import pandas as pd
from torch.utils.data import Dataset
import random

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
        files = os.listdir(os.path.join(DATA_PATH,folder))
        num_images = len(files)
        if num_images < min_img:
            continue
        if (max_img is not None) and (num_images > max_img):
            files = random.sample(files, max_img)
        for file in files:
            with PIL.Image.open(os.path.join(DATA_PATH,folder,file)) as img:
                width, height = img.size
                form = img.format
                temp_dict={
                    'label' : folder,
                    'width' : width,
                    'height' : height,
                    'format' : form,
                    'path' : os.path.join(DATA_PATH,folder,file)
                }
                list_rows.append(temp_dict)
    df = pd.DataFrame(list_rows)
    if size_constraints:
        df = df.loc[df['width'] == 1536]
    if min_img > 0:
        df = filter_min_img_df(df, min_img)
    if debug_data:
        df = df.iloc[:99]
    return df


class ImageDataset_from_df(Dataset):
    def __init__(self, df,transform=None, target_transform=None, name='default_data'):
        if target_transform:
            self.captions = clip.tokenize(df["label"].apply(target_transform).tolist())
        else:
            self.captions = clip.tokenize(df["label"].tolist())
        self.images = df["path"].tolist()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocessor = clip.load("ViT-B/32", device=device)
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        image = self.preprocessor(image) #preprocess from clip.load
        caption = self.captions[idx]

        return image,caption