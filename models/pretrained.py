import os
import PIL
from PIL import Image
import pandas as pd
import numpy as np
import torch
import clip
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import yaml
import argparse
import sys

class ImageDataset_from_df(Dataset):
    def __init__(self, df):

        self.images = df["path"].tolist()
        self.caption = clip.tokenize(df["label"].tolist())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocessor = clip.load("ViT-B/32", device=device)

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        
        images = self.preprocessor(Image.open(self.images[idx])) #preprocess from clip.load
        caption = self.caption[idx]
        return images,caption

def pretrained_model(DATA_PATH: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    sys.path.append(f'{REPO_PATH}scripts')
    import load_geoguessr_data
    geoguessr_df = load_geoguessr_data.load_data(DATA_PATH=DATA_PATH)

    dataset = ImageDataset_from_df(geoguessr_df)
    batch_size = 100

    with torch.no_grad():
        for images, texts in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
            print(images, texts)
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print(probs)
            max_index = probs.argmax()
            print(probs[0][max_index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='the user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='the path to the yaml file with the stored paths')
    args = parser.parse_args()
    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        pretrained_model(DATA_PATH)
