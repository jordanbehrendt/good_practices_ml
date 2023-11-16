import sys
sys.path.append('.')
#----------------------------------------------
import torch
import clip
import tqdm
import yaml
import argparse
import sys
import scripts.load_geoguessr_data as geo_data
import numpy as np
from torch.utils.data import DataLoader



def pretrained_model(DATA_PATH: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)
    
    sys.path.append(f'{REPO_PATH}scripts')
    geoguessr_df = geo_data.load_data(DATA_PATH=DATA_PATH)

    dataset = geo_data.ImageDataset_from_df(geoguessr_df)
    batch_size = 10

    with torch.no_grad():
        for images, texts in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
            print(images, texts)
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print(probs)
            max_index = probs[0].argmax()
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
