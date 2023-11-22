import sys
sys.path.append('.')
#----------------------------------------------
import torch
import clip
import tqdm
import yaml
import argparse
import os
import scripts.load_geoguessr_data as geo_data
import scripts.helpers as scripts
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
import sklearn.model_selection


def pretrained_model(DATA_PATH: str, debug: bool):
    seed = 1234
    random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    sys.path.append(f'{REPO_PATH}scripts')
    geoguessr_df = geo_data.load_data(DATA_PATH=DATA_PATH, min_img = 20, max_img = 5000, size_constraints= True, debug_data=debug)
    if debug:
        test = geoguessr_df
        train = geoguessr_df
    else:  
        train, test = sklearn.model_selection.train_test_split(geoguessr_df, test_size = 0.2, random_state = seed, stratify = geoguessr_df["label"])

    standard_dataset = geo_data.ImageDataset_from_df(test)
    v1_dataset = geo_data.ImageDataset_from_df(test, target_transform=(lambda x : f"This image shows the country {x}"), name="elab_prompt")
    v2_dataset = geo_data.ImageDataset_from_df(test, target_transform=(lambda x : f"A google streetview image from {x}"), name="street_prompt")

    dataset_collection = [standard_dataset,v1_dataset,v2_dataset]

    batch_size = 100
    for dataset in dataset_collection:
        all_probabilities = []
        all_texts = []
        print(f"Running data from dataset: {dataset.name}")
        with torch.no_grad():
            for images, texts in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
                
                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                max_index = probs.argmax(axis=1)  # Finding the index of the maximum probability for each sample
                max_probabilities = probs[range(probs.shape[0]), max_index]
                all_probabilities.extend(max_probabilities)
                all_texts.extend(texts.cpu().numpy())
        performance_data = pd.DataFrame({
            'Probability': all_probabilities,
            'Text': all_texts,
        })
        scripts.save_data_to_file(performance_data,"pretrained",dataset.name,os.path.join(REPO_PATH,'Experiments/'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True, help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        pretrained_model(DATA_PATH,args.debug)
