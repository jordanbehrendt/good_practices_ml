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
from typing import List


def test_model(DATA_PATH: str, REPO_PATH: str,model, model_name: str, experiment_name: str, test_sets: List[pd.DataFrame], possible_labels: List[str], batch_size: int = 450) -> None:
    seed = 1234
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset in test_sets:
        if dataset.target_transform:
            possible_labels = [dataset.target_transform(x) for x in possible_labels]
        else:
            possible_labels = possible_labels
        label_tokens = clip.tokenize(possible_labels)
        print(f"Running data from dataset: {dataset.name}")
        
        for batch_number, (images, labels) in enumerate(tqdm.tqdm(DataLoader(dataset, batch_size=batch_size),desc=f"Testing on {dataset.name}")):
            
            images = images.to(device)
                
            with torch.no_grad():

                logits_per_image, logits_per_text = model(images, label_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            max_index = probs.argmax(axis=1)  # Finding the index of the maximum probability for each sample
            max_probabilities = probs[range(probs.shape[0]), max_index]
            predicted_label = np.array(possible_labels)[max_index]

            performance_data = pd.DataFrame({
                'Probabilities': max_probabilities,
                'Predicted labels': predicted_label,
                'label' : labels,
                'All-Probs' : probs.tolist()
            })
            scripts.save_data_to_file(performance_data,model_name,dataset.name,batch_number,experiment_name = experiment_name, output_dir=os.path.join(REPO_PATH,'Experiments/'))

