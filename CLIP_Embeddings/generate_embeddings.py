import clip
import torch
import pandas as pd
import sys
sys.path.append('.')
import PIL
import os
import numpy as np
from utils import load_dataset
import argparse
import ast
from sklearn.metrics.pairwise import cosine_similarity
import yaml

def save_dataframe_in_batches(df, batch_size, base_filename):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    for i in range(num_batches):
        batch = df[i * batch_size:(i + 1) * batch_size]
        batch.to_csv(f"{base_filename}_{i}.csv", index=False)


def calculate_distances(image_embedding, prompt_embeddings: list):
    """ Calculate the cosine similarity between the image embedding and the prompt embeddings

    Args:
        embedding_str (tensor): The image embedding as a tensor
        prompt_embeddings (list): List of prompt embeddings

    Returns:
        np.array: The model input
    """
    image_embedding_values = np.array(image_embedding.flatten().tolist()).reshape(1, -1)

    prompt_distances = []
    # Reshape the vectors to be 2D arrays for sklearn's cosine_similarity
    #image_embedding = image_embedding.reshape(1, -1)
    for prompt_embedding in prompt_embeddings:
        prompt_embedding_values = np.array(prompt_embedding.flatten().tolist()).reshape(1, -1)
        # Calculate Cosine Similarity         
        prompt_distances.append(cosine_similarity(image_embedding_values, prompt_embedding_values)[0,0])

    model_input = np.concatenate((image_embedding_values[0], np.array(prompt_distances))).astype(np.float32)
    return model_input.tobytes()


def generate_embeddings(REPO_PATH,DATA_PATH):
    """
    Generates embeddings for the geoguessr, tourist and aerial datasets and saves them to csv files

    Args:
        REPO_PATH (str): The path to the repository
        DATA_PATH (str): The path to the data folder
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    # load image data
    geoguessr_df = load_dataset.load_data(f'{DATA_PATH}/geoguessr')
    tourist_df = load_dataset.load_data(f'{DATA_PATH}/tourist')
    aerial_df = load_dataset.load_data(f'{DATA_PATH}/aerial')

    # generate Prompts
    country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv')["Country"].to_list()
    country_prompt = list(map((lambda x: f"This image shows the country {x}"),country_list))

    with torch.no_grad():
        # generate image embeddings
        geoguessr_df["Embedding"] = geoguessr_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
        tourist_df["Embedding"] = tourist_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
        aerial_df["Embedding"] = aerial_df["path"].apply(lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))

        # generate prompt embeddings
        simple_tokens = clip.tokenize(country_list)
        promt_token = clip.tokenize(country_prompt)

        simple_embedding = model.encode_text(simple_tokens)
        prompt_embedding = model.encode_text(promt_token)

    # generate model inputs, by appending distances to the prompt embeddings
    geoguessr_df["model_input"] = geoguessr_df["Embedding"].apply(lambda x: calculate_distances(x, prompt_embedding))
    aerial_df["model_input"] = aerial_df["Embedding"].apply(lambda x: calculate_distances(x, prompt_embedding))
    tourist_df["model_input"] = tourist_df["Embedding"].apply(lambda x: calculate_distances(x, prompt_embedding))

    # save image embeddings
    save_dataframe_in_batches(geoguessr_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/geoguessr_embeddings")
    save_dataframe_in_batches(tourist_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/tourist_embeddings")
    save_dataframe_in_batches(aerial_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/aerial_embeddings")

    # save prompt embeddings
    torch.save(simple_embedding, f'{REPO_PATH}/CLIP_Embeddings/Prompt/prompt_simple_embedding.pt')
    torch.save(prompt_embedding, f'{REPO_PATH}/CLIP_Embeddings/Prompt/prompt_image_shows_embedding.pt')

if __name__ == "__main__":
    """Generates embeddings for the geoguessr, tourist and aerial datasets and saves them to csv files
    """
    parser = argparse.ArgumentParser(description='Generate Embeddings')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        DATA_PATH = paths['data_path'][args.user]
        generate_embeddings(REPO_PATH,DATA_PATH)