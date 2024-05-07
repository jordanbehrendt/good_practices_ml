import clip
import torch
import pandas as pd
import sys
import PIL
import os
import numpy as np
sys.path.append("./../")
from utils import load_dataset
import argparse
import ast
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)

parser = argparse.ArgumentParser()
parser.add_argument("--tourist_data_path", type=str, help="Path to Tourist dataset")
parser.add_argument("--geoguessr_data_path", type=str, help="Path to Geoguessr dataset")
parser.add_argument("--aerial_data_path", type=str, help="Path to Aerial dataset")
parser.add_argument("--REPO_PATH", type=str, help="Path to the repository")
args = parser.parse_args()


def calculate_distances(embedding_str: str, prompt_embeddings: list):
    start = embedding_str.find('[[')
    end = embedding_str.find(']]')+2
    embedding_str = embedding_str[start:end]
    image_embedding = torch.tensor(ast.literal_eval(embedding_str))
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

# load image data
geoguessr_df = load_dataset.load_data(DATA_PATH=args.geoguessr_data_path)
tourist_df = load_dataset.load_data(DATA_PATH=args.tourist_data_path)
aerial_df = load_dataset.load_data(DATA_PATH=args.aerial_data_path)

# generate Prompts
country_list = pd.read_csv(os.path.join(args.REPO_PATH,"country_list/country_list_region_and_continent.csv"))["Country"].to_list()
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


geoguessr_df.to_csv(os.path.join(args.REPO_PATH,"Image/geoguessr_embeddings.csv"))
tourist_df.to_csv(os.path.join(args.REPO_PATH,"Image/tourist_embeddings.csv"))
aerial_df.to_csv(os.path.join(args.REPO_PATH,"Image/aerial_embeddings.csv"))

torch.save(simple_embedding, os.path.join(args.REPO_PATH,'/CLIP_Embeddings/Prompt/prompt_simple_embedding.pt'))
torch.save(prompt_embedding, os.path.join(args.REPO_PATH,'/CLIP_Embeddings/Prompt/prompt_image_shows_embedding.pt'))