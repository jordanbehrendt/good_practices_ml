#tester = ModelTester(dataset=my_dataset, model=my_model, prompt=my_prompt, batch_size=32, country_list=my_country_list, seed=42, folder_path='./', model_name='MyModel', prompt_name='MyPrompt', custom_tag='Tag1')

import sys
sys.path.append('.')

import models
from models import model_tester
import scripts
from scripts import load_dataset
import clip
import torch
import csv
import pandas as pd

seed = 1234

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device)

#geoguessr = load_dataset.load_data('/share/temp/bjordan/good_practices_in_machine_learning/compressed_dataset/', 20, 5000, False, False, seed)
#geoguessr = load_dataset.ImageDataset_from_df(geoguessr, preprocessor, name= "GeoGuesser")
#bigfoto = load_dataset.load_data('/share/temp/bjordan/good_practices_in_machine_learning/BigFoto/', 0, 5000, False, False, seed)
#bigfoto = load_dataset.ImageDataset_from_df(bigfoto, preprocessor, name= "BigFoto")
aerialmap = load_dataset.load_data('/share/temp/bjordan/good_practices_in_machine_learning/OpenAerialMap/', 0, 5000, False, False, seed)
aerialmap = load_dataset.ImageDataset_from_df(aerialmap, preprocessor, name= "OpenAerialMap")
#mars = load_dataset.load_data('/share/temp/bjordan/good_practices_in_machine_learning/Mars/', 0, 5000, False, False, seed)
#mars = load_dataset.ImageDataset_from_df(mars, preprocessor, name= "Mars")

default_prompt = lambda x: f"{x}"
image_prompt = lambda x: f"This image shows the country {x}"
geoguessr_context_prompt = lambda x: f"A google streetview image from {x}"
bigfoto_context_prompt = lambda x: f"A holiday photo from {x}"
aerialmap_context_prompt = lambda x: f"An aerial image from {x}"
mars_context_prompt = lambda x: f"Image of a landing site from {x}"

geoguessr_batch_size = 450
bifgoto_batch_size = 126
aerialmap_batch_size = 15
mars_batch_size = 8

country_list = pd.read_csv('/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list.csv')["Country"].to_list()
#mars_list = 

folder_path = './'
model_name = 'pretrained_clip'

default_prompt_name = 'country_prompt'
image_prompt_name = 'image_from_prompt'
context_prompt_name = 'context_prompt'

aerialmap_default = model_tester.ModelTester(aerialmap, model, default_prompt, aerialmap_batch_size, country_list, seed, folder_path, model_name, default_prompt_name, 'v1')
aerialmap_default.run_test()
aerialmap_image = model_tester.ModelTester(aerialmap, model, image_prompt, aerialmap_batch_size, country_list, seed, folder_path, model_name, image_prompt_name, 'v1')
aerialmap_image.run_test()
aerialmap_context = model_tester.ModelTester(aerialmap, model, aerialmap_context_prompt, aerialmap_batch_size, country_list, seed, folder_path, model_name, context_prompt_name, 'v1')
aerialmap_context.run_test()