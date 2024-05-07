import sys
sys.path.append('.')

from typing import Callable, List
from torch.utils.data import DataLoader
from utils import load_dataset
import utils.load_dataset as geo_data
import clip
import torch
import csv
import pandas as pd
import argparse
import yaml
import os
import tqdm
from datetime import datetime
import random


class ModelTester:
    """
    A class for testing a PyTorch model on a specified dataset and saving the results as csv.

    Attributes:
        test_set (geo_data.ImageDataset_from_df): The test dataset.
        model (torch.nn.Module): The model to test.
        prompt (Callable): Transformation for the prompt given the country name.
        batch_size (int): The batch size to use.
        country_list (List[str]): List of all possible countries.
        seed (int): Random seed used for operations.
        folder_path (str): Path to save results to.
        model_name (str): The name of the model that is used.
        prompt_name (str): The name of the prompt used.
        custom_tag (str): Custom tag for naming the experiment.

    Methods:
        __init__(self, dataset: geo_data.ImageDataset_from_df, model: torch.nn.Module, prompt: Callable, batch_size: int, country_list: List[str], seed: int, folder_path: str, model_name: str, prompt_name: str, custom_tag: str):
            Initializes a new instance of the ModelTester class.

        run_test(self):
            Runs the model on the given test set with the specified batch size.
            The results are saved as CSV files using the structure:
            {output_folder}/Experiments/{model_name}/{prompt_name}/{dataset_name}-{custom_tag}/{date}-{batch_number}.csv

        __save_data_to_file(self, data: pd.DataFrame, model_name: str, prompt_name: str, dataset_name: str, batch_number: str, custom_tag: str = None, output_dir='./Experiments/'):
            Saves data from a Pandas DataFrame as a CSV file in the specified structure.

    Usage:
        # Example usage:
        tester = ModelTester(dataset=my_dataset, model=my_model, prompt=my_prompt, batch_size=32, country_list=my_country_list, seed=42, folder_path='./', model_name='MyModel', prompt_name='MyPrompt', custom_tag='Tag1')
        tester.run_test()
    """

    def __init__(self, dataset: geo_data.ImageDataset_from_df, model: torch.nn.Module, prompt: List[Callable], batch_size: int, country_list: List[str], seed: int, folder_path: str, model_name: str, prompt_name: List[str], custom_tag: str):
        """Generate a ModelTester object, that can be used to test the model.

        Args:
            dataset (geo_data.ImageDataset_from_df): The test-dataset.
            model (torch.nn.Module): The Model to test.
            prompt (List[Callable]): Transformations for prompts given the countryname.
            batch_size (int): The batch size to use.
            country_list (List[str]): List of all possible countries.
            seed (int): Random seed used for operations.
            folder_path (str): Path to save results to.
            model_name (str): The name of the model that is used.
            prompt_name (List[str]): The name of all prompts used.
            custom_tag (str): Custom tag for naming experiment.
        """
        self.test_set = dataset
        self.model = model
        self.country_list = country_list
        self.batch_size = batch_size
        self.seed = seed
        self.prompt = prompt
        self.folder_path = folder_path
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.custom_tag = custom_tag
        self.performance_data = None

    def run_test(self):
        """Runs the model on the given test set, with the given batchsize.
        The results are saved as csv files using the strucutre:
        {output_folder}/Experiments/{model_name}/{prompt_name}/{dateset_name}-{custom_tag}/{date}-{batch_number}.csv
        """
        random.seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for promt, promt_name in zip(self.prompt,self.prompt_name):
            country_tokens = clip.tokenize(list(map(promt, self.country_list)))
            print(f"Running data from dataset: {self.test_set.name}")
            for batch_number, (images, labels) in enumerate(tqdm.tqdm(DataLoader(self.test_set, batch_size=self.batch_size), desc=f"Testing on {self.test_set.name}")):

                images = images.to(device)

                with torch.no_grad():

                    logits_per_image, _ = self.model(images, country_tokens)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                performance_data = pd.DataFrame({
                    'label': labels,
                    'All-Probs': probs.tolist()
                })
                self.__save_data_to_file(performance_data, self.model_name, promt_name, self.test_set.name,
                                        batch_number, self.custom_tag, self.folder_path)

    def __save_data_to_file(self, data: pd.DataFrame, model_name: str, prompt_name: str, dataset_name: str, batch_number: str, custom_tag: str = None, output_dir='./Experiments/'):
        """Saves data from a Pandas DataFrame as a csv file in the way: 
        {model_name}/{prompt_name}/{dateset_name}-{custom_tag}/{date}-{batch_number}.csv
        Args:
            data (pd.DataFrame): The data to save.
            model_name (str): The name of model used for the datageneration.
            prompt_name (str): The name of the promt used in the experiment.
            dataset_name (str): The name of the dataset used in the experiment.
            batch_number (str): The batch number of the generated data.
            custom_tag (str, optional): A custom tag to add to the experiment-name, intended for versioning.
            output_dir (str, optional): The path where the file is saved. Defaults to './Experiments/'.

        Raises:
            TypeError: Data parameter must be a pandas DataFrame
        """
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The 'data' parameter must be a pandas DataFrame.")

        # Create directory structure
        experiment_dir = os.path.join(output_dir, model_name, prompt_name,
                                      f"{dataset_name}-{custom_tag}" if custom_tag else dataset_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Generate file name
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M")
        file_name = f"{timestamp}--batch-{batch_number}.csv"
        file_path = os.path.join(experiment_dir, file_name)

        # Save the DataFrame to a CSV file
        try:
            data.to_csv(file_path, index=False)
            print(f"Model performance saved to {file_path} successfully.")
        except Exception as e:
            print(
                f"Error: Unable to save model performance to {file_path}. {str(e)}")


def run_experiments(DATA_PATH: str, REPO_PATH: str):
    """Runs CLIP experiments for 10 different seeds, over the 3 datasets and 2 prompts
    The experiment results will be saved in '{REPO_PATH}/CLIP_Experiment/clip_results'
    Args:
        DATA_PATH (str): path to the data folder.
        REPO_PATH (str): path to the repo folder
    """
    seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    for seed in seeds:
        datasets = []

        geoguessr = load_dataset.load_data(f'{DATA_PATH}/compressed_dataset/', 0, 5000, False, False, seed)
        geoguessr = geoguessr.head(int(len(geoguessr)*0.2))
        geoguessr = load_dataset.ImageDataset_from_df(geoguessr, preprocessor, name= "geoguessr")
        datasets.append(geoguessr)
        tourist = load_dataset.load_data(f'{DATA_PATH}/tourist/', 0, 5000, False, False, seed)
        tourist = load_dataset.ImageDataset_from_df(tourist, preprocessor, name= "tourist")
        datasets.append(tourist)
        aerialmap = load_dataset.load_data(f'{DATA_PATH}/aerial/', 0, 5000, False, False, seed)
        aerialmap = load_dataset.ImageDataset_from_df(aerialmap, preprocessor, name= "aerial")
        datasets.append(aerialmap)

        default_prompt = lambda x: f"{x}"
        image_from_prompt = lambda x: f"This image shows the country {x}"

        batch_sizes = []

        geoguessr_batch_size = 430
        batch_sizes.append(geoguessr_batch_size)
        tourist_batch_size = 115
        batch_sizes.append(tourist_batch_size)
        aerialmap_batch_size = 14
        batch_sizes.append(aerialmap_batch_size)

        country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv')["Country"].to_list()

        folder_path = f'{REPO_PATH}/CLIP_Experiment'
        model_name = f'clip_results/seed_{seed}'

        default_prompt_name = 'default_prompt'
        image_prompt_name = 'image_from_prompt'

        for i in range(0,len(datasets)):
            test = ModelTester(datasets[i], model, [default_prompt, image_from_prompt], batch_sizes[i], country_list, seed, folder_path, model_name, [default_prompt_name, image_prompt_name] , '')
            test.run_test()


if __name__ == "__main__":
    """Runs the initial CLIP experiments
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path'][args.user]
        REPO_PATH = paths['repo_path'][args.user]
        run_experiments(DATA_PATH, REPO_PATH)