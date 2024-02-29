import sys
sys.path.append('.')
sys.path.append('./scripts')
#----------------------------------------------
import time
import matplotlib.pyplot as plt
import math
import yaml
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sn
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import ast
from scripts import load_dataset, geo_metrics
import os
from models import nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import datetime


class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_dataframe, country_list, region_list, num_folds=10, num_epochs=3, learning_rate=0.001, starting_regional_loss_portion=0.9, regional_loss_decline=0.2, train_dataset_name="Balanced", batch_size=260) -> None:
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.training_dataset_name = train_dataset_name
        self.train_dataframe = train_dataframe
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.country_list = pd.read_csv(country_list)
        self.region_list = pd.read_csv(region_list, delimiter=',')
        self.regional_ordering_index = [8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101, 105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140, 146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163, 166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192, 20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204, 78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72, 196, 13, 50, 55, 59, 62, 65, 69,
                                        86, 88, 92, 94, 113, 115, 142, 168, 172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147, 157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30, 138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57, 71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22, 37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21, 47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10, 18, 36, 49, 63, 110, 114, 118, 141]
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = Regional_Loss(self.country_list)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_count = 0
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.regional_portion = starting_regional_loss_portion
        self.regional_loss_decline = regional_loss_decline
        self.batch_size = batch_size

        # self.region_criterion = Regional_Loss(self.country_list, self.region_list)
        self.writer = SummaryWriter(
            log_dir=f'runs/{self.training_dataset_name}/starting_regional_loss_portion-{starting_regional_loss_portion}/regional_loss_decline-{regional_loss_decline}/{self.timestamp}')
        self.start_training()

    def get_ohe_labels(self, labels):
        ohe_array = []
        for label in labels:
            ref_country_row = self.country_list[self.country_list['Country'] == label]
            ohe_array.append(torch.eye(len(self.country_list))
                             [ref_country_row.index.values[0]])
        return torch.stack(ohe_array, dim=0)

    def train_one_fold(self, train_loader):
        """Train one Epoch of the model. Based on Pytorch Tutorial.

        Args:
            epoch_index (int): Current epoch
            tb_writer (orch.utils.tensorboard.writer.SummaryWriter): Tensorboard wirter

        Returns:
            float: Average loss for the epoch
        """
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_loader):

            # Zero gradients for every batch!
            self.optimizer.zero_grad()
            # Every data instance is an input + label pair
            inputs, labels = data
            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            regional_loss, country_loss = self.criterion(outputs, labels)
            loss = self.calculate_weighted_loss(regional_loss, country_loss)

            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            self.batch_count += 1

            # print(f"batch {i} loss: {loss}")
            # if i % 10 == 9:
            #     last_loss = running_loss/            
            # print(f"batch {i} loss: {loss}")
            # if i % 10 == 9:
            #     last_lo
        fold_mean_loss = running_loss / len(train_loader)
        return fold_mean_loss

    def validate(self, epoch_index, fold_index, validation_dataset):
        inputs, targets = validation_dataset[:]
        # self.createConfusionMatrix(true_countries,predicted_countries,"Validation Confusion Matrix",epoch_index*self.num_folds + fold_index)
        outputs = self.model(inputs)
        avg_validation_region_accuracy = self.criterion.claculate_region_accuracy(
            outputs, targets)
        avg_validation_accuracy = self.criterion.calculate_country_accuracy(
            outputs, targets)
        # avg_validation_loss = validation_loss / len(validation_loader)

        print('Epoch {} Fold {} Validation Accuracy: {}, Validation Regional Accuracy: {}'.format(
            epoch_index + 1, fold_index + 1, avg_validation_accuracy, avg_validation_region_accuracy))
        self.writer.add_scalar(
            'Validation Accuracy', avg_validation_accuracy, epoch_index*self.num_folds + fold_index)
        self.writer.add_scalar('Validation Regional Accuracy',
                               avg_validation_region_accuracy, epoch_index*self.num_folds + fold_index)

        # self.writer.add_scalar('Validation Loss', avg_validation_loss, epoch_index*self.num_folds + fold_index)

        # torch.save(self.model.state_dict(),f'saved_models/model_{self.training_dataset_name}_epoch_{epoch_index}_batch_{i}')

    def start_training(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_size = math.floor(
            len(self.train_dataframe.index) / self.num_folds)

        for epoch_index in range(self.num_epochs):
            if epoch_index > 0:
                self.regional_portion = self.regional_loss_decline * self.regional_portion
            for fold_index in range(self.num_folds):
                self.model.train()  # Set the model to training mode
                fold_validation_df = pd.DataFrame()
                drop_indices = []
                if fold_index == self.num_folds - 1:
                    fold_validation_df = self.train_dataframe.iloc[fold_index*validation_size:]
                    drop_indices = range(
                        fold_index*validation_size, len(self.train_dataframe.index))
                else:
                    fold_validation_df = self.train_dataframe.iloc[fold_index*validation_size: (
                        fold_index+1)*validation_size]
                    drop_indices = range(
                        fold_index*validation_size, (fold_index+1)*validation_size)

                fold_training_df = self.train_dataframe.drop(drop_indices)
                train_dataset = load_dataset.EmbeddingDataset_from_df(
                    fold_training_df, 'train')
                train_loader = DataLoader(
                    train_dataset, batch_size=250, shuffle=False)
                avg_training_loss = self.train_one_fold(train_loader)
                self.writer.add_scalar(
                    'Training Loss', avg_training_loss, epoch_index*self.num_folds + fold_index)

                self.model.eval()  # Set the model to evaluation mode

                validation_dataset = load_dataset.EmbeddingDataset_from_df(
                    fold_validation_df, 'validation')
                # validation_loader = DataLoader(validation_dataset, shuffle=False)

                self.validate(epoch_index, fold_index, validation_dataset)

                # self.writer.add_scalars('Training vs. Validation Loss',
                #         { 'Training' : avg_training_loss, 'Validation' : avg_validation_loss },
                #         epoch_index*self.num_folds + fold_index + 1)
                # print(f"Epoch [{epoch_index+1}/{self.num_epochs}] - Fold [{fold_index+1}/{self.num_folds}] - Average Train Loss: {avg_training_loss:.4f} - Val Loss: {avg_validation_loss:.4f}")
                self.writer.flush()
            torch.save(self.model.state_dict,
                       f'saved_models/model_{self.training_dataset_name}_{timestamp}_{epoch_index+1}')

    def test_model(self, test_loader):
        # test_loss = 0.0
        test_accuracy = 0.0
        test_region_accuracy = 0.0
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predicted_countries = []
            true_countries = []
            for test_inputs, test_labels in test_loader:
                test_outputs = self.model(test_inputs)
                test_outputs = test_outputs.to("cpu")
                test_accuracy += geo_metrics.calculate_country_accuracy(
                    self.country_list, test_outputs, test_labels)
                test_region_accuracy += geo_metrics.calculate_region_accuracy(
                    self.country_list, test_outputs, test_labels)
                # test_regional_loss, test_country_loss = self.criterion(test_outputs, test_labels)
                # test_loss += self.calculate_weighted_loss(test_regional_loss,test_country_loss).item()
                predicted_country_index = np.argmax(
                    test_outputs, axis=1).item()
                predicted_countries.append(predicted_country_index)
                true_country_index = self.country_list.index[self.country_list['Country'] == test_labels[0]].tolist()[
                    0]
                true_countries.append(true_country_index)
            self.createConfusionMatrix(
                true_countries, predicted_countries, "Confusion Matrix", None)
        avg_test_region_accuracy = test_region_accuracy / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)
        # avg_test_loss = test_loss / len(test_loader)
        self.writer.add_scalar('Test Accuracy', avg_test_accuracy)
        self.writer.add_scalar('Test Regional Accuracy',
                               avg_test_region_accuracy)
        # self.writer.add_scalar('Test Loss', avg_test_loss)
        print('Training Dataset {} Test Accuracy: {}, Test Regional Accuracy: {}'.format(
            self.training_dataset_name, avg_test_accuracy, avg_test_region_accuracy))

        # print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    def createConfusionMatrix(self, true_countries, predicted_countries, figure_label, index):
        """
        Creates and visualizes the confusion matrix for country and region predictions.

        Args:
            true_countries (list): List of true country labels.
            predicted_countries (list): List of predicted country labels.
            figure_label (str): Label for the generated figures.
            index (int): Index for adding figures to the writer.

        Returns:
            None
        """
        # constant for classes
        classes = self.country_list['Country']
        np_classes = np.array(classes)

        # Build country confusion matrix
        cf_matrix = confusion_matrix(
            true_countries, predicted_countries, labels=range(0, 211))
        ordered_index = np.argsort(-cf_matrix.diagonal())
        ordered_matrix = cf_matrix[ordered_index][:, ordered_index]

        regionally_ordered_matrix = cf_matrix[self.regional_ordering_index][:,
                                                                            self.regional_ordering_index]

        ordered_classes = np_classes[ordered_index]
        regionally_ordered_classes = np_classes[self.regional_ordering_index]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
        ordered_df_cm = pd.DataFrame(
            ordered_matrix, index=ordered_classes, columns=ordered_classes)
        regionally_ordered_df_cm = pd.DataFrame(
            regionally_ordered_matrix, index=regionally_ordered_classes, columns=regionally_ordered_classes)

        np_regions = np.sort(
            np.array(list(set(self.country_list['Intermediate Region Name']))))

        # Build region confusion matrix
        true_regions = []
        predicted_regions = []
        for i in range(0, len(true_countries)):
            true_regions.append(ast.literal_eval(
                self.country_list.iloc[true_countries[i]]["One Hot Region"]).index(1))
            predicted_regions.append(ast.literal_eval(
                self.country_list.iloc[predicted_countries[i]]["One Hot Region"]).index(1))

        regions_cf_matrix = confusion_matrix(
            true_regions, predicted_regions, labels=range(0, len(np_regions)))
        regions_ordered_index = np.argsort(-regions_cf_matrix.diagonal())
        regions_ordered_matrix = regions_cf_matrix[regions_ordered_index][:,
                                                                          regions_ordered_index]

        ordered_regions = np_regions[regions_ordered_index]

        regions_df_cm = pd.DataFrame(
            regions_cf_matrix, index=np_regions, columns=np_regions)
        regions_ordered_df_cm = pd.DataFrame(
            regions_ordered_matrix, index=ordered_regions, columns=ordered_regions)
        plt.figure(1, figsize=(120, 70))
        figure = sn.heatmap(df_cm, cmap=sn.cubehelix_palette(
            as_cmap=True)).get_figure()
        plt.figure(2, figsize=(120, 70))
        ordered_figure = sn.heatmap(
            ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(3, figsize=(120, 70))
        regionally_ordered_figure = sn.heatmap(
            regionally_ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(4, figsize=(120, 70))
        regions_figure = sn.heatmap(
            regions_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(5, figsize=(120, 70))
        regions_ordered_figure = sn.heatmap(
            regions_ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        if (index == None):
            self.writer.add_figure(f"{figure_label}-unordered", figure)
            self.writer.add_figure(f"{figure_label}-ordered", ordered_figure)
            self.writer.add_figure(
                f"{figure_label}-regionally_ordered", regionally_ordered_figure)
            self.writer.add_figure(f"{figure_label}-regions", regions_figure)
            self.writer.add_figure(
                f"{figure_label}-regions_ordered", regions_ordered_figure)
        else:
            self.writer.add_figure(f"{figure_label}-unordered", figure, index)
            self.writer.add_figure(
                f"{figure_label}-ordered", ordered_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regionally_ordered", regionally_ordered_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regions", regions_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regions_ordered", regions_ordered_figure, index)
        return

    def calculate_weighted_loss(self, regional_loss_mean, country_loss_mean):
        loss = torch.tensor([], dtype=torch.float32, device=self.device)
        loss.requires_grad = True
        total_loss = (self.regional_portion * regional_loss_mean) + \
            ((1-self.regional_portion) * country_loss_mean)

        loss = torch.cat((loss, total_loss.unsqueeze(0)), dim=0)

        self.writer.add_scalar('Batch Loss', loss.item(), self.batch_count)
        self.writer.add_scalar(
            'Unweighted Regional Batch Loss', regional_loss_mean.item(), self.batch_count)
        self.writer.add_scalar(
            'Unweighted Country Batch Loss', country_loss_mean.item(), self.batch_count)
        self.writer.add_scalar('Weighted Regional Batch Loss',
                               self.regional_portion * regional_loss_mean.item(), self.batch_count)
        self.writer.add_scalar('Weighted Country Batch Loss', (
            1-self.regional_portion) * country_loss_mean.item(), self.batch_count)
        return loss


class Regional_Loss(torch.nn.Module):
    def __init__(self, country_list):
        """
        Initializes the Regional_Loss object.

        Args:
            country_list (pandas.DataFrame): A DataFrame containing the infromation of country_list_region_and_continent.csv.

        Attributes:
            device (torch.device): The device (CPU or GPU) on which the model will be trained.
            country_list (pandas.DataFrame): The input country list DataFrame.
            country_dict (dict): A dictionary mapping country names to indices in the country_list.
            region_indices (list): A list of lists containing the indices of countries in each region.
            regions (torch.Tensor): A tensor containing the region indices for each country.
            regions_dict (dict): A dictionary mapping country indices to region indices.
            selective_sum_operator (torch.Tensor): A tensor used for selective sum operation.

        """

        super(Regional_Loss, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.country_list = country_list
        self.country_dict = {country: index for index,
                             country in enumerate(self.country_list["Country"])}
        self.region_indices = country_list.groupby('Intermediate Region Name')[
            'Country'].apply(lambda x: list(x.index)).to_list()
        self.regions = self.country_list["One Hot Region"].apply(lambda x: torch.argmax(
            torch.tensor(ast.literal_eval(x), dtype=torch.float32, device=self.device)))
        self.regions_dict = {index: region for index,
                             region in enumerate(self.regions)}
        self.selective_sum_operator = torch.zeros(len(self.region_indices), len(
            self.country_list), dtype=torch.float32, device=self.device)
        for i, indices in enumerate(self.region_indices):
            self.selective_sum_operator[i, indices] = 1
        self.selective_sum_operator

    def forward(self, outputs, targets):
        """
        Forward pass of the model.

        Args:
            outputs (torch.Tensor): The output tensor from the model.
            targets (list): The list of target values.

        Returns:
            tuple: A tuple containing the mean region loss and mean country loss.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the indices of all targets, corrseponding to the region index of the one hot encoded region vector
        target_region_enc = torch.tensor(
            [self.regions_dict[target] for target in target_countries_idxs], device=self.device)
        # sum the outputs of the countries in each region
        region_outputs = torch.matmul(
            outputs, self.selective_sum_operator.transpose(0, 1))

        country_loss = F.cross_entropy(outputs, torch.tensor(
            target_countries_idxs, device=self.device))
        region_loss = F.cross_entropy(region_outputs, target_region_enc)

        return region_loss.mean(), country_loss.mean()

    def claculate_region_accuracy(self, outputs, targets):
        """
        Calculates the accuracy of region predictions.

        Args:
            outputs (torch.Tensor): The output tensor from the model.
            targets (list): The list of target countries.

        Returns:
            torch.Tensor: The mean accuracy of region predictions.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the indices of all targets, corrseponding to the region index of the one hot encoded region vector
        target_region_enc = torch.tensor(
            [self.regions_dict[target] for target in target_countries_idxs], device=self.device)
        # sum the outputs of the countries in each region
        region_outputs = torch.matmul(
            outputs, self.selective_sum_operator.transpose(0, 1))

        region_predictions = torch.argmax(region_outputs, axis=1)
        # calculate the accuracy of the region predictions
        return torch.mean((region_predictions == target_region_enc).float())

    def calculate_country_accuracy(self, outputs, targets):
        """
        Calculates the accuracy of country predictions.

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (list): The target country labels.

        Returns:
            torch.Tensor: The mean accuracy of country predictions.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the index of the preidcted country
        country_predictions = torch.argmax(outputs, axis=1)
        # calculate the accuracy of the country predictions
        return torch.mean((country_predictions == torch.tensor(target_countries_idxs, device=self.device)).float())


def create_and_train_model(REPO_PATH: str):
    """
    Creates and trains a model using the specified repository path.

    Args:
        REPO_PATH (str): The path to the repository.

    Returns:
        None
    """

    country_list = f'{REPO_PATH}/data_finding/country_list_region.csv'
    region_list = f'{REPO_PATH}/data_finding/UNSD_Methodology.csv'

    testing_directory = f'{REPO_PATH}/Embeddings/Testing'
    testing_file_list = [file for file in os.listdir(testing_directory)]
    testing_dfs = []
    # Iterate through the files, read them as DataFrames, and append to the list4
    for file in testing_file_list:
        file_path = os.path.join(testing_directory, file)
        df = pd.read_csv(file_path)
        testing_dfs.append(df)
    testing_combined_df = pd.concat(testing_dfs, ignore_index=True)
    test_dataset = load_dataset.EmbeddingDataset_from_df(
        testing_combined_df, "test")
    test_loader = DataLoader(test_dataset, shuffle=False)

    training_datasets = [
        'Unbalanced',
        'Weakly_Balanced',
        'Weakly_Balanced_Replace',
        'Strongly_Balanced',
        'Strongly_Balanced_Replace'
    ]

    for elem in training_datasets:
        # Directory containing CSV files
        training_directory = f'{REPO_PATH}/Embeddings/Training/{elem}'
        # Get a list of all filenames in each directory
        training_file_list = [file for file in os.listdir(training_directory)]
        # Initialize an empty list to store DataFrames
        training_dfs = []
        # Iterate through the files, read them as DataFrames, and append to the list
        for file in training_file_list:
            file_path = os.path.join(training_directory, file)
            df = pd.read_csv(file_path)
            training_dfs.append(df)
        # Concatenate all DataFrames in the list into a single DataFrame
        training_combined_df = pd.concat(training_dfs, ignore_index=True)
        hyperparameters = [
            {'starting_regional_loss_portion': 0.0,
             'regional_loss_decline': 1.0},
            {'starting_regional_loss_portion': 0.25,
             'regional_loss_decline': 1.0},
            {'starting_regional_loss_portion': 0.8,
             'regional_loss_decline': 0.5}
        ]
        bs = 260
        if elem == 'Strongly_Balanced' or elem == 'Strongly_Balanced_Replace':
            bs = 110
        for i in range(0, len(hyperparameters)):
            model = nn.FinetunedClip()
            trained_model = ModelTrainer(model, training_combined_df, country_list, region_list,
                                         batch_size=bs, num_epochs=15,
                                         starting_regional_loss_portion=hyperparameters[i]['starting_regional_loss_portion'], regional_loss_decline=hyperparameters[i]['regional_loss_decline'], train_dataset_name=elem)
            trained_model.test_model(test_loader)
    print("END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    # parser.add_argument('--training_dataset_name', metavar='str', required=True, help='the name of the dataset')
    # parser.add_argument('--starting_regional_loss_portion', metavar='float', required=True, help='the starting regional loss portion')
    # parser.add_argument('--regional_loss_decline', metavar='float', required=True, help='the factor with which the regional loss portion is multiplied each epoch')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        create_and_train_model(REPO_PATH)
