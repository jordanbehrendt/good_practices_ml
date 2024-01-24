import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import datetime
import sys
sys.path.append('.')
from models import nn
import os
sys.path.append('./scripts')
from scripts import load_dataset
import sklearn.model_selection
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import yaml

class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_loader, val_loader, country_list, region_list, num_epochs = 10, learning_rate = 0.003) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.country_list = pd.read_csv(country_list)
        self.region_list = pd.read_csv(region_list,delimiter=',')
        self.region_criterion = Regional_Loss(self.country_list, self.region_list)
        self.writer = SummaryWriter()
        self.start_training()

    def get_ohe_labels(self, labels):
        ohe_array = []
        for label in labels:
            ref_country_row = self.country_list[self.country_list['Country'] == label]
            ohe_array.append(torch.eye(len(self.country_list))[ref_country_row.index.values[0]])
        return torch.stack(ohe_array, dim=0)

    def train_one_epoch(self, epoch_index):
        """Train one Epoch of the model. Based on Pytorch Tutorial.

        Args:
            epoch_index (int): Current epoch
            tb_writer (orch.utils.tensorboard.writer.SummaryWriter): Tensorboard wirter

        Returns:
            float: Average loss for the epoch
        """
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # Make predictions for this batch
            outputs = self.model(inputs)
            # Compute the loss and its gradients
            ohe_list = self.get_ohe_labels(labels)
            loss = self.criterion(outputs, ohe_list)
            accuracy, region_accuracy = self.region_criterion(outputs, labels)
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            #print(f"batch {i} loss: {loss}, accuracy: {accuracy}, region_accuracy: {region_accuracy}")
            if i % 100 == 99:
                last_loss= (running_loss / 100) # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                val_loss = 0.0
                with torch.no_grad():
                    for i, data in enumerate(self.val_loader):
                        val_inputs, val_labels = data
                        val_outputs = self.model(val_inputs)
                        val_accuracy, val_region_accuracy = self.region_criterion(val_outputs, val_labels)
                        # val_loss += self.criterion(val_outputs, val_targets).item()
                avg_vaccuracy = val_accuracy / len(self.val_loader)
                print('  batch {} validation accuracy: {}'.format(i + 1, avg_vaccuracy))

        return sum(last_loss)/len(last_loss)

    def start_training(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            avg_loss = self.train_one_epoch(epoch)

            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in self.val_loader:
                    val_outputs = self.model(val_inputs)
                    val_loss += self.criterion(val_outputs, val_targets).item()
            avg_vloss = val_loss / len(self.val_loader)
            self.writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Train Loss: {running_loss/len(self.train_loader):.4f} - Val Loss: {val_loss/len(self.val_loader):.4f}")
            self.writer.flush()
            torch.save(self.model.state_dict,f'model_{timestamp}_{epoch+1}')

    def test_model(self, test_loader):
        test_loss = 0.0
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_outputs = self.model(test_inputs)
                test_loss += self.criterion(test_outputs, test_targets).item()
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")

class Regional_Loss(torch.nn.Module):
    def __init__(self, country_list, region_list):
        super(Regional_Loss, self).__init__()
        self.country_list = country_list
        self.region_list = region_list

    def forward(self, outputs, targets):
        loss = []
        accuracies = []
        region_accuracies = []
        for output, target in zip(outputs,targets):
            ref_country = target
            ref_country_row = self.country_list[self.country_list['Country'] == ref_country]
            if ref_country_row.empty:
                print(f"Country {ref_country} not found in country list")
                continue
            #ref_alph2code = ref_country_row['Alpha2Code'].values[0]
            #ref_region_row = self.region_list[self.region_list['ISO-alpha2 Code'] == ref_alph2code]
            #if ref_region_row.empty:
            #    print(f"Alpha2Code {ref_alph2code} not found in region list")
            #    continue
            #ref_region = ref_region_row['Intermediate Region Name'].values[0]
            ref_region = ref_country_row['Intermediate Region Name'].values[0]
            pred_country = self.country_list['Country'].iloc[torch.argmax(output).item()]

            #ohe = torch.eye(len(output))[ref_country_row.index.values[0]]
            #cross_entropy_loss = F.cross_entropy(output, ohe)

            pred_country_row = self.country_list[self.country_list['Country'] == pred_country]
            if pred_country_row.empty:
                print(f"Country {pred_country} not found in country list")
                continue
            #pred_alpha2code = pred_country_row['Alpha2Code'].values[0]
            #pred_region_row = self.region_list[self.region_list['ISO-alpha2 Code'] == pred_alpha2code]
            #if pred_region_row.empty:
            #    print(f"Alpha2Code {pred_alpha2code} not found in region list")
            #    continue
            #pred_region = pred_region_row['Intermediate Region Name'].values[0]
            pred_region = pred_country_row['Intermediate Region Name'].values[0]

            if ref_country == pred_country:
                accuracies.append(1)
                region_accuracies.append(1)
            elif ref_region == pred_region:
                accuracies.append(0)
                region_accuracies.append(0.5)
            else:
                accuracies.append(0)
                region_accuracies.append(0)
            #loss.append(cross_entropy_loss)

        #loss = torch.tensor(loss)
        #loss.requires_grad = True

        #return loss.mean(), float(np.mean(accuracies)), float(np.mean(region_accuracies))
        return float(np.mean(accuracies)), float(np.mean(region_accuracies))




def create_and_train_model(REPO_PATH: str, training_dataset_name: str):
    # Directory containing CSV files
    training_directory = f'{REPO_PATH}/Embeddings/Training/{training_dataset_name}'
    validation_directory = f'{REPO_PATH}/Embeddings/Validation/{training_dataset_name}'
    testing_directory = f'{REPO_PATH}/Embeddings/Testing'
    country_list = f'{REPO_PATH}/data_finding/country_list_region.csv'
    region_list = f'{REPO_PATH}/data_finding/UNSD_Methodology.csv'

    # Get a list of all filenames in each directory
    training_file_list = [file for file in os.listdir(training_directory)]
    validation_file_list = [file for file in os.listdir(validation_directory)]
    testing_file_list = [file for file in os.listdir(testing_directory)]

    # Initialize an empty list to store DataFrames
    training_dfs = []
    validation_dfs = []
    testing_dfs = []


    # Iterate through the files, read them as DataFrames, and append to the list
    for file in training_file_list:
        file_path = os.path.join(training_directory, file)
        df = pd.read_csv(file_path)
        training_dfs.append(df)
    # Iterate through the files, read them as DataFrames, and append to the list
    for file in validation_file_list:
        file_path = os.path.join(validation_directory, file)
        df = pd.read_csv(file_path)
        validation_dfs.append(df)
    # Iterate through the files, read them as DataFrames, and append to the list
    for file in testing_file_list:
        file_path = os.path.join(testing_directory, file)
        df = pd.read_csv(file_path)
        testing_dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    training_combined_df = pd.concat(training_dfs, ignore_index=True)
    validation_combined_df = pd.concat(validation_dfs, ignore_index=True)
    testing_combined_df = pd.concat(testing_dfs, ignore_index=True)


    train_dataset = load_dataset.EmbeddingDataset_from_df(training_combined_df, "train")
    val_dataset = load_dataset.EmbeddingDataset_from_df(validation_combined_df, "val")
    test_dataset = load_dataset.EmbeddingDataset_from_df(testing_combined_df, "test")

    train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=250, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=250, shuffle=True)



    model = nn.FinetunedClip()
    trainer = ModelTrainer(model, train_loader, val_loader, country_list, region_list)
    trainer.test_model(test_loader)
    print("END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    parser.add_argument('--training_dataset_name', metavar='str', required=True, help='the name of the dataset')
    args = parser.parse_args()


    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        create_and_train_model(REPO_PATH, args.training_dataset_name)

