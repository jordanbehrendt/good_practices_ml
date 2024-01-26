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
from scripts import load_dataset, geo_metrics
import ast
import sklearn.model_selection
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import yaml
 

class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_loader, val_loader, country_list, region_list, num_epochs = 10, learning_rate = 0.001, region_loss_portion = 0.25, train_dataset_name="default") -> None:
        self.model = model
        self.training_dataset_name = train_dataset_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.country_list = pd.read_csv(country_list)
        self.region_list = pd.read_csv(region_list,delimiter=',')
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = Regional_Loss(self.country_list, region_loss_portion)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        #self.region_criterion = Regional_Loss(self.country_list, self.region_list)
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
            # Zero gradients for every batch!
            self.optimizer.zero_grad()
            # Every data instance is an input + label pair
            inputs, labels = data
            # Make predictions for this batch
            outputs = self.model(inputs)
            # Compute the loss and its gradients
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            print(f"batch {i} loss: {loss}")
            if i % 10 == 9:
                last_loss = running_loss/10
                self.validate(epoch_index, i, running_loss)
                running_loss = 0.0

        return last_loss

    def validate(self, epoch_index, i, running_loss):
        last_loss= (running_loss / 100) # loss per batch
        tb_x = epoch_index * len(self.train_loader) + i + 1
        self.writer.add_scalar('Loss/train', last_loss, tb_x)
        val_loss = 0.0
        val_accuracy= 0.0
        val_region_accuracy = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in self.val_loader:
                val_outputs = self.model(val_inputs)

                val_accuracy += geo_metrics.calculate_country_accuracy(self.country_list, val_outputs, val_labels)

                val_region_accuracy += geo_metrics.calculate_region_accuracy(self.country_list, val_outputs, val_labels)
                val_loss += self.criterion(val_outputs, val_labels).item()
        avg_val_region_accuracy = val_region_accuracy / len(self.val_loader)
        avg_vaccuracy = val_accuracy / len(self.val_loader)
        avg_val_loss = val_loss / len(self.val_loader)
        print('  batch {} validation accuracy: {}, regional accuracy: {}'.format(i + 1, avg_vaccuracy, avg_val_region_accuracy))
        self.writer.add_scalar('Accuracy/Validation', avg_vaccuracy, epoch_index*104+i)
        self.writer.add_scalar('Accuracy/Region Validation', avg_val_region_accuracy, epoch_index*104+i)
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch_index*104+i)

        torch.save(self.model.state_dict(),f'saved_models/model_{self.training_dataset_name}_epoch_{epoch_index}_batch_{i}')
        
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
    def __init__(self, country_list, region_portion):
        super(Regional_Loss, self).__init__()
        self.country_list = country_list
        self.region_country_dict =  country_list.groupby('Intermediate Region Name')['Country'].apply(lambda x: list(x.index))
        self.region_portion = region_portion
        self.country_portion = 1 - region_portion


    def forward(self, outputs, targets):
        loss = torch.tensor([], dtype=torch.float32)
        loss.requires_grad = True
        for output, target in zip(outputs,targets):
            ref_country = target
            ref_country_row = self.country_list[self.country_list['Country'] == ref_country]
            if ref_country_row.empty:
                print(f"Country {ref_country} not found in country list")
                continue

            target_region_enc = torch.tensor(ast.literal_eval(ref_country_row['One Hot Region'].values[0]), dtype=torch.float32)
            target_country_enc = torch.tensor(ast.literal_eval(ref_country_row['One Hot Country'].values[0]), dtype=torch.float32)
            output_region_enc = torch.tensor([], dtype=torch.float32)
            
            for region_index in self.region_country_dict:
                sum_output = torch.sum(output[region_index], dim=0)
                output_region_enc = torch.cat((output_region_enc, sum_output.unsqueeze(0)), dim=0)

            region_cross_entropy_loss = F.cross_entropy(output_region_enc, torch.argmax(target_region_enc))
            country_cross_entropy_loss = F.cross_entropy(output, torch.argmax(target_country_enc))

            total_loss = (self.region_portion * region_cross_entropy_loss) + (self.country_portion * country_cross_entropy_loss)
            loss = torch.cat((loss, total_loss.unsqueeze(0)), dim=0)

        return loss.mean()




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
    trainer = ModelTrainer(model, train_loader, val_loader, country_list, region_list, train_dataset_name=training_dataset_name)
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

