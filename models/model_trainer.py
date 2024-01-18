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

class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_loader, val_loader, country_list, region_list, num_epochs = 10, learning_rate = 0.001) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.country_list = pd.read_csv(country_list)
        self.region_list = pd.read_csv(region_list,delimiter=',')
        self.criterion = Regional_Loss(self.country_list, self.region_list)
        self.writer = SummaryWriter()
        self.start_training()

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
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self.model(inputs)
            # Compute the loss and its gradients
            loss = self.criterion(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            if i % 50 == 49:
                last_loss= (running_loss / 50) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

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
        for output, target in zip(outputs,targets):
            ref_country = target
            ref_country_row = self.country_list[self.country_list['Country'] == ref_country]
            if ref_country_row.empty:
                print(f"Country {ref_country} not found in country list")
                continue
            ref_alph2code = ref_country_row['Alpha2Code'].values[0]
            ref_region_row = self.region_list[self.region_list['ISO-alpha2 Code'] == ref_alph2code]
            if ref_region_row.empty:
                print(f"Alpha2Code {ref_alph2code} not found in region list")
                continue
            ref_region = ref_region_row['Intermediate Region Name'].values[0]
            pred_country = self.country_list['Country'].iloc[torch.argmax(output).item()]

            ohe = torch.eye(len(output))[ref_country_row.index.values[0]]
            cross_entropy_loss = F.cross_entropy(output, ohe)

            pred_country_row = self.country_list[self.country_list['Country'] == pred_country]
            if pred_country_row.empty:
                print(f"Country {pred_country} not found in country list")
                continue
            pred_alpha2code = pred_country_row['Alpha2Code'].values[0]
            pred_region_row = self.region_list[self.region_list['ISO-alpha2 Code'] == pred_alpha2code]
            if pred_region_row.empty:
                print(f"Alpha2Code {pred_alpha2code} not found in region list")
                continue
            pred_region = pred_region_row['Intermediate Region Name'].values[0]

            #if ref_country == pred_country:
            #    loss.append(0.)
            #elif ref_region == pred_region:
            #    loss.append(0.5)
            #else:
            #    loss.append(1.)
            if ref_region == pred_region:
                cross_entropy_loss -= 1
            loss.append(cross_entropy_loss)

        loss = torch.tensor(loss)
        loss.requires_grad = True

        return loss.mean()

# Directory containing CSV files
directory = '/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/Embeddings/Image'
country_list = "/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list.csv"
region_list = "/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/UNSD_Methodology.csv"


# Get a list of filenames that start with "geoguessr" and end with ".csv"
file_list = [file for file in os.listdir(directory) if file.startswith('geoguessr') and file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in file_list:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

train, test = sklearn.model_selection.train_test_split(combined_df, test_size=0.2, random_state=1234, shuffle=True)
val, test = sklearn.model_selection.train_test_split(test, test_size=0.5, random_state=1234, shuffle=True)

train_dataset = load_dataset.EmbeddingDataset_from_df(train, "train")
val_dataset = load_dataset.EmbeddingDataset_from_df(val, "val")
test_dataset = load_dataset.EmbeddingDataset_from_df(test, "test")

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)



model = nn.FinetunedClip()
trainer = ModelTrainer(model, train_loader, val_loader, country_list, region_list)
trainer.test_model(test_loader)
print("END")
